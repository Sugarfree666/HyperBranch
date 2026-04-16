from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from ..config import ReasoningConfig, RetrievalConfig
from ..data.loaders import DatasetBundle
from ..models import EvidenceItem, HyperedgeCandidate, RetrievalControlState, TaskFrame, ThoughtState, VectorMatch
from ..utils import content_tokens, lexical_overlap_score, normalize_label, short_text


GENERIC_TOPIC_ENTITY_TOKENS = {
    "area",
    "center",
    "centre",
    "city",
    "college",
    "community",
    "concept",
    "county",
    "district",
    "entity",
    "farm",
    "group",
    "location",
    "network",
    "organization",
    "person",
    "place",
    "program",
    "project",
    "region",
    "school",
    "state",
    "system",
    "team",
}
TEMPORAL_TOPIC_ENTITY_TOKENS = {
    "centuries",
    "century",
    "day",
    "days",
    "decade",
    "decades",
    "era",
    "eras",
    "hundred",
    "month",
    "months",
    "season",
    "seasons",
    "time",
    "times",
    "year",
    "years",
}


class EvidenceRetriever:
    def __init__(
        self,
        dataset: DatasetBundle,
        embedder: Any,
        config: RetrievalConfig,
        logger: logging.Logger,
        reasoning_config: ReasoningConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.embedder = embedder
        self.config = config
        self.reasoning_config = reasoning_config
        self.logger = logger
        self._cache: dict[str, list[EvidenceItem]] = {}
        self._entity_labels = [node_id for node_id, node in dataset.graph.nodes.items() if node.role == "entity"]
        self._hyperedge_labels = [node_id for node_id, node in dataset.graph.nodes.items() if node.role == "hyperedge"]
        self._entity_row_id_by_label = self._build_row_lookup(dataset.entity_store)
        self._hyperedge_row_id_by_label = self._build_row_lookup(dataset.hyperedge_store)
        self._entity_labels_by_normalized = self._build_normalized_label_lookup(self._entity_labels)

    def anchor_task_frame(self, question: str, task_frame: TaskFrame) -> dict[str, Any]:
        grounding = self._ground_task_frame_entities(question, task_frame)
        grounded_topic_entities = list(grounding.get("grounded_topic_entities", []))
        grounding_metadata = dict(grounding.get("metadata", {}))
        if grounded_topic_entities or grounding_metadata:
            task_frame.apply_entity_grounding(grounded_topic_entities, grounding_metadata)

        query_texts = self._dedupe_texts(
            [
                question,
                *task_frame.topic_entities,
                *task_frame.hard_constraints,
                task_frame.relation_intent,
                task_frame.answer_type_hint,
            ]
        )
        entity_matches = list(grounding.get("entity_matches", []))
        initial_entity_ids = [match.label for match in entity_matches[: self.config.entity_top_k]]

        hyperedge_candidates = self._rank_hyperedges(
            query_texts=query_texts,
            topic_entities=task_frame.topic_entities,
            hard_constraints=task_frame.hard_constraints,
            relation_intent=task_frame.relation_intent,
            relation_skeleton=task_frame.relation_skeleton,
            initial_entity_ids=initial_entity_ids,
            connected_entity_ids=initial_entity_ids,
            branch_kind="anchor",
            focus_texts=[],
            exclude_hyperedge_ids=set(),
            existing_hyperedge_ids=set(),
        )
        initial_hyperedge_ids = [
            candidate.hyperedge_id for candidate in hyperedge_candidates[: self.config.hyperedge_top_k]
        ]
        self.logger.info(
            "Anchored task frame with %s grounded topics, %s initial entities and %s initial hyperedges",
            len(task_frame.topic_entities),
            len(initial_entity_ids),
            len(initial_hyperedge_ids),
        )
        return {
            "entity_matches": entity_matches,
            "initial_entity_ids": initial_entity_ids,
            "initial_hyperedge_ids": initial_hyperedge_ids,
            "initial_hyperedge_candidates": hyperedge_candidates[: self.config.hyperedge_top_k],
            "entity_grounding": grounding_metadata,
            "grounded_topic_entities": list(task_frame.topic_entities),
        }

    def retrieve_branch_candidates(
        self,
        question: str,
        task_frame: TaskFrame,
        branch_kind: str,
        control_state: RetrievalControlState,
        evidence_subgraph: dict[str, Any] | None = None,
        exclude_hyperedge_ids: set[str] | None = None,
        *,
        channel_id: str = "",
    ) -> list[HyperedgeCandidate]:
        evidence_subgraph = evidence_subgraph or {}
        connected_entity_ids = [
            str(item).strip()
            for item in evidence_subgraph.get("expansion_frontier_entity_ids", [])
            if str(item).strip()
        ]
        explored_entity_ids = {
            str(item).strip()
            for item in evidence_subgraph.get("explored_entity_ids", [])
            if str(item).strip()
        }
        existing_hyperedge_ids = {
            str(item).strip() for item in evidence_subgraph.get("hyperedge_ids", []) if str(item).strip()
        }
        exclude = set(exclude_hyperedge_ids or set())
        query_texts = self._branch_query_texts(question, task_frame, branch_kind, control_state)
        channel_key = f"{channel_id}::{branch_kind}" if channel_id else branch_kind
        control_state.branch_queries[channel_key] = list(query_texts)
        control_state.branch_queries[branch_kind] = list(query_texts)

        candidates = self._rank_hyperedges(
            query_texts=query_texts,
            topic_entities=task_frame.topic_entities,
            hard_constraints=task_frame.hard_constraints,
            relation_intent=task_frame.relation_intent,
            relation_skeleton=task_frame.relation_skeleton,
            initial_entity_ids=list(task_frame.initial_entity_ids),
            connected_entity_ids=connected_entity_ids,
            branch_kind=branch_kind,
            focus_texts=control_state.current_focus(),
            exclude_hyperedge_ids=exclude,
            existing_hyperedge_ids=existing_hyperedge_ids,
            fallback_to_initial_entities=not connected_entity_ids and not explored_entity_ids and not existing_hyperedge_ids,
        )
        filtered = self._apply_control_filter(branch_kind, candidates, control_state)
        annotated: list[HyperedgeCandidate] = []
        for candidate in filtered[: self.config.branch_candidate_pool]:
            if channel_id:
                candidate.channel_id = channel_id
                if channel_id not in candidate.supporting_channel_ids:
                    candidate.supporting_channel_ids.append(channel_id)
                candidate.notes = [f"channel:{normalize_label(channel_id)}", *candidate.notes]
            annotated.append(candidate)
        self.logger.info(
            "Retrieved %s branch candidates for %s%s with queries=%s and weights=%s",
            len(annotated),
            branch_kind,
            f" on {normalize_label(channel_id)}" if channel_id else "",
            query_texts,
            control_state.branch_weights,
        )
        return annotated

    def fuse_frontier(
        self,
        task_frame: TaskFrame,
        branch_candidates: dict[str, list[HyperedgeCandidate]],
        evidence_subgraph: dict[str, Any],
        control_state: RetrievalControlState,
        top_k: int,
    ) -> tuple[list[HyperedgeCandidate], dict[str, Any]]:
        existing_hyperedge_ids = {
            str(item).strip() for item in evidence_subgraph.get("hyperedge_ids", []) if str(item).strip()
        }
        aggregate: dict[str, HyperedgeCandidate] = {}
        contributions: dict[str, dict[str, float]] = defaultdict(dict)

        for branch_kind, candidates in branch_candidates.items():
            for candidate in candidates:
                entry = aggregate.get(candidate.hyperedge_id)
                if entry is None:
                    entry = HyperedgeCandidate(
                        hyperedge_id=candidate.hyperedge_id,
                        hyperedge_text=candidate.hyperedge_text,
                        score=candidate.score,
                        branch_kind="frontier",
                        branch_score=candidate.branch_score,
                        coverage_gain=candidate.coverage_gain,
                        constraint_gain=candidate.constraint_gain,
                        relation_gain=candidate.relation_gain,
                        connector_gain=candidate.connector_gain,
                        novelty_gain=candidate.novelty_gain,
                        focus_gain=candidate.focus_gain,
                        penalty=candidate.penalty,
                        entity_ids=list(candidate.entity_ids),
                        chunk_ids=list(candidate.chunk_ids),
                        matched_topic_entities=list(candidate.matched_topic_entities),
                        support_entities=list(candidate.support_entities),
                        channel_id=candidate.channel_id,
                        supporting_channel_ids=list(candidate.supporting_channel_ids),
                        supporting_chunks=list(candidate.supporting_chunks),
                        score_breakdown=dict(candidate.score_breakdown),
                        notes=list(candidate.notes),
                        reason=candidate.reason,
                    )
                    aggregate[candidate.hyperedge_id] = entry
                else:
                    entry.branch_score = max(entry.branch_score, candidate.branch_score)
                    entry.coverage_gain = max(entry.coverage_gain, candidate.coverage_gain)
                    entry.constraint_gain = max(entry.constraint_gain, candidate.constraint_gain)
                    entry.relation_gain = max(entry.relation_gain, candidate.relation_gain)
                    entry.connector_gain = max(entry.connector_gain, candidate.connector_gain)
                    entry.novelty_gain = max(entry.novelty_gain, candidate.novelty_gain)
                    entry.focus_gain = max(entry.focus_gain, candidate.focus_gain)
                    entry.penalty = max(entry.penalty, candidate.penalty)
                    for entity_id in candidate.entity_ids:
                        if entity_id not in entry.entity_ids:
                            entry.entity_ids.append(entity_id)
                    for chunk_id in candidate.chunk_ids:
                        if chunk_id not in entry.chunk_ids:
                            entry.chunk_ids.append(chunk_id)
                    for entity_id in candidate.support_entities:
                        if entity_id not in entry.support_entities:
                            entry.support_entities.append(entity_id)
                    for channel_id in candidate.supporting_channel_ids:
                        if channel_id not in entry.supporting_channel_ids:
                            entry.supporting_channel_ids.append(channel_id)
                    for entity_id in candidate.matched_topic_entities:
                        if entity_id not in entry.matched_topic_entities:
                            entry.matched_topic_entities.append(entity_id)

                contributions[candidate.hyperedge_id][branch_kind] = candidate.branch_score

        frontier: list[HyperedgeCandidate] = []
        for hyperedge_id, candidate in aggregate.items():
            anchor_score = (0.55 * candidate.coverage_gain) + (0.45 * candidate.connector_gain)
            repeat_penalty = 1.0 if hyperedge_id in existing_hyperedge_ids else candidate.penalty
            fused_score = (
                (control_state.branch_weights.get("constraint", 0.0) * candidate.constraint_gain)
                + (control_state.branch_weights.get("relation", 0.0) * candidate.relation_gain)
                + (control_state.branch_weights.get("anchor", 0.0) * anchor_score)
                + ((self.reasoning_config.fused_novelty_weight if self.reasoning_config else 0.12) * candidate.novelty_gain)
                + ((self.reasoning_config.fused_focus_weight if self.reasoning_config else 0.16) * candidate.focus_gain)
                - ((self.reasoning_config.fused_penalty_weight if self.reasoning_config else 0.18) * repeat_penalty)
            )
            candidate.fused_score = fused_score
            candidate.score = fused_score
            candidate.reason = (
                f"constraint={candidate.constraint_gain:.3f}, relation={candidate.relation_gain:.3f}, "
                f"anchor={anchor_score:.3f}, novelty={candidate.novelty_gain:.3f}, "
                f"focus={candidate.focus_gain:.3f}, penalty={repeat_penalty:.3f}"
            )
            candidate.score_breakdown.update(
                {
                    "constraint_weight": round(control_state.branch_weights.get("constraint", 0.0), 4),
                    "relation_weight": round(control_state.branch_weights.get("relation", 0.0), 4),
                    "anchor_weight": round(control_state.branch_weights.get("anchor", 0.0), 4),
                    "anchor_score": round(anchor_score, 4),
                    "focus": round(candidate.focus_gain, 4),
                    "penalty": round(repeat_penalty, 4),
                    "fused_score": round(fused_score, 4),
                }
            )
            candidate.notes.append(
                "contributors:"
                + ",".join(
                    f"{branch_kind}={score:.3f}"
                    for branch_kind, score in sorted(contributions[hyperedge_id].items())
                )
            )
            frontier.append(candidate)

        frontier.sort(key=lambda item: item.fused_score, reverse=True)
        selected_frontier = frontier[:top_k]
        branch_contribution_summary = {
            branch_kind: [candidate.hyperedge_id for candidate in candidates[: self.config.hyperedge_top_k]]
            for branch_kind, candidates in branch_candidates.items()
        }
        merge_result = {
            "frontier_hyperedge_ids": [candidate.hyperedge_id for candidate in selected_frontier],
            "frontier": [candidate.to_dict() for candidate in selected_frontier],
            "branch_contributions": branch_contribution_summary,
            "preferred_branches": self._preferred_branches(branch_candidates),
            "coverage_summary": {
                "frontier_size": len(selected_frontier),
                "evidence_hyperedges": len(evidence_subgraph.get("hyperedge_ids", [])),
                "topic_entity_coverage": self._frontier_topic_coverage(task_frame, selected_frontier),
            },
            "answer_hypotheses": self._derive_answer_hypotheses(task_frame, selected_frontier),
            "missing_requirements": list(control_state.missing_requirements),
            "next_focus": list(control_state.next_focus),
            "notes": "Global frontier fused from explicit branch scores.",
        }
        return selected_frontier, merge_result

    def combine_channel_frontiers(
        self,
        task_frame: TaskFrame,
        channel_frontiers: dict[str, list[HyperedgeCandidate]],
        channel_merge_results: dict[str, dict[str, Any]],
        evidence_subgraph: dict[str, Any],
        control_state: RetrievalControlState,
        top_k: int,
    ) -> tuple[list[HyperedgeCandidate], dict[str, Any]]:
        existing_hyperedge_ids = {
            str(item).strip() for item in evidence_subgraph.get("hyperedge_ids", []) if str(item).strip()
        }
        aggregate: dict[str, HyperedgeCandidate] = {}
        supporting_channels: dict[str, set[str]] = defaultdict(set)
        branch_contributions: dict[str, list[str]] = defaultdict(list)

        for channel_id, frontier_candidates in channel_frontiers.items():
            channel_result = channel_merge_results.get(channel_id, {})
            preferred_branches = [
                str(item).strip()
                for item in channel_result.get("preferred_branches", [])
                if str(item).strip()
            ]
            for candidate in frontier_candidates:
                entry = aggregate.get(candidate.hyperedge_id)
                if entry is None:
                    entry = self._clone_candidate(candidate)
                    aggregate[candidate.hyperedge_id] = entry
                else:
                    self._merge_candidate_state(entry, candidate)
                supporting_channels[candidate.hyperedge_id].add(channel_id)
                for branch_kind in preferred_branches[:2]:
                    if candidate.hyperedge_id not in branch_contributions[branch_kind]:
                        branch_contributions[branch_kind].append(candidate.hyperedge_id)

        combined: list[HyperedgeCandidate] = []
        for hyperedge_id, candidate in aggregate.items():
            channel_count = len(supporting_channels[hyperedge_id])
            repeat_penalty = 1.0 if hyperedge_id in existing_hyperedge_ids else candidate.penalty
            channel_bonus = 0.08 * max(channel_count - 1, 0)
            combined_score = max(candidate.fused_score, candidate.branch_score, candidate.score) + channel_bonus - (
                (self.reasoning_config.fused_penalty_weight if self.reasoning_config else 0.18) * repeat_penalty
            )
            candidate.score = combined_score
            candidate.fused_score = combined_score
            candidate.supporting_channel_ids = sorted(supporting_channels[hyperedge_id])
            candidate.notes.append(
                "channels:" + ",".join(normalize_label(channel_id) for channel_id in candidate.supporting_channel_ids)
            )
            candidate.score_breakdown.update(
                {
                    "channel_count": channel_count,
                    "channel_bonus": round(channel_bonus, 4),
                    "combined_score": round(combined_score, 4),
                }
            )
            combined.append(candidate)

        combined.sort(
            key=lambda item: (
                len(item.supporting_channel_ids),
                item.fused_score,
                item.coverage_gain,
                item.novelty_gain,
            ),
            reverse=True,
        )
        selected_frontier = combined[:top_k]
        merge_result = {
            "frontier_hyperedge_ids": [candidate.hyperedge_id for candidate in selected_frontier],
            "frontier": [candidate.to_dict() for candidate in selected_frontier],
            "branch_contributions": dict(branch_contributions),
            "channel_frontiers": {
                channel_id: [candidate.hyperedge_id for candidate in candidates]
                for channel_id, candidates in channel_frontiers.items()
            },
            "preferred_branches": self._preferred_branches_from_channel_merges(channel_merge_results),
            "coverage_summary": {
                "frontier_size": len(selected_frontier),
                "evidence_hyperedges": len(evidence_subgraph.get("hyperedge_ids", [])),
                "topic_entity_coverage": self._frontier_topic_coverage(task_frame, selected_frontier),
                "active_channels": len([channel_id for channel_id, candidates in channel_frontiers.items() if candidates]),
            },
            "answer_hypotheses": self._derive_answer_hypotheses(task_frame, selected_frontier),
            "missing_requirements": list(control_state.missing_requirements),
            "next_focus": list(control_state.next_focus),
            "notes": "Global frontier aggregated from entity-parallel channel frontiers.",
        }
        return selected_frontier, merge_result

    def rank_expansion_entities(
        self,
        question: str,
        task_frame: TaskFrame,
        frontier_candidates: list[HyperedgeCandidate],
        control_state: RetrievalControlState,
        *,
        exclude_entity_ids: set[str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        exclude = set(exclude_entity_ids or set())
        support_map: dict[str, list[str]] = defaultdict(list)
        frontier_size = max(len(frontier_candidates), 1)
        query_texts = self._dedupe_texts(
            [
                question,
                *task_frame.topic_entities,
                *task_frame.hard_constraints,
                task_frame.relation_intent,
                task_frame.answer_type_hint,
                *control_state.current_focus(),
            ]
        )
        query_vectors = self.embedder.embed_texts(query_texts, stage="expansion_entities") if query_texts else []

        for candidate in frontier_candidates:
            hyperedge_label = short_text(
                normalize_label(candidate.hyperedge_text or candidate.hyperedge_id),
                180,
            )
            for entity_id in candidate.entity_ids:
                if entity_id in exclude:
                    continue
                if hyperedge_label not in support_map[entity_id]:
                    support_map[entity_id].append(hyperedge_label)

        scored: list[dict[str, Any]] = []
        for entity_id, supported_by in support_map.items():
            profile_text = self._entity_profile_text(entity_id)
            if not profile_text:
                profile_text = normalize_label(entity_id)
            question_match = max(
                self._hybrid_text_score(
                    [question, *task_frame.topic_entities, *task_frame.hard_constraints, task_frame.relation_intent],
                    profile_text,
                ),
                self._entity_vector_similarity(query_vectors, entity_id),
            )
            focus_match = self._hybrid_text_score(control_state.current_focus(), profile_text)
            answer_type_match = self._hybrid_text_score([task_frame.answer_type_hint], profile_text)
            support_score = len(supported_by) / frontier_size
            coarse_score = (
                (0.45 * question_match)
                + (0.2 * focus_match)
                + (0.15 * answer_type_match)
                + (0.2 * support_score)
            )
            scored.append(
                {
                    "entity_id": entity_id,
                    "entity_label": normalize_label(entity_id),
                    "description": short_text(profile_text, 280),
                    "source_hyperedges": supported_by[:2],
                    "question_match": round(question_match, 4),
                    "focus_match": round(focus_match, 4),
                    "support_count": len(supported_by),
                    "coarse_score": round(coarse_score, 4),
                }
            )

        scored.sort(
            key=lambda item: (
                item["focus_match"] > 0.0,
                item["coarse_score"],
                item["support_count"],
            ),
            reverse=True,
        )
        return scored[:top_k]

    def build_evidence_items(
        self,
        thought_id: str,
        branch_kind: str,
        candidates: list[HyperedgeCandidate],
        limit: int | None = None,
    ) -> list[EvidenceItem]:
        target_limit = limit or self.config.evidence_keep
        evidence_items: list[EvidenceItem] = []
        seen_chunk_ids: set[str] = set()
        for rank, candidate in enumerate(candidates, start=1):
            for chunk_id in candidate.chunk_ids:
                if chunk_id in seen_chunk_ids:
                    continue
                content = self._compose_evidence_content(candidate.hyperedge_id, chunk_id)
                if not content:
                    continue
                evidence_items.append(
                    EvidenceItem(
                        evidence_id=f"ev-{thought_id}-{rank}",
                        chunk_id=chunk_id,
                        content=content,
                        score=candidate.fused_score or candidate.branch_score or candidate.score,
                        source_node_ids=[candidate.hyperedge_id, *candidate.entity_ids[:8]],
                        source_edge_ids=self.dataset.graph.adjacency.get(candidate.hyperedge_id, [])[:12],
                        notes=[
                            f"branch:{branch_kind}",
                            *( [f"channel:{normalize_label(candidate.channel_id)}"] if candidate.channel_id else [] ),
                            f"hyperedge:{normalize_label(candidate.hyperedge_id)}",
                            *candidate.notes[:4],
                        ],
                    )
                )
                seen_chunk_ids.add(chunk_id)
                break
            if len(evidence_items) >= target_limit:
                break
        return evidence_items

    def retrieve(self, thought: ThoughtState) -> list[EvidenceItem]:
        cache_key = self._cache_key(thought)
        cached_items = self._cache.get(cache_key)
        if cached_items is not None:
            self.logger.info("Reused %s cached evidence items for thought %s", len(cached_items), thought.thought_id)
            return list(cached_items)

        branch_kind = str(thought.metadata.get("branch_kind", thought.metadata.get("intent", "generic")) or "generic")
        query_texts = self._dedupe_texts([thought.content, *thought.grounding.anchor_texts[:2]])
        candidates = self._rank_hyperedges(
            query_texts=query_texts,
            topic_entities=thought.grounding.anchor_texts,
            hard_constraints=[],
            relation_intent=thought.objective,
            relation_skeleton="",
            initial_entity_ids=[],
            connected_entity_ids=thought.grounding.node_ids,
            branch_kind=branch_kind,
            focus_texts=[],
            exclude_hyperedge_ids=set(),
            existing_hyperedge_ids=set(),
        )
        evidence_items = self.build_evidence_items(
            thought_id=thought.thought_id,
            branch_kind=branch_kind,
            candidates=candidates[: self.config.evidence_keep],
            limit=self.config.evidence_keep,
        )
        self.logger.info("Retrieved %s evidence items for thought %s", len(evidence_items), thought.thought_id)
        self._cache[cache_key] = list(evidence_items)
        return evidence_items

    def _rank_entities(self, query_texts: list[str]) -> list[VectorMatch]:
        scores: dict[str, float] = defaultdict(float)
        match_meta: dict[str, VectorMatch] = {}

        for match in self._vector_entity_matches(query_texts):
            scores[match.label] = max(scores[match.label], match.score)
            existing = match_meta.get(match.label)
            if existing is None or match.score > existing.score:
                match_meta[match.label] = match

        lexical_matches = self._lexical_matches(
            query_texts=query_texts,
            candidate_labels=self._entity_labels,
            top_k=self.config.lexical_anchor_top_k,
        )
        for label, score in lexical_matches:
            if score <= 0:
                continue
            scores[label] = max(scores[label], score)
            if label not in match_meta or score > match_meta[label].score:
                match_meta[label] = VectorMatch(
                    item_id=label,
                    label=label,
                    score=score,
                    metadata={"method": "lexical"},
                )

        ranked = sorted(match_meta.values(), key=lambda item: scores[item.label], reverse=True)
        return ranked[: max(self.config.entity_top_k * 2, self.config.lexical_anchor_top_k)]

    def _ground_task_frame_entities(self, question: str, task_frame: TaskFrame) -> dict[str, Any]:
        raw_topic_entities = self._dedupe_texts([*task_frame.topic_entities, *task_frame.anchors])
        grounded_by_label: dict[str, VectorMatch] = {}
        seed_traces: list[dict[str, Any]] = []
        filtered_non_discriminative: list[str] = []

        for seed in raw_topic_entities:
            matches = self._link_topic_seed(seed, stage="seed")
            accepted, filtered = self._partition_grounded_matches(matches, task_frame)
            for match in accepted:
                existing = grounded_by_label.get(match.label)
                if existing is None or match.score > existing.score:
                    grounded_by_label[match.label] = match
            for label in filtered:
                if label not in filtered_non_discriminative:
                    filtered_non_discriminative.append(label)
            seed_traces.append(
                {
                    "seed": seed,
                    "stage": "seed",
                    "linked_entities": [match.to_dict() for match in matches],
                    "accepted_entities": [match.label for match in accepted],
                    "filtered_non_discriminative": list(filtered),
                }
            )

        used_question_fallback = False
        if not grounded_by_label:
            used_question_fallback = True
            fallback_matches = self._link_topic_seed(question, stage="question_fallback")
            accepted, filtered = self._partition_grounded_matches(fallback_matches, task_frame)
            for match in accepted:
                existing = grounded_by_label.get(match.label)
                if existing is None or match.score > existing.score:
                    grounded_by_label[match.label] = match
            for label in filtered:
                if label not in filtered_non_discriminative:
                    filtered_non_discriminative.append(label)
            seed_traces.append(
                {
                    "seed": question,
                    "stage": "question_fallback",
                    "linked_entities": [match.to_dict() for match in fallback_matches],
                    "accepted_entities": [match.label for match in accepted],
                    "filtered_non_discriminative": list(filtered),
                }
            )

        entity_matches = sorted(
            grounded_by_label.values(),
            key=lambda match: (
                str(match.metadata.get("method", "")) == "exact",
                match.score,
                lexical_overlap_score(raw_topic_entities or [question], match.label),
            ),
            reverse=True,
        )
        grounded_topic_entities = [match.label for match in entity_matches[: self.config.entity_top_k]]
        metadata = {
            "method": "proh_topic_linking",
            "raw_topic_entities": list(raw_topic_entities),
            "used_question_fallback": used_question_fallback,
            "seed_traces": seed_traces,
            "filtered_non_discriminative": filtered_non_discriminative,
            "grounded_topic_entities": grounded_topic_entities,
        }
        return {
            "entity_matches": entity_matches,
            "grounded_topic_entities": grounded_topic_entities,
            "metadata": metadata,
        }

    def _link_topic_seed(self, seed: str, stage: str) -> list[VectorMatch]:
        cleaned_seed = normalize_label(seed)
        if not cleaned_seed:
            return []

        exact_matches = self._exact_entity_matches(cleaned_seed, stage)
        if exact_matches:
            return exact_matches[: max(self.config.topic_entity_link_top_k, 1)]

        query_vectors = self.embedder.embed_texts([cleaned_seed], stage="topic_entity_linking")
        if not query_vectors:
            return []
        matches = self.dataset.entity_store.query(
            query_vectors[0],
            top_k=max(self.config.topic_entity_link_top_k, 1),
        )
        linked: list[VectorMatch] = []
        for match in matches:
            if match.score < self.config.topic_entity_link_threshold:
                continue
            metadata = dict(match.metadata)
            metadata.update(
                {
                    "seed": cleaned_seed,
                    "stage": stage,
                    "method": "vector",
                }
            )
            linked.append(
                VectorMatch(
                    item_id=match.item_id,
                    label=match.label,
                    score=match.score,
                    metadata=metadata,
                )
            )
        return linked

    def _exact_entity_matches(self, seed: str, stage: str) -> list[VectorMatch]:
        normalized_seed = normalize_label(seed).lower()
        if not normalized_seed:
            return []
        matches: list[VectorMatch] = []
        for label in self._entity_labels_by_normalized.get(normalized_seed, [])[: max(self.config.topic_entity_link_top_k, 1)]:
            matches.append(
                VectorMatch(
                    item_id=self._entity_row_id_by_label.get(label, label),
                    label=label,
                    score=1.0,
                    metadata={
                        "seed": seed,
                        "stage": stage,
                        "method": "exact",
                    },
                )
            )
        return matches

    def _partition_grounded_matches(
        self,
        matches: list[VectorMatch],
        task_frame: TaskFrame,
    ) -> tuple[list[VectorMatch], list[str]]:
        accepted: list[VectorMatch] = []
        filtered: list[str] = []
        for match in matches:
            if self._is_discriminative_topic_entity(match.label, task_frame):
                accepted.append(match)
            else:
                filtered.append(match.label)
        return accepted, filtered

    def _is_discriminative_topic_entity(self, entity_id: str, task_frame: TaskFrame) -> bool:
        tokens = set(content_tokens(entity_id))
        if not tokens:
            return False
        if tokens.issubset(TEMPORAL_TOPIC_ENTITY_TOKENS):
            return False
        if len(tokens) == 1 and tokens.issubset(GENERIC_TOPIC_ENTITY_TOKENS):
            return False

        answer_type_tokens = set(content_tokens(task_frame.answer_type_hint))
        if answer_type_tokens and tokens.issubset(answer_type_tokens | GENERIC_TOPIC_ENTITY_TOKENS):
            return False
        if len(tokens) <= 2 and tokens.issubset(GENERIC_TOPIC_ENTITY_TOKENS):
            return False

        node = self.dataset.graph.nodes.get(entity_id)
        entity_type = str(getattr(node, "entity_type", "") or "").strip().lower()
        if entity_type in {"date", "duration", "number", "numeric", "time", "year"}:
            return False
        return True

    def _rank_hyperedges(
        self,
        query_texts: list[str],
        topic_entities: list[str],
        hard_constraints: list[str],
        relation_intent: str,
        relation_skeleton: str,
        initial_entity_ids: list[str],
        connected_entity_ids: list[str],
        branch_kind: str,
        focus_texts: list[str],
        exclude_hyperedge_ids: set[str],
        existing_hyperedge_ids: set[str],
        fallback_to_initial_entities: bool = True,
    ) -> list[HyperedgeCandidate]:
        pool: set[str] = set()
        frontier_entities = list(connected_entity_ids)
        if not frontier_entities and fallback_to_initial_entities:
            frontier_entities = list(initial_entity_ids)
        pool.update(self.dataset.graph.expand_from_entities(frontier_entities))
        pool.update(match.label for match in self._vector_hyperedge_matches(query_texts))
        pool.update(
            label
            for label, score in self._lexical_matches(
                query_texts=query_texts,
                candidate_labels=self._hyperedge_labels,
                top_k=self.config.branch_candidate_pool,
            )
            if score > 0
        )
        pool.difference_update(exclude_hyperedge_ids)

        candidates: list[HyperedgeCandidate] = []
        initial_entity_set = set(initial_entity_ids)
        connected_entity_set = set(connected_entity_ids)
        topic_entity_set = set(topic_entities)

        for hyperedge_id in pool:
            if hyperedge_id not in self.dataset.graph.nodes:
                continue
            entity_ids = self.dataset.graph.hyperedge_entity_ids(hyperedge_id)
            chunk_ids = self.dataset.graph.hyperedge_chunk_ids(hyperedge_id)
            hyperedge_text = normalize_label(hyperedge_id)
            chunk_text = " ".join(self.dataset.get_chunk_text(chunk_id)[:400] for chunk_id in chunk_ids[:1]).strip()
            full_text = f"{hyperedge_text} {chunk_text}".strip()

            question_gain = self._hybrid_hyperedge_score([query_texts[0]], hyperedge_id, hyperedge_text)
            relation_gain = self._hybrid_hyperedge_score(
                [text for text in [relation_intent, relation_skeleton] if text],
                hyperedge_id,
                hyperedge_text,
            )
            constraint_gain = self._hybrid_text_score(hard_constraints, full_text)
            matched_topic_entities = self._matched_entities(entity_ids, initial_entity_set, topic_entity_set)
            coverage_gain = len(matched_topic_entities) / max(len(topic_entities) or len(initial_entity_ids) or 1, 1)
            connector_gain = len(set(entity_ids) & connected_entity_set) / max(len(entity_ids) or 1, 1)
            novelty_gain = len([entity_id for entity_id in entity_ids if entity_id not in connected_entity_set]) / max(
                len(entity_ids) or 1,
                1,
            )
            focus_gain = self._hybrid_text_score(focus_texts, full_text)
            penalty = 0.35 if hyperedge_id in existing_hyperedge_ids else 0.0
            anchor_score = (0.55 * coverage_gain) + (0.45 * connector_gain)

            if branch_kind == "constraint":
                branch_score = (0.46 * constraint_gain) + (0.2 * question_gain) + (0.16 * focus_gain) + (0.1 * coverage_gain) + (0.08 * connector_gain)
                reason = "Constraint branch prioritizes hard-constraint match."
            elif branch_kind == "relation":
                branch_score = (0.48 * relation_gain) + (0.2 * question_gain) + (0.16 * focus_gain) + (0.08 * connector_gain) + (0.08 * coverage_gain)
                reason = "Relation branch prioritizes relation intent and skeleton match."
            else:
                branch_score = (0.38 * anchor_score) + (0.18 * question_gain) + (0.16 * focus_gain) + (0.14 * novelty_gain) + (0.14 * relation_gain)
                reason = "Anchor branch prioritizes coverage, connectivity, and novelty."

            candidate = HyperedgeCandidate(
                hyperedge_id=hyperedge_id,
                hyperedge_text=hyperedge_text,
                score=branch_score,
                branch_kind=branch_kind,
                branch_score=branch_score,
                coverage_gain=coverage_gain,
                constraint_gain=constraint_gain,
                relation_gain=relation_gain,
                connector_gain=connector_gain,
                novelty_gain=novelty_gain,
                focus_gain=focus_gain,
                penalty=penalty,
                entity_ids=entity_ids,
                chunk_ids=chunk_ids,
                matched_topic_entities=matched_topic_entities,
                support_entities=matched_topic_entities or entity_ids[:6],
                supporting_chunks=[short_text(chunk_text, 220)] if chunk_text else [],
                score_breakdown={
                    "question": round(question_gain, 4),
                    "constraint": round(constraint_gain, 4),
                    "relation": round(relation_gain, 4),
                    "coverage": round(coverage_gain, 4),
                    "connector": round(connector_gain, 4),
                    "novelty": round(novelty_gain, 4),
                    "focus": round(focus_gain, 4),
                    "penalty": round(penalty, 4),
                    "branch_score": round(branch_score, 4),
                },
                notes=[
                    f"matched_topics={len(matched_topic_entities)}",
                    f"entity_degree={len(entity_ids)}",
                ],
                reason=reason,
            )
            candidates.append(candidate)

        candidates.sort(
            key=lambda item: (
                item.focus_gain >= self.config.focus_match_min_score,
                item.branch_score,
                item.coverage_gain,
                item.novelty_gain,
            ),
            reverse=True,
        )
        return candidates[: max(self.config.branch_candidate_pool, self.config.hyperedge_top_k)]

    def _apply_control_filter(
        self,
        branch_kind: str,
        candidates: list[HyperedgeCandidate],
        control_state: RetrievalControlState,
    ) -> list[HyperedgeCandidate]:
        if not control_state.current_focus():
            control_state.candidate_filters[branch_kind] = {
                "prefer_focus_match": False,
                "focus_threshold": self.config.focus_match_min_score,
                "expansion_frontier_size": 0,
            }
            return candidates

        prefer_focus_match = control_state.branch_weights.get(branch_kind, 0.0) >= max(control_state.branch_weights.values())
        focus_candidates = [candidate for candidate in candidates if candidate.focus_gain >= self.config.focus_match_min_score]
        control_state.candidate_filters[branch_kind] = {
            "prefer_focus_match": prefer_focus_match,
            "focus_threshold": self.config.focus_match_min_score,
            "focus_candidate_count": len(focus_candidates),
        }
        if prefer_focus_match and focus_candidates:
            remainder = [candidate for candidate in candidates if candidate.focus_gain < self.config.focus_match_min_score]
            return focus_candidates + remainder
        return candidates

    def _preferred_branches(self, branch_candidates: dict[str, list[HyperedgeCandidate]]) -> list[str]:
        ranked = sorted(
            branch_candidates,
            key=lambda branch_kind: max((candidate.branch_score for candidate in branch_candidates[branch_kind]), default=0.0),
            reverse=True,
        )
        return ranked

    def _preferred_branches_from_channel_merges(self, channel_merge_results: dict[str, dict[str, Any]]) -> list[str]:
        scores: dict[str, float] = defaultdict(float)
        for merge_result in channel_merge_results.values():
            for index, branch_kind in enumerate(merge_result.get("preferred_branches", [])):
                cleaned = str(branch_kind).strip()
                if not cleaned:
                    continue
                scores[cleaned] += max(0.0, 1.0 - (0.2 * index))
        ranked = sorted(scores, key=lambda branch_kind: scores[branch_kind], reverse=True)
        return ranked

    def _clone_candidate(self, candidate: HyperedgeCandidate) -> HyperedgeCandidate:
        return HyperedgeCandidate(
            hyperedge_id=candidate.hyperedge_id,
            hyperedge_text=candidate.hyperedge_text,
            score=candidate.score,
            branch_kind=candidate.branch_kind,
            branch_score=candidate.branch_score,
            fused_score=candidate.fused_score,
            coverage_gain=candidate.coverage_gain,
            constraint_gain=candidate.constraint_gain,
            relation_gain=candidate.relation_gain,
            connector_gain=candidate.connector_gain,
            novelty_gain=candidate.novelty_gain,
            focus_gain=candidate.focus_gain,
            penalty=candidate.penalty,
            entity_ids=list(candidate.entity_ids),
            chunk_ids=list(candidate.chunk_ids),
            matched_topic_entities=list(candidate.matched_topic_entities),
            support_entities=list(candidate.support_entities),
            channel_id=candidate.channel_id,
            supporting_channel_ids=list(candidate.supporting_channel_ids),
            supporting_chunks=list(candidate.supporting_chunks),
            score_breakdown=dict(candidate.score_breakdown),
            notes=list(candidate.notes),
            reason=candidate.reason,
        )

    def _merge_candidate_state(self, entry: HyperedgeCandidate, candidate: HyperedgeCandidate) -> None:
        entry.branch_score = max(entry.branch_score, candidate.branch_score)
        entry.fused_score = max(entry.fused_score, candidate.fused_score)
        entry.coverage_gain = max(entry.coverage_gain, candidate.coverage_gain)
        entry.constraint_gain = max(entry.constraint_gain, candidate.constraint_gain)
        entry.relation_gain = max(entry.relation_gain, candidate.relation_gain)
        entry.connector_gain = max(entry.connector_gain, candidate.connector_gain)
        entry.novelty_gain = max(entry.novelty_gain, candidate.novelty_gain)
        entry.focus_gain = max(entry.focus_gain, candidate.focus_gain)
        entry.penalty = max(entry.penalty, candidate.penalty)
        for entity_id in candidate.entity_ids:
            if entity_id not in entry.entity_ids:
                entry.entity_ids.append(entity_id)
        for chunk_id in candidate.chunk_ids:
            if chunk_id not in entry.chunk_ids:
                entry.chunk_ids.append(chunk_id)
        for entity_id in candidate.support_entities:
            if entity_id not in entry.support_entities:
                entry.support_entities.append(entity_id)
        for entity_id in candidate.matched_topic_entities:
            if entity_id not in entry.matched_topic_entities:
                entry.matched_topic_entities.append(entity_id)
        for channel_id in candidate.supporting_channel_ids:
            if channel_id not in entry.supporting_channel_ids:
                entry.supporting_channel_ids.append(channel_id)
        for chunk_text in candidate.supporting_chunks:
            if chunk_text not in entry.supporting_chunks:
                entry.supporting_chunks.append(chunk_text)
        for note in candidate.notes:
            if note not in entry.notes:
                entry.notes.append(note)

    def _frontier_topic_coverage(self, task_frame: TaskFrame, frontier: list[HyperedgeCandidate]) -> float:
        if not task_frame.topic_entities:
            return 0.0
        covered: set[str] = set()
        for candidate in frontier:
            covered.update(candidate.matched_topic_entities)
        return len(covered) / max(len(task_frame.topic_entities), 1)

    def _derive_answer_hypotheses(self, task_frame: TaskFrame, frontier: list[HyperedgeCandidate]) -> list[str]:
        scores: dict[str, float] = defaultdict(float)
        blocked_texts = [*task_frame.topic_entities, *task_frame.hard_constraints, task_frame.relation_intent, task_frame.question]
        for candidate in frontier:
            for entity_id in candidate.support_entities or candidate.entity_ids:
                if lexical_overlap_score(blocked_texts, entity_id) > 0.5:
                    continue
                scores[normalize_label(entity_id)] += max(candidate.fused_score, candidate.branch_score)
        return [
            entity_id
            for entity_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]
        ]

    def _vector_entity_matches(self, query_texts: list[str]) -> list[VectorMatch]:
        if not query_texts:
            return []
        query_vectors = self.embedder.embed_texts(query_texts, stage="anchor_entities")
        matches: list[VectorMatch] = []
        for vector in query_vectors:
            matches.extend(self.dataset.entity_store.query(vector, top_k=self.config.entity_top_k))
        return matches

    def _vector_hyperedge_matches(self, query_texts: list[str]) -> list[VectorMatch]:
        if not query_texts:
            return []
        query_vectors = self.embedder.embed_texts(query_texts, stage="branch_hyperedges")
        matches: list[VectorMatch] = []
        for vector in query_vectors:
            matches.extend(self.dataset.hyperedge_store.query(vector, top_k=self.config.branch_candidate_pool))
        return matches

    def _lexical_matches(
        self,
        query_texts: list[str],
        candidate_labels: list[str],
        top_k: int,
    ) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        for label in candidate_labels:
            score = lexical_overlap_score(query_texts, label)
            if score > 0:
                scored.append((label, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _hybrid_hyperedge_score(self, query_texts: list[str], hyperedge_id: str, hyperedge_text: str) -> float:
        if not query_texts:
            return 0.0
        lexical = self._hybrid_text_score(query_texts, hyperedge_text)
        row_id = self._hyperedge_row_id_by_label.get(hyperedge_id)
        if row_id is None:
            return lexical
        query_vectors = self.embedder.embed_texts(query_texts, stage="hyperedge_similarity")
        vector = max(self.dataset.hyperedge_store.similarity(query_vector, row_id) for query_vector in query_vectors)
        return max(lexical, max(vector, 0.0))

    def _hybrid_text_score(self, query_texts: list[str], candidate_text: str) -> float:
        if not query_texts or not candidate_text:
            return 0.0
        return lexical_overlap_score(query_texts, candidate_text)

    def _matched_entities(
        self,
        entity_ids: list[str],
        initial_entity_set: set[str],
        topic_entity_set: set[str],
    ) -> list[str]:
        matched: list[str] = []
        for entity_id in entity_ids:
            if entity_id in initial_entity_set:
                matched.append(entity_id)
                continue
            if lexical_overlap_score(list(topic_entity_set), entity_id) > 0:
                matched.append(entity_id)
        return matched

    def _compose_evidence_content(self, hyperedge_id: str, chunk_id: str) -> str:
        hyperedge_text = normalize_label(hyperedge_id)
        chunk_text = self.dataset.get_chunk_text(chunk_id)
        if not chunk_text:
            return hyperedge_text
        return f"{hyperedge_text}\n\n{short_text(chunk_text, 900)}"

    def _branch_query_texts(
        self,
        question: str,
        task_frame: TaskFrame,
        branch_kind: str,
        control_state: RetrievalControlState,
    ) -> list[str]:
        control_focus = control_state.current_focus()
        if branch_kind == "constraint":
            return self._dedupe_texts([question, *task_frame.hard_constraints, *control_focus, *task_frame.topic_entities])
        if branch_kind == "relation":
            return self._dedupe_texts(
                [question, task_frame.relation_intent, task_frame.relation_skeleton, *control_focus, *task_frame.topic_entities]
            )
        if branch_kind == "anchor":
            return self._dedupe_texts(
                [question, *task_frame.topic_entities, task_frame.answer_type_hint, *control_focus]
            )
        return self._dedupe_texts([question, *task_frame.topic_entities, task_frame.relation_intent, *control_focus])

    def _build_row_lookup(self, store: Any) -> dict[str, str]:
        lookup: dict[str, str] = {}
        if not hasattr(store, "rows") or not hasattr(store, "row_ids"):
            return lookup
        for row, row_id in zip(store.rows, store.row_ids, strict=True):
            label = store._label_for_row(row, row_id)
            lookup[str(label)] = str(row_id)
        return lookup

    def _build_normalized_label_lookup(self, labels: list[str]) -> dict[str, list[str]]:
        lookup: dict[str, list[str]] = defaultdict(list)
        for label in labels:
            normalized = normalize_label(label).lower()
            if not normalized:
                continue
            lookup[normalized].append(label)
        return lookup

    def _dedupe_texts(self, texts: list[str]) -> list[str]:
        deduped: list[str] = []
        for text in texts:
            cleaned = str(text).strip()
            if cleaned and cleaned not in deduped:
                deduped.append(cleaned)
        return deduped

    def _cache_key(self, thought: ThoughtState) -> str:
        anchors = "||".join(thought.grounding.anchor_texts[:2])
        return f"{thought.thought_id}::{thought.content}::{anchors}"

    def _entity_profile_text(self, entity_id: str) -> str:
        node = self.dataset.graph.nodes.get(entity_id)
        parts = [normalize_label(entity_id)]
        if node is not None and node.description:
            parts.append(normalize_label(node.description))
        if node is not None:
            for chunk_id in node.source_ids[:1]:
                chunk_text = self.dataset.get_chunk_text(chunk_id)
                if chunk_text:
                    parts.append(short_text(chunk_text, 220))
        return " ".join(part for part in parts if part).strip()

    def _entity_vector_similarity(self, query_vectors: list[np.ndarray] | list[Any], entity_id: str) -> float:
        row_id = self._entity_row_id_by_label.get(entity_id)
        if row_id is None or not query_vectors:
            return 0.0
        return max(self.dataset.entity_store.similarity(query_vector, row_id) for query_vector in query_vectors)
