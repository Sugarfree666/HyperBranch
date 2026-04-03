from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from ..config import RetrievalConfig
from ..data.loaders import DatasetBundle
from ..models import EvidenceItem, HyperedgeCandidate, TaskFrame, ThoughtState, VectorMatch
from ..utils import lexical_overlap_score, normalize_label, short_text


class EvidenceRetriever:
    def __init__(
        self,
        dataset: DatasetBundle,
        embedder: Any,
        config: RetrievalConfig,
        logger: logging.Logger,
    ) -> None:
        self.dataset = dataset
        self.embedder = embedder
        self.config = config
        self.logger = logger
        self._cache: dict[str, list[EvidenceItem]] = {}
        self._entity_labels = [node_id for node_id, node in dataset.graph.nodes.items() if node.role == "entity"]
        self._hyperedge_labels = [node_id for node_id, node in dataset.graph.nodes.items() if node.role == "hyperedge"]
        self._entity_row_id_by_label = self._build_row_lookup(dataset.entity_store)
        self._hyperedge_row_id_by_label = self._build_row_lookup(dataset.hyperedge_store)

    def anchor_task_frame(self, question: str, task_frame: TaskFrame) -> dict[str, Any]:
        query_texts = self._dedupe_texts(
            [
                question,
                *task_frame.topic_entities,
                *task_frame.hard_constraints,
                task_frame.relation_intent,
                task_frame.answer_type_hint,
            ]
        )
        entity_matches = self._rank_entities(query_texts)
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
            exclude_hyperedge_ids=set(),
        )
        initial_hyperedge_ids = [
            candidate.hyperedge_id for candidate in hyperedge_candidates[: self.config.hyperedge_top_k]
        ]
        self.logger.info(
            "Anchored task frame with %s initial entities and %s initial hyperedges",
            len(initial_entity_ids),
            len(initial_hyperedge_ids),
        )
        return {
            "entity_matches": entity_matches,
            "initial_entity_ids": initial_entity_ids,
            "initial_hyperedge_ids": initial_hyperedge_ids,
            "initial_hyperedge_candidates": hyperedge_candidates[: self.config.hyperedge_top_k],
        }

    def retrieve_branch_candidates(
        self,
        question: str,
        task_frame: TaskFrame,
        branch_kind: str,
        evidence_subgraph: dict[str, Any] | None = None,
        exclude_hyperedge_ids: set[str] | None = None,
    ) -> list[HyperedgeCandidate]:
        evidence_subgraph = evidence_subgraph or {}
        connected_entity_ids = [
            str(item).strip() for item in evidence_subgraph.get("entity_ids", []) if str(item).strip()
        ] or list(task_frame.initial_entity_ids)
        known_hyperedges = {
            str(item).strip() for item in evidence_subgraph.get("hyperedge_ids", []) if str(item).strip()
        }
        exclude = set(exclude_hyperedge_ids or set())
        query_texts = self._branch_query_texts(question, task_frame, branch_kind)
        candidates = self._rank_hyperedges(
            query_texts=query_texts,
            topic_entities=task_frame.topic_entities,
            hard_constraints=task_frame.hard_constraints,
            relation_intent=task_frame.relation_intent,
            relation_skeleton=task_frame.relation_skeleton,
            initial_entity_ids=list(task_frame.initial_entity_ids),
            connected_entity_ids=connected_entity_ids,
            branch_kind=branch_kind,
            exclude_hyperedge_ids=exclude,
            known_hyperedge_ids=known_hyperedges,
        )
        self.logger.info("Retrieved %s branch candidates for %s", len(candidates), branch_kind)
        return candidates[: self.config.branch_candidate_pool]

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
                        score=candidate.score,
                        source_node_ids=[candidate.hyperedge_id, *candidate.entity_ids[:8]],
                        source_edge_ids=self.dataset.graph.adjacency.get(candidate.hyperedge_id, [])[:12],
                        notes=[
                            f"branch:{branch_kind}",
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
            exclude_hyperedge_ids=set(),
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
        exclude_hyperedge_ids: set[str],
        known_hyperedge_ids: set[str] | None = None,
    ) -> list[HyperedgeCandidate]:
        pool = set(known_hyperedge_ids or set())
        pool.update(self.dataset.graph.expand_from_entities(initial_entity_ids))
        pool.update(self.dataset.graph.expand_from_entities(connected_entity_ids))
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
            constraint_texts = [*hard_constraints, relation_intent, relation_skeleton]

            question_score = self._hybrid_hyperedge_score([query_texts[0]], hyperedge_id, hyperedge_text)
            relation_score = self._hybrid_hyperedge_score(
                [text for text in [relation_intent, relation_skeleton] if text],
                hyperedge_id,
                hyperedge_text,
            )
            constraint_score = self._hybrid_text_score(constraint_texts, hyperedge_text + " " + chunk_text)
            matched_topic_entities = self._matched_entities(entity_ids, initial_entity_set, topic_entity_set)
            coverage = len(matched_topic_entities) / max(len(topic_entities) or len(initial_entity_ids) or 1, 1)
            connector_strength = len(set(entity_ids) & connected_entity_set) / max(len(entity_ids) or 1, 1)
            novelty = len([entity_id for entity_id in entity_ids if entity_id not in connected_entity_set]) / max(
                len(entity_ids) or 1,
                1,
            )
            novelty_bonus = min(0.2, novelty * 0.2)
            known_bonus = 0.05 if hyperedge_id in (known_hyperedge_ids or set()) else 0.0

            if branch_kind == "constraint":
                score = (0.32 * question_score) + (0.38 * constraint_score) + (0.18 * coverage) + (0.12 * connector_strength)
            elif branch_kind == "relation":
                score = (0.28 * question_score) + (0.42 * relation_score) + (0.18 * connector_strength) + (0.12 * coverage)
            elif branch_kind == "anchor":
                score = (0.26 * question_score) + (0.18 * relation_score) + (0.34 * coverage) + (0.16 * connector_strength) + (0.06 * novelty)
            else:
                score = (0.4 * question_score) + (0.25 * relation_score) + (0.2 * connector_strength) + (0.15 * coverage)

            score += novelty_bonus + known_bonus
            candidates.append(
                HyperedgeCandidate(
                    hyperedge_id=hyperedge_id,
                    hyperedge_text=hyperedge_text,
                    score=score,
                    branch_kind=branch_kind,
                    entity_ids=entity_ids,
                    chunk_ids=chunk_ids,
                    matched_topic_entities=matched_topic_entities,
                    supporting_chunks=[short_text(chunk_text, 220)] if chunk_text else [],
                    score_breakdown={
                        "question": round(question_score, 4),
                        "relation": round(relation_score, 4),
                        "constraint": round(constraint_score, 4),
                        "coverage": round(coverage, 4),
                        "connector": round(connector_strength, 4),
                        "novelty": round(novelty, 4),
                        "known_bonus": round(known_bonus, 4),
                    },
                    notes=[
                        f"matched_topics={len(matched_topic_entities)}",
                        f"entity_degree={len(entity_ids)}",
                    ],
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: max(self.config.branch_candidate_pool, self.config.hyperedge_top_k)]

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

    def _branch_query_texts(self, question: str, task_frame: TaskFrame, branch_kind: str) -> list[str]:
        if branch_kind == "constraint":
            return self._dedupe_texts([question, *task_frame.hard_constraints, *task_frame.topic_entities])
        if branch_kind == "relation":
            return self._dedupe_texts(
                [question, task_frame.relation_intent, task_frame.relation_skeleton, *task_frame.topic_entities]
            )
        if branch_kind == "anchor":
            return self._dedupe_texts([question, *task_frame.topic_entities, task_frame.answer_type_hint])
        return self._dedupe_texts([question, *task_frame.topic_entities, task_frame.relation_intent])

    def _build_row_lookup(self, store: Any) -> dict[str, str]:
        lookup: dict[str, str] = {}
        if not hasattr(store, "rows") or not hasattr(store, "row_ids"):
            return lookup
        for row, row_id in zip(store.rows, store.row_ids, strict=True):
            label = store._label_for_row(row, row_id)
            lookup[str(label)] = str(row_id)
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
