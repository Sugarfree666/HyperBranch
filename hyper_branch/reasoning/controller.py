from __future__ import annotations

import logging
from typing import Any

from ..config import Config
from ..data.loaders import DatasetBundle
from ..logging_utils import TraceStore
from ..models import EvidenceSubgraph, RetrievalControlState, TaskFrame, ThoughtGraph, ThoughtState
from ..retrieval.evidence import EvidenceRetriever
from .operations import ThoughtOperationExecutor
from .taskframe import TaskFrameBuilder, TaskFrameRegistry


class ThoughtController:
    def __init__(
        self,
        config: Config,
        dataset: DatasetBundle,
        taskframe_builder: TaskFrameBuilder,
        registry: TaskFrameRegistry,
        scorer: Any,
        evidence_retriever: EvidenceRetriever,
        executor: ThoughtOperationExecutor,
        llm_service: Any,
        logger: logging.Logger,
        trace_store: TraceStore,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.taskframe_builder = taskframe_builder
        self.registry = registry
        self.scorer = scorer
        self.evidence_retriever = evidence_retriever
        self.executor = executor
        self.llm_service = llm_service
        self.logger = logger
        self.trace_store = trace_store
        self._counter = 0

    def run(self, question: str) -> dict[str, Any]:
        task_frame = self.taskframe_builder.build(question)
        anchor_payload = self.evidence_retriever.anchor_task_frame(question, task_frame)
        task_frame.initial_entity_ids = list(anchor_payload.get("initial_entity_ids", []))
        task_frame.initial_hyperedge_ids = list(anchor_payload.get("initial_hyperedge_ids", []))
        self.registry.register_anchor_matches(task_frame, anchor_payload.get("entity_matches", []))

        root_thought = self.executor.create_root_thought(self._next_id("th"), question, task_frame)
        thought_graph = ThoughtGraph(question=question, root_id=root_thought.thought_id)
        thought_graph.add_thought(root_thought)

        control_state = self._initial_control_state()
        evidence_subgraph = EvidenceSubgraph()
        initial_candidates = list(anchor_payload.get("initial_hyperedge_candidates", []))[
            : self.config.reasoning.evidence_top_k_per_branch
        ]
        initial_anchor_thought: ThoughtState | None = None
        last_merge_thought: ThoughtState | None = None
        if initial_candidates:
            anchor_thought_id = self._next_id("th")
            initial_evidence = self.evidence_retriever.build_evidence_items(
                thought_id=anchor_thought_id,
                branch_kind="initial",
                candidates=initial_candidates,
                limit=self.config.reasoning.evidence_top_k_per_branch,
            )
            initial_anchor_thought = self.executor.create_initial_anchor_thought(
                thought_id=anchor_thought_id,
                task_frame=task_frame,
                candidates=initial_candidates,
                evidence_items=initial_evidence,
                parent_ids=[root_thought.thought_id],
            )
            thought_graph.add_thought(initial_anchor_thought)
            evidence_subgraph.record_branch_result(
                "initial",
                initial_candidates,
                {
                    "branch_kind": "initial",
                    "query_texts": [question, *task_frame.topic_entities],
                    "recommended_hyperedges": [candidate.to_dict() for candidate in initial_candidates],
                    "control_state": control_state.to_dict(),
                    "notes": "Initial anchoring from topic entities and answer-type hints.",
                },
            )
            evidence_subgraph.add_frontier(
                iteration=0,
                candidates=initial_candidates,
                evidence_items=initial_evidence,
                control_state=control_state.to_dict(),
            )
            self.registry.register_reasoning(task_frame, initial_anchor_thought)

        branch_heads: dict[str, ThoughtState] = {}
        stalled_steps = 0
        previous_signature = self._progress_signature(evidence_subgraph, {})
        termination_reason: str | None = None
        latest_merge_result: dict[str, Any] = {
            "frontier_hyperedge_ids": [],
            "frontier": [],
            "branch_contributions": {},
            "preferred_branches": [],
            "coverage_summary": {},
            "answer_hypotheses": [],
            "missing_requirements": [],
            "next_focus": [],
            "notes": "",
        }

        for iteration in range(1, self.config.reasoning.max_steps + 1):
            control_state.iteration = iteration
            self._log_control_state(control_state)
            self.logger.info("Iterative reasoning step %s/%s", iteration, self.config.reasoning.max_steps)

            branch_candidates: dict[str, list[Any]] = {}
            branch_results: list[dict[str, Any]] = []
            branch_thoughts: list[ThoughtState] = []

            for branch_kind in ("constraint", "relation", "anchor"):
                previous_branch = branch_heads.get(branch_kind)
                exclude_hyperedge_ids = set(evidence_subgraph.branch_support.get(branch_kind, []))
                candidates = self.evidence_retriever.retrieve_branch_candidates(
                    question=question,
                    task_frame=task_frame,
                    branch_kind=branch_kind,
                    control_state=control_state,
                    evidence_subgraph=evidence_subgraph.to_dict(),
                    exclude_hyperedge_ids=exclude_hyperedge_ids,
                )
                selected_candidates = candidates[: self.config.reasoning.branch_top_k]
                branch_candidates[branch_kind] = selected_candidates

                branch_result = self._build_branch_result(branch_kind, selected_candidates, control_state)
                branch_result["query_texts"] = list(control_state.branch_queries.get(branch_kind, []))
                branch_result["control_state"] = control_state.to_dict()
                branch_result["notes"] = f"{branch_kind} branch ranking is retrieval/scoring-led."

                thought_id = self._next_id("th")
                branch_evidence = self.evidence_retriever.build_evidence_items(
                    thought_id=thought_id,
                    branch_kind=branch_kind,
                    candidates=selected_candidates[: self.config.reasoning.evidence_top_k_per_branch],
                    limit=self.config.reasoning.evidence_top_k_per_branch,
                )
                parent_ids = self._branch_parent_ids(previous_branch, last_merge_thought, initial_anchor_thought, root_thought)
                self.executor.retire_previous_branch(previous_branch)
                branch_thought = self.executor.create_branch_thought(
                    thought_id=thought_id,
                    task_frame=task_frame,
                    branch_kind=branch_kind,
                    iteration=iteration,
                    branch_result=branch_result,
                    candidates=selected_candidates,
                    evidence_items=branch_evidence,
                    parent_ids=parent_ids,
                )
                thought_graph.add_thought(branch_thought)
                branch_heads[branch_kind] = branch_thought
                branch_thoughts.append(branch_thought)
                evidence_subgraph.record_branch_result(branch_kind, selected_candidates, branch_result)
                self.registry.register_reasoning(task_frame, branch_thought)
                branch_results.append(branch_result)

            if not branch_results:
                termination_reason = "no_branch_updates"
                break

            global_frontier, latest_merge_result = self.evidence_retriever.fuse_frontier(
                task_frame=task_frame,
                branch_candidates=branch_candidates,
                evidence_subgraph=evidence_subgraph.to_dict(),
                control_state=control_state,
                top_k=self.config.reasoning.branch_top_k,
            )
            frontier_evidence = self.evidence_retriever.build_evidence_items(
                thought_id=self._next_id("evset"),
                branch_kind="frontier",
                candidates=global_frontier,
                limit=min(self.config.retrieval.evidence_keep, len(global_frontier)),
            )
            evidence_subgraph.add_frontier(
                iteration=iteration,
                candidates=global_frontier,
                evidence_items=frontier_evidence,
                control_state=control_state.to_dict(),
            )

            merge_thought = self.executor.create_merge_thought(
                thought_id=self._next_id("th"),
                task_frame=task_frame,
                iteration=iteration,
                merge_result=latest_merge_result,
                evidence_items=frontier_evidence,
                parent_ids=[thought.thought_id for thought in branch_thoughts],
            )
            thought_graph.add_thought(merge_thought)
            last_merge_thought = merge_thought
            self.registry.register_reasoning(task_frame, merge_thought)

            sufficiency = self.llm_service.judge_sufficiency(
                question=question,
                task_frame=task_frame,
                merge_result=latest_merge_result,
                evidence_subgraph=evidence_subgraph.to_dict(),
                iteration=iteration,
                retrieval_control_state=control_state.to_dict(),
            )
            latest_merge_result["missing_requirements"] = list(sufficiency.get("missing_requirements", []))
            latest_merge_result["next_focus"] = list(sufficiency.get("next_focus", []))
            thought_graph.append_history(
                "iteration_completed",
                {
                    "iteration": iteration,
                    "retrieval_control_state": control_state.to_dict(),
                    "branch_results": branch_results,
                    "merge_result": latest_merge_result,
                    "sufficiency": sufficiency,
                },
            )
            self.trace_store.log_event(
                "iteration_completed",
                {
                    "iteration": iteration,
                    "retrieval_control_state": control_state.to_dict(),
                    "branch_results": branch_results,
                    "merge_result": latest_merge_result,
                    "sufficiency": sufficiency,
                },
            )

            if bool(sufficiency.get("enough")):
                termination_reason = "evidence_sufficient"
                break

            current_signature = self._progress_signature(evidence_subgraph, latest_merge_result)
            if current_signature == previous_signature:
                stalled_steps += 1
            else:
                stalled_steps = 0
                previous_signature = current_signature
            if stalled_steps >= self.config.reasoning.max_stalled_steps:
                termination_reason = "stalled"
                break

            control_state = self._advance_control_state(control_state, sufficiency, latest_merge_result)

        if not termination_reason:
            termination_reason = "budget_exhausted"

        thought_graph.termination_reason = termination_reason
        final_payload = self.llm_service.synthesize_answer(
            question=question,
            task_frame=task_frame,
            thought_graph=thought_graph,
            evidence_subgraph=evidence_subgraph.to_dict(),
            merge_result=latest_merge_result,
        )
        final_parent_ids = [
            thought.thought_id
            for thought in [last_merge_thought, *branch_heads.values()]
            if thought is not None
        ]
        answer_thought = self.executor.create_answer_thought(
            thought_id=self._next_id("th"),
            task_frame=task_frame,
            final_payload=final_payload,
            evidence_items=list(evidence_subgraph.evidence),
            parent_ids=final_parent_ids,
        )
        thought_graph.add_thought(answer_thought)
        thought_graph.status = "done"
        thought_graph.final_answer = final_payload
        task_frame.mark_slot("target-0", evidence_id=answer_thought.thought_id, status="verified", note="Final answer synthesized.")

        return {
            "task_frame": task_frame.to_dict(),
            "thought_graph": thought_graph.to_dict(),
            "final_answer": final_payload,
            "evidence_subgraph": evidence_subgraph.to_dict(),
        }

    def _initial_control_state(self) -> RetrievalControlState:
        weights = self._normalize_weights(
            {
                "constraint": self.config.reasoning.base_constraint_weight,
                "relation": self.config.reasoning.base_relation_weight,
                "anchor": self.config.reasoning.base_anchor_weight,
            }
        )
        return RetrievalControlState(iteration=0, branch_weights=weights, notes=["Initialized control state."])

    def _advance_control_state(
        self,
        previous: RetrievalControlState,
        sufficiency: dict[str, Any],
        merge_result: dict[str, Any],
    ) -> RetrievalControlState:
        missing_requirements = [
            str(item).strip()
            for item in sufficiency.get("missing_requirements", [])
            if str(item).strip()
        ]
        next_focus = [str(item).strip() for item in sufficiency.get("next_focus", []) if str(item).strip()]
        weights = {
            "constraint": self.config.reasoning.base_constraint_weight,
            "relation": self.config.reasoning.base_relation_weight,
            "anchor": self.config.reasoning.base_anchor_weight,
        }
        notes: list[str] = []

        for text in [*missing_requirements, *next_focus]:
            lowered = text.lower()
            if any(marker in lowered for marker in ("constraint", "time", "location", "type", "must", "direct supporting evidence")):
                weights["constraint"] += self.config.reasoning.control_weight_step
                notes.append(f"Raised constraint weight due to: {text}")
            if any(marker in lowered for marker in ("relation", "skeleton", "intent", "shared concept", "close the relation", "relation closure")):
                weights["relation"] += self.config.reasoning.control_weight_step
                notes.append(f"Raised relation weight due to: {text}")
            if any(marker in lowered for marker in ("anchor", "bridge", "connect", "connector", "coverage", "entity", "cluster")):
                weights["anchor"] += self.config.reasoning.control_weight_step
                notes.append(f"Raised anchor weight due to: {text}")

        if not notes:
            preferred_branches = merge_result.get("preferred_branches", [])
            if preferred_branches:
                preferred = str(preferred_branches[0]).strip()
                if preferred in weights:
                    weights[preferred] += self.config.reasoning.control_weight_step / 2.0
                    notes.append(f"Raised {preferred} weight based on previous frontier contributions.")

        normalized_weights = self._normalize_weights(weights)
        return RetrievalControlState(
            iteration=previous.iteration + 1,
            branch_weights=normalized_weights,
            missing_requirements=missing_requirements,
            next_focus=next_focus,
            notes=notes or ["Control state carried forward without additional emphasis."],
        )

    def _build_branch_result(
        self,
        branch_kind: str,
        candidates: list[Any],
        control_state: RetrievalControlState,
    ) -> dict[str, Any]:
        return {
            "branch_kind": branch_kind,
            "branch_weight": control_state.branch_weights.get(branch_kind, 0.0),
            "recommended_hyperedges": [candidate.to_dict() for candidate in candidates],
            "recommended_ids": [candidate.hyperedge_id for candidate in candidates],
            "top_branch_score": max((candidate.branch_score for candidate in candidates), default=0.0),
            "gain_summary": {
                "coverage_gain": max((candidate.coverage_gain for candidate in candidates), default=0.0),
                "constraint_gain": max((candidate.constraint_gain for candidate in candidates), default=0.0),
                "relation_gain": max((candidate.relation_gain for candidate in candidates), default=0.0),
                "connector_gain": max((candidate.connector_gain for candidate in candidates), default=0.0),
                "novelty_gain": max((candidate.novelty_gain for candidate in candidates), default=0.0),
                "focus_gain": max((candidate.focus_gain for candidate in candidates), default=0.0),
            },
        }

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        total = sum(max(value, 0.0) for value in weights.values())
        if total <= 0:
            return {"constraint": 1 / 3, "relation": 1 / 3, "anchor": 1 / 3}
        return {key: max(value, 0.0) / total for key, value in weights.items()}

    def _log_control_state(self, control_state: RetrievalControlState) -> None:
        payload = control_state.to_dict()
        self.trace_store.log_event("retrieval_control_state", payload)
        self.logger.info(
            "Control state iteration=%s weights=%s focus=%s missing=%s",
            payload["iteration"],
            payload["branch_weights"],
            payload["next_focus"],
            payload["missing_requirements"],
        )

    def _branch_parent_ids(
        self,
        previous_branch: ThoughtState | None,
        last_merge_thought: ThoughtState | None,
        initial_anchor_thought: ThoughtState | None,
        root_thought: ThoughtState,
    ) -> list[str]:
        parent_ids: list[str] = []
        for candidate in (previous_branch, last_merge_thought, initial_anchor_thought, root_thought):
            if candidate is None:
                continue
            if candidate.thought_id not in parent_ids:
                parent_ids.append(candidate.thought_id)
        return parent_ids

    def _progress_signature(
        self,
        evidence_subgraph: EvidenceSubgraph,
        merge_result: dict[str, Any],
    ) -> tuple[int, int, tuple[str, ...], tuple[str, ...]]:
        return (
            len(evidence_subgraph.hyperedge_ids),
            len(evidence_subgraph.chunk_ids),
            tuple(str(item) for item in merge_result.get("frontier_hyperedge_ids", [])),
            tuple(str(item) for item in merge_result.get("answer_hypotheses", [])),
        )

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:04d}"
