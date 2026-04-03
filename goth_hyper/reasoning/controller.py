from __future__ import annotations

import logging
from typing import Any

from ..config import Config
from ..data.loaders import DatasetBundle
from ..logging_utils import TraceStore
from ..models import EvidenceSubgraph, HyperedgeCandidate, TaskFrame, ThoughtGraph, ThoughtState
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

        evidence_subgraph = EvidenceSubgraph()
        initial_candidates = list(anchor_payload.get("initial_hyperedge_candidates", []))[
            : self.config.reasoning.evidence_top_k_per_branch
        ]
        initial_anchor_thought = None
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
            evidence_subgraph.add_support("initial", initial_candidates, initial_evidence)
            self.registry.register_reasoning(task_frame, initial_anchor_thought)

        branch_heads: dict[str, ThoughtState] = {}
        stalled_steps = 0
        previous_signature = self._progress_signature(task_frame, evidence_subgraph, {})
        termination_reason: str | None = None
        latest_merge_result: dict[str, Any] = {
            "consensus_answer": "",
            "agreement_groups": [],
            "conflicts": [],
            "preferred_branches": [],
            "missing_requirements": [],
            "notes": "",
        }

        for iteration in range(1, self.config.reasoning.max_steps + 1):
            self.logger.info("Iterative reasoning step %s/%s", iteration, self.config.reasoning.max_steps)
            branch_results: list[dict[str, Any]] = []
            iteration_evidence_items = []

            for branch_kind in ("constraint", "relation", "anchor"):
                previous_branch = branch_heads.get(branch_kind)
                exclude_hyperedge_ids = set(evidence_subgraph.branch_support.get(branch_kind, []))
                candidates = self.evidence_retriever.retrieve_branch_candidates(
                    question=question,
                    task_frame=task_frame,
                    branch_kind=branch_kind,
                    evidence_subgraph=evidence_subgraph.to_dict(),
                    exclude_hyperedge_ids=exclude_hyperedge_ids,
                )
                if not candidates and exclude_hyperedge_ids:
                    candidates = self.evidence_retriever.retrieve_branch_candidates(
                        question=question,
                        task_frame=task_frame,
                        branch_kind=branch_kind,
                        evidence_subgraph=evidence_subgraph.to_dict(),
                        exclude_hyperedge_ids=set(),
                    )

                selection_payload = self.llm_service.select_branch_candidates(
                    question=question,
                    task_frame=task_frame,
                    branch_kind=branch_kind,
                    candidate_hyperedges=candidates,
                    evidence_subgraph=evidence_subgraph.to_dict(),
                    top_k=self.config.reasoning.branch_top_k,
                )
                selected_candidates = self._select_candidates(candidates, selection_payload)
                if not selected_candidates and candidates:
                    selected_candidates = candidates[: self.config.reasoning.evidence_top_k_per_branch]

                thought_id = self._next_id("th")
                selected_for_evidence = selected_candidates[: self.config.reasoning.evidence_top_k_per_branch]
                evidence_items = self.evidence_retriever.build_evidence_items(
                    thought_id=thought_id,
                    branch_kind=branch_kind,
                    candidates=selected_for_evidence,
                    limit=self.config.reasoning.evidence_top_k_per_branch,
                )
                parent_ids = self._branch_parent_ids(previous_branch, last_merge_thought, initial_anchor_thought, root_thought)
                self.executor.retire_previous_branch(previous_branch)
                branch_thought = self.executor.create_branch_thought(
                    thought_id=thought_id,
                    task_frame=task_frame,
                    branch_kind=branch_kind,
                    iteration=iteration,
                    selection_payload=selection_payload,
                    candidates=selected_candidates,
                    evidence_items=evidence_items,
                    parent_ids=parent_ids,
                )
                thought_graph.add_thought(branch_thought)
                branch_heads[branch_kind] = branch_thought
                iteration_evidence_items.extend(evidence_items)
                evidence_subgraph.add_support(
                    branch_kind,
                    selected_for_evidence,
                    evidence_items,
                    branch_answer={
                        "branch_kind": branch_kind,
                        "candidate_answer": selection_payload.get("candidate_answer", ""),
                        "confidence": selection_payload.get("confidence", 0.0),
                        "selected_hyperedge_ids": [candidate.hyperedge_id for candidate in selected_candidates],
                        "supporting_facts": selection_payload.get("supporting_facts", []),
                        "missing_requirements": selection_payload.get("missing_requirements", []),
                        "thought_id": branch_thought.thought_id,
                    },
                )
                self.registry.register_reasoning(task_frame, branch_thought)
                branch_results.append(
                    {
                        "branch_kind": branch_kind,
                        "thought_id": branch_thought.thought_id,
                        "candidate_answer": selection_payload.get("candidate_answer", ""),
                        "confidence": selection_payload.get("confidence", 0.0),
                        "selected_hyperedge_ids": [candidate.hyperedge_id for candidate in selected_candidates],
                        "supporting_facts": selection_payload.get("supporting_facts", []),
                        "missing_requirements": selection_payload.get("missing_requirements", []),
                        "selected_hyperedges": [candidate.to_dict() for candidate in selected_candidates],
                    }
                )

            if not branch_results:
                termination_reason = "no_branch_updates"
                break

            latest_merge_result = self.llm_service.reconcile_branches(
                question=question,
                task_frame=task_frame,
                branch_results=branch_results,
                evidence_subgraph=evidence_subgraph.to_dict(),
            )
            merge_thought = self.executor.create_merge_thought(
                thought_id=self._next_id("th"),
                task_frame=task_frame,
                iteration=iteration,
                merge_result=latest_merge_result,
                evidence_items=iteration_evidence_items,
                parent_ids=[branch_results_item["thought_id"] for branch_results_item in branch_results],
            )
            thought_graph.add_thought(merge_thought)
            last_merge_thought = merge_thought
            self.registry.register_reasoning(task_frame, merge_thought)

            sufficiency = self.llm_service.judge_sufficiency(
                question=question,
                task_frame=task_frame,
                branch_results=branch_results,
                merge_result=latest_merge_result,
                evidence_subgraph=evidence_subgraph.to_dict(),
                iteration=iteration,
            )
            thought_graph.append_history(
                "iteration_completed",
                {
                    "iteration": iteration,
                    "branch_results": branch_results,
                    "merge_result": latest_merge_result,
                    "sufficiency": sufficiency,
                },
            )
            self.trace_store.log_event(
                "iteration_completed",
                {
                    "iteration": iteration,
                    "branch_results": branch_results,
                    "merge_result": latest_merge_result,
                    "sufficiency": sufficiency,
                },
            )

            if bool(sufficiency.get("enough")):
                termination_reason = "evidence_sufficient"
                break
            if task_frame.is_satisfied() and str(latest_merge_result.get("consensus_answer", "")).strip():
                termination_reason = "taskframe_satisfied"
                break

            current_signature = self._progress_signature(task_frame, evidence_subgraph, latest_merge_result)
            if current_signature == previous_signature:
                stalled_steps += 1
            else:
                stalled_steps = 0
                previous_signature = current_signature
            if stalled_steps >= self.config.reasoning.max_stalled_steps:
                termination_reason = "stalled"
                break

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

    def _select_candidates(
        self,
        candidates: list[HyperedgeCandidate],
        selection_payload: dict[str, Any],
    ) -> list[HyperedgeCandidate]:
        selected_ids = selection_payload.get("selected_hyperedge_ids", [])
        candidate_map = {candidate.hyperedge_id: candidate for candidate in candidates}
        selected: list[HyperedgeCandidate] = []
        if isinstance(selected_ids, list):
            for hyperedge_id in selected_ids:
                candidate = candidate_map.get(str(hyperedge_id))
                if candidate is not None:
                    selected.append(candidate)
        return selected

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
        task_frame: TaskFrame,
        evidence_subgraph: EvidenceSubgraph,
        merge_result: dict[str, Any],
    ) -> tuple[int, int, str, int]:
        return (
            len(evidence_subgraph.hyperedge_ids),
            len(evidence_subgraph.chunk_ids),
            str(merge_result.get("consensus_answer", "")).strip(),
            len(task_frame.get_open_slots()),
        )

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:04d}"
