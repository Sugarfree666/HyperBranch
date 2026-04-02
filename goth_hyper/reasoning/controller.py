from __future__ import annotations

import logging
from typing import Any

from ..config import Config
from ..data.loaders import DatasetBundle
from ..logging_utils import TraceStore
from ..models import Grounding, TaskFrame, ThoughtGraph, ThoughtState
from ..retrieval.evidence import EvidenceRetriever
from .operations import ThoughtOperationExecutor
from .scoring import ThoughtScorer
from .taskframe import TaskFrameBuilder, TaskFrameRegistry


class ThoughtController:
    def __init__(
        self,
        config: Config,
        dataset: DatasetBundle,
        taskframe_builder: TaskFrameBuilder,
        registry: TaskFrameRegistry,
        scorer: ThoughtScorer,
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
        thought_graph = self._initialize_thought_graph(question, task_frame)

        for step in range(1, self.config.reasoning.max_steps + 1):
            self.logger.info("Reasoning step %s/%s", step, self.config.reasoning.max_steps)
            frontier = thought_graph.active_frontier()
            evidence_cache = {
                thought.thought_id: self.evidence_retriever.retrieve(thought)
                for thought in frontier
                if thought.role in {"hypothesis", "constraint", "bridge"} and thought.status == "active"
            }
            self.scorer.score_thoughts(question, frontier, evidence_by_thought=evidence_cache)
            shortlisted = self.scorer.shortlist(frontier)
            if not shortlisted:
                thought_graph.termination_reason = "no_active_frontier"
                break

            selected_ids = self.llm_service.select_thoughts(
                question,
                task_frame,
                shortlisted,
                top_k=self.config.reasoning.llm_top_k,
            )
            if not selected_ids:
                selected_ids = [thought.thought_id for thought in shortlisted[: self.config.reasoning.llm_top_k]]
            self.trace_store.log_event("scheduler_selected", {"step": step, "thought_ids": selected_ids})

            for thought_id in selected_ids:
                current_thought = thought_graph.get(thought_id)
                related_thoughts = self._related_thoughts(thought_graph, current_thought)
                evidence_items = evidence_cache.get(thought_id)
                if evidence_items is None:
                    evidence_items = self.evidence_retriever.retrieve(current_thought)
                self.executor.execute(
                    question=question,
                    task_frame=task_frame,
                    thought_graph=thought_graph,
                    thought=current_thought,
                    evidence_items=evidence_items,
                    related_thoughts=related_thoughts,
                    id_factory=self._next_id,
                )

            if task_frame.is_satisfied():
                thought_graph.termination_reason = "taskframe_satisfied"
                break

        final_payload = self._finalize_answer(question, task_frame, thought_graph)
        return {
            "task_frame": task_frame.to_dict(),
            "thought_graph": thought_graph.to_dict(),
            "final_answer": final_payload,
        }

    def _initialize_thought_graph(self, question: str, task_frame: TaskFrame) -> ThoughtGraph:
        root_thought = ThoughtState(
            thought_id=self._next_id("th"),
            role="root",
            content=question,
            grounding=Grounding(anchor_texts=task_frame.anchors, notes=["question-root"]),
            status="expanded",
        )
        thought_graph = ThoughtGraph(question=question, root_id=root_thought.thought_id)
        thought_graph.add_thought(root_thought)

        seed_payloads = self.llm_service.initialize_seed_thoughts(question, task_frame)
        if not seed_payloads:
            seed_payloads = [
                {
                    "role": "hypothesis",
                    "content": task_frame.hypothesis_template,
                    "grounding_hints": {"anchors": task_frame.anchors},
                }
            ]
        for payload in seed_payloads:
            hints = payload.get("grounding_hints") if isinstance(payload.get("grounding_hints"), dict) else {}
            thought = ThoughtState(
                thought_id=self._next_id("th"),
                role=str(payload.get("role", "hypothesis")),
                content=str(payload.get("content", task_frame.hypothesis_template)),
                grounding=Grounding(
                    anchor_texts=list(hints.get("anchors", task_frame.anchors)),
                    notes=list(hints.get("notes", [])),
                ),
                status="active",
                parent_ids=[root_thought.thought_id],
            )
            thought_graph.add_thought(thought)
        self.logger.info("Initialized ThoughtGraph with %s seed thoughts", len(thought_graph.thoughts) - 1)
        self.trace_store.log_event(
            "thought_graph_initialized",
            {
                "root_id": root_thought.thought_id,
                "seed_thought_count": len(thought_graph.thoughts) - 1,
            },
        )
        return thought_graph

    def _related_thoughts(self, thought_graph: ThoughtGraph, current_thought: ThoughtState) -> list[ThoughtState]:
        others = [thought for thought in thought_graph.thoughts.values() if thought.thought_id != current_thought.thought_id]
        current_anchor_set = set(current_thought.grounding.anchor_texts)
        current_chunk_set = set(current_thought.grounding.chunk_ids)

        def score(other: ThoughtState) -> tuple[int, float]:
            shared_anchors = len(current_anchor_set & set(other.grounding.anchor_texts))
            shared_chunks = len(current_chunk_set & set(other.grounding.chunk_ids))
            return (shared_anchors + shared_chunks, other.score)

        others.sort(key=score, reverse=True)
        return others[:5]

    def _finalize_answer(self, question: str, task_frame: TaskFrame, thought_graph: ThoughtGraph) -> dict[str, Any]:
        verified_evidence = [
            thought
            for thought in thought_graph.thoughts.values()
            if thought.role == "evidence" and thought.status == "verified"
        ]
        if len(verified_evidence) < self.config.reasoning.min_verified_evidence:
            fallback_candidates = [thought for thought in thought_graph.thoughts.values() if thought.role == "evidence"]
            fallback_candidates.sort(key=lambda item: item.score, reverse=True)
            verified_evidence = fallback_candidates[: self.config.reasoning.min_verified_evidence]

        final_payload = self.llm_service.synthesize_answer(question, task_frame, thought_graph, verified_evidence)
        parent_ids = [thought.thought_id for thought in verified_evidence[:4]]
        answer_thought = ThoughtState(
            thought_id=self._next_id("th"),
            role="answer",
            content=str(final_payload.get("answer", "")),
            grounding=Grounding(
                anchor_texts=list(task_frame.anchors),
                chunk_ids=[chunk_id for thought in verified_evidence for chunk_id in thought.grounding.chunk_ids][:8],
                evidence_ids=parent_ids,
                notes=[str(final_payload.get("reasoning_summary", ""))],
            ),
            score=max((thought.score for thought in verified_evidence), default=0.0),
            status="completed",
            parent_ids=parent_ids,
            metadata=final_payload,
        )
        thought_graph.add_thought(answer_thought)
        thought_graph.status = "done"
        thought_graph.final_answer = final_payload
        if not thought_graph.termination_reason:
            thought_graph.termination_reason = "budget_exhausted"
        task_frame.mark_slot("target-0", evidence_id=answer_thought.thought_id, status="verified", note="Final answer synthesized.")
        self.logger.info("Final answer synthesized with %s evidence parents", len(parent_ids))
        self.trace_store.log_event(
            "final_answer_synthesized",
            {
                "answer_thought_id": answer_thought.thought_id,
                "evidence_parent_count": len(parent_ids),
            },
        )
        return final_payload

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:04d}"
