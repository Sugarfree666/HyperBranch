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
                if thought.kind == "reasoning" and thought.status == "active"
            }
            for thought in frontier:
                thought.grounding.update_with_evidence(evidence_cache.get(thought.thought_id, []))
            self.scorer.score_thoughts(question, frontier)
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
            kind="reasoning",
            content=question,
            objective=question,
            slot_id=None,
            grounding=Grounding(anchor_texts=task_frame.anchors, notes=["question-root"]),
            status="root",
        )
        thought_graph = ThoughtGraph(question=question, root_id=root_thought.thought_id)
        thought_graph.add_thought(root_thought)

        open_slots = task_frame.get_open_slots()
        for slot in open_slots:
            content = self._seed_content_for_slot(task_frame, slot.slot_id, slot.text)
            thought = ThoughtState(
                thought_id=self._next_id("th"),
                kind="reasoning",
                content=content,
                objective=slot.text,
                slot_id=slot.slot_id,
                grounding=Grounding(
                    anchor_texts=list(task_frame.anchors),
                    notes=[f"slot-kind:{slot.kind}", f"slot-id:{slot.slot_id}"],
                ),
                status="active",
                parent_ids=[root_thought.thought_id],
                metadata={"intent": slot.kind},
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
        current_slot_id = current_thought.slot_id

        def score(other: ThoughtState) -> tuple[int, float]:
            shared_slot = 1 if current_slot_id and other.slot_id == current_slot_id else 0
            shared_anchors = len(current_anchor_set & set(other.grounding.anchor_texts))
            shared_chunks = len(current_chunk_set & set(other.grounding.chunk_ids))
            return (shared_slot + shared_anchors + shared_chunks, other.score)

        others.sort(key=score, reverse=True)
        return others[:5]

    def _finalize_answer(self, question: str, task_frame: TaskFrame, thought_graph: ThoughtGraph) -> dict[str, Any]:
        verified_reasoning = [
            thought
            for thought in thought_graph.thoughts.values()
            if thought.kind == "reasoning" and thought.status == "verified"
        ]
        if len(verified_reasoning) < self.config.reasoning.min_verified_reasoning:
            fallback_candidates = [thought for thought in thought_graph.thoughts.values() if thought.kind == "reasoning"]
            fallback_candidates.sort(key=lambda item: item.score, reverse=True)
            verified_reasoning = fallback_candidates[: self.config.reasoning.min_verified_reasoning]

        final_payload = self.llm_service.synthesize_answer(question, task_frame, thought_graph, verified_reasoning)
        parent_ids = [thought.thought_id for thought in verified_reasoning[:4]]
        answer_thought = ThoughtState(
            thought_id=self._next_id("th"),
            kind="answer",
            content=str(final_payload.get("answer", "")),
            objective=task_frame.target,
            slot_id="target-0",
            grounding=Grounding(
                anchor_texts=list(task_frame.anchors),
                chunk_ids=[chunk_id for thought in verified_reasoning for chunk_id in thought.grounding.chunk_ids][:8],
                evidence=[item for thought in verified_reasoning for item in thought.grounding.evidence][:8],
                notes=[str(final_payload.get("reasoning_summary", ""))],
            ),
            score=max((thought.score for thought in verified_reasoning), default=0.0),
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
        self.logger.info("Final answer synthesized with %s reasoning parents", len(parent_ids))
        self.trace_store.log_event(
            "final_answer_synthesized",
            {
                "answer_thought_id": answer_thought.thought_id,
                "reasoning_parent_count": len(parent_ids),
            },
        )
        return final_payload

    def _seed_content_for_slot(self, task_frame: TaskFrame, slot_id: str, objective: str) -> str:
        return f"Resolve task slot '{slot_id}' for question '{task_frame.question}': {objective}"

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:04d}"
