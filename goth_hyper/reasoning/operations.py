from __future__ import annotations

import logging
from typing import Any, Callable

from ..logging_utils import TraceStore
from ..models import EvidenceItem, Grounding, TaskFrame, ThoughtGraph, ThoughtState
from .taskframe import TaskFrameRegistry


class ThoughtOperationExecutor:
    def __init__(
        self,
        llm_service: Any,
        registry: TaskFrameRegistry,
        evidence_score_threshold: float,
        logger: logging.Logger,
        trace_store: TraceStore,
    ) -> None:
        self.llm_service = llm_service
        self.registry = registry
        self.evidence_score_threshold = evidence_score_threshold
        self.logger = logger
        self.trace_store = trace_store

    def execute(
        self,
        question: str,
        task_frame: TaskFrame,
        thought_graph: ThoughtGraph,
        thought: ThoughtState,
        evidence_items: list[EvidenceItem],
        related_thoughts: list[ThoughtState],
        id_factory: Callable[[str], str],
    ) -> dict[str, Any]:
        thought.grounding.update_with_evidence(evidence_items)
        decision = self.llm_service.decide_operation(question, task_frame, thought, evidence_items, related_thoughts)
        operation = str(decision.get("operation", "verify")).lower()

        result: dict[str, Any] = {
            "operation": operation,
            "new_thought_ids": [],
            "verified_reasoning_ids": [],
        }

        if operation == "expand":
            thought.status = "expanded"
            new_thoughts = decision.get("new_thoughts", []) or [
                {
                    "content": f"Refine branch from: {thought.content}",
                    "objective": thought.objective,
                    "slot_id": thought.slot_id,
                    "metadata": {"intent": thought.metadata.get("intent", "followup")},
                }
            ]
            for payload in new_thoughts:
                new_state = self._build_child_thought(id_factory, thought, payload)
                thought_graph.add_thought(new_state)
                result["new_thought_ids"].append(new_state.thought_id)

        elif operation == "revise":
            thought.status = "revised"
            revised_payload = {
                "content": decision.get("revised_content") or thought.content,
                "objective": thought.objective,
                "slot_id": thought.slot_id,
                "metadata": {"intent": thought.metadata.get("intent", "revise")},
            }
            revised_state = self._build_child_thought(id_factory, thought, revised_payload)
            thought_graph.add_thought(revised_state)
            result["new_thought_ids"].append(revised_state.thought_id)

        elif operation == "merge":
            thought.status = "merged"
            merge_with = [thought_id for thought_id in decision.get("merge_with_thought_ids", []) if thought_id in thought_graph.thoughts]
            for merge_id in merge_with:
                thought_graph.get(merge_id).status = "merged"
            merged_payload = (
                decision.get("new_thoughts")
                or [
                    {
                        "content": decision.get("revised_content") or thought.content,
                        "objective": thought.objective,
                        "slot_id": thought.slot_id,
                        "metadata": {"intent": "merge"},
                    }
                ]
            )[0]
            merged_state = self._build_child_thought(
                id_factory,
                thought,
                merged_payload,
                extra_parent_ids=merge_with,
            )
            thought_graph.add_thought(merged_state)
            result["new_thought_ids"].append(merged_state.thought_id)

        else:
            verification = decision.get("verification", {})
            verdict = str(verification.get("verdict", "insufficient")).lower()
            if verdict == "supported":
                thought.status = "verified"
                selected_ids = verification.get("evidence_ids", [])
                if isinstance(selected_ids, list) and selected_ids:
                    selected_items = [item for item in evidence_items if item.evidence_id in set(selected_ids)]
                    if selected_items:
                        thought.grounding.evidence = []
                        thought.grounding.chunk_ids = []
                        thought.grounding.node_ids = []
                        thought.grounding.update_with_evidence(selected_items)
                thought.score = max(thought.score, self.evidence_score_threshold)
                self.registry.register_reasoning(task_frame, thought)
                result["verified_reasoning_ids"].append(thought.thought_id)
            elif verdict == "refuted":
                thought.status = "rejected"
            else:
                thought.status = str(decision.get("new_status", "active"))

        thought_graph.recompute_frontier()
        thought_graph.append_history(
            "operation_executed",
            {
                "thought_id": thought.thought_id,
                "operation": operation,
                "decision": decision,
                "new_thought_ids": result["new_thought_ids"],
                "verified_reasoning_ids": result["verified_reasoning_ids"],
            },
        )
        self.trace_store.log_event(
            "operation_executed",
            {
                "thought_id": thought.thought_id,
                "operation": operation,
                "new_thought_ids": result["new_thought_ids"],
                "verified_reasoning_ids": result["verified_reasoning_ids"],
            },
        )
        return result

    def _build_child_thought(
        self,
        id_factory: Callable[[str], str],
        parent_thought: ThoughtState,
        payload: dict[str, Any],
        extra_parent_ids: list[str] | None = None,
    ) -> ThoughtState:
        return ThoughtState(
            thought_id=id_factory("th"),
            kind="reasoning",
            content=str(payload.get("content", parent_thought.content)),
            objective=str(payload.get("objective", parent_thought.objective)),
            slot_id=str(payload.get("slot_id")) if payload.get("slot_id") is not None else parent_thought.slot_id,
            grounding=Grounding(
                anchor_texts=list(parent_thought.grounding.anchor_texts),
                node_ids=list(parent_thought.grounding.node_ids),
                chunk_ids=list(parent_thought.grounding.chunk_ids),
                evidence=list(parent_thought.grounding.evidence),
                notes=list(parent_thought.grounding.notes),
            ),
            status="active",
            parent_ids=[parent_thought.thought_id] + list(extra_parent_ids or []),
            metadata=dict(payload.get("metadata", parent_thought.metadata)),
        )
