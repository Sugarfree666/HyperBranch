from __future__ import annotations

import logging
from typing import Any, Callable

from ..logging_utils import TraceStore
from ..models import EvidenceItem, Grounding, TaskFrame, ThoughtGraph, ThoughtState
from ..utils import short_text
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
        evidence_thoughts = self._materialize_evidence_thoughts(thought_graph, thought, evidence_items)
        decision = self.llm_service.decide_operation(question, task_frame, thought, evidence_items, related_thoughts)
        operation = str(decision.get("operation", "verify")).lower()

        result: dict[str, Any] = {
            "operation": operation,
            "new_thought_ids": [],
            "verified_evidence_ids": [],
        }

        if operation == "expand":
            thought.status = "expanded"
            new_thoughts = decision.get("new_thoughts", []) or [
                {
                    "role": "bridge",
                    "content": f"Refine branch from: {thought.content}",
                    "grounding_hints": {"anchors": thought.grounding.anchor_texts},
                }
            ]
            for payload in new_thoughts:
                new_state = self._build_child_thought(id_factory, thought, payload, evidence_thoughts)
                thought_graph.add_thought(new_state)
                result["new_thought_ids"].append(new_state.thought_id)

        elif operation == "revise":
            thought.status = "revised"
            revised_payload = {
                "role": thought.role,
                "content": decision.get("revised_content") or thought.content,
                "grounding_hints": {"anchors": thought.grounding.anchor_texts},
            }
            revised_state = self._build_child_thought(id_factory, thought, revised_payload, evidence_thoughts)
            thought_graph.add_thought(revised_state)
            result["new_thought_ids"].append(revised_state.thought_id)

        elif operation == "merge":
            thought.status = "merged"
            merge_with = [thought_id for thought_id in decision.get("merge_with_thought_ids", []) if thought_id in thought_graph.thoughts]
            for merge_id in merge_with:
                thought_graph.get(merge_id).status = "merged"
            merged_payload = (decision.get("new_thoughts") or [{"role": thought.role, "content": decision.get("revised_content") or thought.content}])[0]
            merged_state = self._build_child_thought(
                id_factory,
                thought,
                merged_payload,
                evidence_thoughts,
                extra_parent_ids=merge_with,
            )
            thought_graph.add_thought(merged_state)
            result["new_thought_ids"].append(merged_state.thought_id)

        else:
            verification = decision.get("verification", {})
            verdict = str(verification.get("verdict", "insufficient")).lower()
            selected_ids = verification.get("evidence_ids", [])
            if not isinstance(selected_ids, list):
                selected_ids = []
            selected_set = set(selected_ids) if selected_ids else {
                evidence_thought.thought_id for evidence_thought in evidence_thoughts[:1]
            }
            if verdict == "supported":
                thought.status = "verified"
                for evidence_thought in evidence_thoughts:
                    if evidence_thought.thought_id in selected_set:
                        evidence_thought.status = "verified"
                        evidence_thought.score = max(evidence_thought.score, 0.5)
                        self.registry.register_evidence(task_frame, evidence_thought)
                        result["verified_evidence_ids"].append(evidence_thought.thought_id)
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
                "verified_evidence_ids": result["verified_evidence_ids"],
            },
        )
        self.trace_store.log_event(
            "operation_executed",
            {
                "thought_id": thought.thought_id,
                "operation": operation,
                "new_thought_ids": result["new_thought_ids"],
                "verified_evidence_ids": result["verified_evidence_ids"],
            },
        )
        return result

    def _materialize_evidence_thoughts(
        self,
        thought_graph: ThoughtGraph,
        parent_thought: ThoughtState,
        evidence_items: list[EvidenceItem],
    ) -> list[ThoughtState]:
        evidence_thoughts: list[ThoughtState] = []
        for evidence in evidence_items:
            if evidence.evidence_id in thought_graph.thoughts:
                evidence_thoughts.append(thought_graph.get(evidence.evidence_id))
                continue
            evidence_state = ThoughtState(
                thought_id=evidence.evidence_id,
                role="evidence",
                content=short_text(evidence.content, 900),
                grounding=Grounding(
                    anchor_texts=list(parent_thought.grounding.anchor_texts),
                    node_ids=list(evidence.source_node_ids),
                    chunk_ids=[evidence.chunk_id],
                    evidence_ids=[evidence.evidence_id],
                    notes=list(evidence.notes),
                ),
                score=evidence.score,
                status="active",
                parent_ids=[parent_thought.thought_id],
                metadata={"source_edge_ids": list(evidence.source_edge_ids)},
            )
            if evidence.score >= self.evidence_score_threshold:
                evidence_state.metadata["high_score_candidate"] = True
            thought_graph.add_thought(evidence_state)
            evidence_thoughts.append(evidence_state)
        return evidence_thoughts

    def _build_child_thought(
        self,
        id_factory: Callable[[str], str],
        parent_thought: ThoughtState,
        payload: dict[str, Any],
        evidence_thoughts: list[ThoughtState],
        extra_parent_ids: list[str] | None = None,
    ) -> ThoughtState:
        hints = payload.get("grounding_hints") if isinstance(payload.get("grounding_hints"), dict) else {}
        evidence_ids = [thought.thought_id for thought in evidence_thoughts[:2]]
        chunk_ids = [chunk_id for thought in evidence_thoughts[:2] for chunk_id in thought.grounding.chunk_ids]
        node_ids = [node_id for thought in evidence_thoughts[:2] for node_id in thought.grounding.node_ids]
        return ThoughtState(
            thought_id=id_factory("th"),
            role=str(payload.get("role", parent_thought.role)),
            content=str(payload.get("content", parent_thought.content)),
            grounding=Grounding(
                anchor_texts=list(hints.get("anchors", parent_thought.grounding.anchor_texts)),
                node_ids=node_ids,
                chunk_ids=chunk_ids,
                evidence_ids=evidence_ids,
                notes=list(hints.get("notes", [])),
            ),
            status="active",
            parent_ids=[parent_thought.thought_id] + list(extra_parent_ids or []),
        )
