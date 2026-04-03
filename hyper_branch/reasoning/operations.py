from __future__ import annotations

import logging
from typing import Any

from ..logging_utils import TraceStore
from ..models import EvidenceItem, Grounding, HyperedgeCandidate, TaskFrame, ThoughtState


class ThoughtOperationExecutor:
    def __init__(self, logger: logging.Logger, trace_store: TraceStore) -> None:
        self.logger = logger
        self.trace_store = trace_store

    def create_root_thought(self, thought_id: str, question: str, task_frame: TaskFrame) -> ThoughtState:
        thought = ThoughtState(
            thought_id=thought_id,
            kind="reasoning",
            content=question,
            objective=task_frame.target,
            slot_id=None,
            grounding=Grounding(anchor_texts=list(task_frame.topic_entities or task_frame.anchors), notes=["question-root"]),
            status="root",
            metadata={"iteration": 0, "branch_kind": "root"},
        )
        self._log_creation("root_created", thought, {})
        return thought

    def create_initial_anchor_thought(
        self,
        thought_id: str,
        task_frame: TaskFrame,
        candidates: list[HyperedgeCandidate],
        evidence_items: list[EvidenceItem],
        parent_ids: list[str],
    ) -> ThoughtState:
        content = "Initial anchoring over topic entities and candidate hyperedges."
        metadata = {
            "iteration": 0,
            "branch_kind": "initial",
            "selected_hyperedge_ids": [candidate.hyperedge_id for candidate in candidates],
            "frontier_hyperedges": [candidate.to_dict() for candidate in candidates],
            "confidence": max((candidate.score for candidate in candidates), default=0.0),
        }
        thought = ThoughtState(
            thought_id=thought_id,
            kind="reasoning",
            content=content,
            objective="Initial E0 and H0 anchoring",
            slot_id="anchor-0" if task_frame.anchors else None,
            grounding=self._build_grounding(task_frame, candidates, evidence_items, notes=["initial-anchor"]),
            score=metadata["confidence"],
            status="grounded",
            parent_ids=list(parent_ids),
            metadata=metadata,
        )
        self._log_creation("initial_anchor_created", thought, {"candidate_count": len(candidates)})
        return thought

    def create_branch_thought(
        self,
        thought_id: str,
        task_frame: TaskFrame,
        branch_kind: str,
        iteration: int,
        branch_result: dict[str, Any],
        candidates: list[HyperedgeCandidate],
        evidence_items: list[EvidenceItem],
        parent_ids: list[str],
    ) -> ThoughtState:
        recommended = [
            f"{candidate.hyperedge_id} ({candidate.branch_score:.3f})"
            for candidate in candidates[:3]
        ]
        content = (
            f"{branch_kind} operator recommended hyperedges: " + " | ".join(recommended)
            if recommended
            else f"{branch_kind} operator found no useful hyperedges at iteration {iteration}"
        )
        confidence = max((candidate.branch_score for candidate in candidates), default=0.0)
        thought = ThoughtState(
            thought_id=thought_id,
            kind="reasoning",
            content=content,
            objective=f"{branch_kind} hyperedge search operator",
            slot_id="target-0",
            grounding=self._build_grounding(
                task_frame,
                candidates,
                evidence_items,
                notes=[f"branch:{branch_kind}", f"iteration:{iteration}", "hyperedge-search"],
            ),
            score=max(confidence, max((candidate.score for candidate in candidates), default=0.0)),
            status="searched" if candidates else "active",
            parent_ids=list(parent_ids),
            metadata={
                "iteration": iteration,
                "branch_kind": branch_kind,
                "selected_hyperedge_ids": [candidate.hyperedge_id for candidate in candidates],
                "frontier_hyperedges": [candidate.to_dict() for candidate in candidates],
                "query_texts": list(branch_result.get("query_texts", [])),
                "recommended_count": len(candidates),
                "control_snapshot": dict(branch_result.get("control_state", {})),
                "confidence": confidence,
                "notes": str(branch_result.get("notes", "") or "").strip(),
            },
        )
        self._log_creation(
            "branch_thought_created",
            thought,
            {
                "branch_kind": branch_kind,
                "iteration": iteration,
                "candidate_count": len(candidates),
            },
        )
        return thought

    def create_merge_thought(
        self,
        thought_id: str,
        task_frame: TaskFrame,
        iteration: int,
        merge_result: dict[str, Any],
        evidence_items: list[EvidenceItem],
        parent_ids: list[str],
    ) -> ThoughtState:
        frontier_ids = [str(item).strip() for item in merge_result.get("frontier_hyperedge_ids", []) if str(item).strip()]
        content = (
            "Global frontier: " + " | ".join(frontier_ids[:3])
            if frontier_ids
            else str(merge_result.get("notes", "") or "").strip() or "Global frontier fusion"
        )
        preferred_branches = [str(item).strip() for item in merge_result.get("preferred_branches", []) if str(item).strip()]
        thought = ThoughtState(
            thought_id=thought_id,
            kind="reasoning",
            content=content or "Global frontier fused from branch operators",
            objective="Fuse branch frontier candidates",
            slot_id="target-0",
            grounding=Grounding(
                anchor_texts=list(task_frame.topic_entities or task_frame.anchors),
                chunk_ids=[item.chunk_id for item in evidence_items],
                evidence=list(evidence_items),
                notes=[f"merge-iteration:{iteration}", *preferred_branches],
            ),
            score=float(len(preferred_branches)),
            status="merged",
            parent_ids=list(parent_ids),
            metadata={
                "iteration": iteration,
                "branch_kind": "merge",
                "frontier_hyperedge_ids": merge_result.get("frontier_hyperedge_ids", []),
                "answer_hypotheses": merge_result.get("answer_hypotheses", []),
                "preferred_branches": preferred_branches,
                "branch_contributions": merge_result.get("branch_contributions", {}),
                "coverage_summary": merge_result.get("coverage_summary", {}),
                "missing_requirements": merge_result.get("missing_requirements", []),
                "notes": str(merge_result.get("notes", "") or "").strip(),
            },
        )
        self._log_creation("merge_thought_created", thought, {"iteration": iteration})
        return thought

    def create_answer_thought(
        self,
        thought_id: str,
        task_frame: TaskFrame,
        final_payload: dict[str, Any],
        evidence_items: list[EvidenceItem],
        parent_ids: list[str],
    ) -> ThoughtState:
        thought = ThoughtState(
            thought_id=thought_id,
            kind="answer",
            content=str(final_payload.get("answer", "") or "").strip(),
            objective=task_frame.target,
            slot_id="target-0",
            grounding=Grounding(
                anchor_texts=list(task_frame.topic_entities or task_frame.anchors),
                chunk_ids=[item.chunk_id for item in evidence_items],
                evidence=list(evidence_items),
                notes=[
                    str(final_payload.get("reasoning_summary", "") or "").strip(),
                    *[str(item).strip() for item in final_payload.get("remaining_gaps", []) if str(item).strip()],
                ],
            ),
            score=float(final_payload.get("confidence", 0.0) or 0.0),
            status="completed",
            parent_ids=list(parent_ids),
            metadata=dict(final_payload),
        )
        self._log_creation("answer_thought_created", thought, {"parent_count": len(parent_ids)})
        return thought

    def retire_previous_branch(self, thought: ThoughtState | None) -> None:
        if thought is None:
            return
        if thought.status in {"verified", "active", "grounded", "searched"}:
            thought.status = "expanded"

    def _build_grounding(
        self,
        task_frame: TaskFrame,
        candidates: list[HyperedgeCandidate],
        evidence_items: list[EvidenceItem],
        notes: list[str] | None = None,
    ) -> Grounding:
        node_ids: list[str] = []
        chunk_ids: list[str] = []
        for candidate in candidates:
            if candidate.hyperedge_id not in node_ids:
                node_ids.append(candidate.hyperedge_id)
            for entity_id in candidate.entity_ids:
                if entity_id not in node_ids:
                    node_ids.append(entity_id)
            for chunk_id in candidate.chunk_ids:
                if chunk_id not in chunk_ids:
                    chunk_ids.append(chunk_id)
        return Grounding(
            anchor_texts=list(task_frame.topic_entities or task_frame.anchors),
            node_ids=node_ids,
            chunk_ids=chunk_ids,
            evidence=list(evidence_items),
            notes=list(notes or []),
        )

    def _log_creation(self, event: str, thought: ThoughtState, payload: dict[str, Any]) -> None:
        self.trace_store.log_event(
            event,
            {
                "thought_id": thought.thought_id,
                "kind": thought.kind,
                "status": thought.status,
                **payload,
            },
        )
        self.logger.info("Created %s thought %s (%s)", thought.kind, thought.thought_id, thought.status)
