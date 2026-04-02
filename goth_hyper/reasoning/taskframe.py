from __future__ import annotations

import logging
from typing import Any

from ..data.loaders import DatasetBundle
from ..logging_utils import TraceStore
from ..models import TaskFrame, ThoughtState, VectorMatch
from ..utils import cosine_similarity, normalize_label


class TaskFrameBuilder:
    def __init__(self, llm_service: Any, dataset: DatasetBundle, logger: logging.Logger, trace_store: TraceStore) -> None:
        self.llm_service = llm_service
        self.dataset = dataset
        self.logger = logger
        self.trace_store = trace_store

    def build(self, question: str) -> TaskFrame:
        payload = self.llm_service.build_task_frame(question, self.dataset.summary)
        task_frame = TaskFrame.from_payload(question, payload)
        self.logger.info(
            "TaskFrame created with %s anchors, %s constraints, %s bridges",
            len(task_frame.anchors),
            len(task_frame.constraints),
            len(task_frame.bridges),
        )
        self.trace_store.log_event("task_frame_created", task_frame.progress_snapshot())
        return task_frame


class TaskFrameRegistry:
    def __init__(self, embedder: Any, threshold: float, logger: logging.Logger, trace_store: TraceStore) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.logger = logger
        self.trace_store = trace_store

    def register_anchor_matches(self, task_frame: TaskFrame, entity_matches: list[VectorMatch]) -> list[str]:
        updated_slots: list[str] = []
        normalized_matches = [(match, normalize_label(match.label).lower()) for match in entity_matches]
        for slot in task_frame.checklist.get("anchors", []):
            anchor_norm = normalize_label(slot.text).lower()
            for match, match_norm in normalized_matches:
                if anchor_norm and (anchor_norm in match_norm or match_norm in anchor_norm):
                    task_frame.mark_slot(
                        slot.slot_id,
                        evidence_id=match.item_id,
                        status="retrieved",
                        note=f"Matched entity seed: {normalize_label(match.label)} ({match.score:.3f})",
                    )
                    updated_slots.append(slot.slot_id)
                    break
        if updated_slots:
            self.trace_store.log_event("taskframe_anchor_registered", {"slot_ids": updated_slots})
        return updated_slots

    def register_evidence(self, task_frame: TaskFrame, evidence_thought: ThoughtState) -> list[str]:
        open_slots = task_frame.get_open_slots()
        if not open_slots:
            return []

        texts = [evidence_thought.content] + [slot.text for slot in open_slots]
        embeddings = self.embedder.embed_texts(texts, stage="taskframe_registry")
        evidence_vector = embeddings[0]
        slot_vectors = embeddings[1:]

        updated_slots: list[str] = []
        status = "verified" if evidence_thought.status == "verified" else "supported"
        for slot, slot_vector in zip(open_slots, slot_vectors, strict=True):
            similarity = max(cosine_similarity(evidence_vector, slot_vector), 0.0)
            if similarity >= self.threshold:
                task_frame.mark_slot(
                    slot.slot_id,
                    evidence_id=evidence_thought.thought_id,
                    status=status,
                    note=f"Registered via evidence similarity {similarity:.3f}",
                )
                updated_slots.append(slot.slot_id)

        if updated_slots:
            self.logger.info(
                "Registered evidence thought %s to TaskFrame slots %s",
                evidence_thought.thought_id,
                ", ".join(updated_slots),
            )
            self.trace_store.log_event(
                "taskframe_evidence_registered",
                {"evidence_thought_id": evidence_thought.thought_id, "slot_ids": updated_slots},
            )
        return updated_slots
