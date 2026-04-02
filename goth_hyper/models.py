from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .utils import normalize_label


@dataclass(slots=True)
class VectorMatch:
    item_id: str
    label: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["display_label"] = normalize_label(self.label)
        return payload


@dataclass(slots=True)
class GraphNode:
    node_id: str
    role: str
    weight: float = 0.0
    source_ids: list[str] = field(default_factory=list)
    entity_type: str | None = None
    description: str | None = None

    @property
    def display_label(self) -> str:
        return normalize_label(self.node_id)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["display_label"] = self.display_label
        return payload


@dataclass(slots=True)
class GraphEdge:
    edge_id: str
    source: str
    target: str
    role: str
    weight: float = 0.0
    source_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_display"] = normalize_label(self.source)
        payload["target_display"] = normalize_label(self.target)
        return payload


@dataclass(slots=True)
class EvidenceItem:
    evidence_id: str
    chunk_id: str
    content: str
    score: float
    source_node_ids: list[str] = field(default_factory=list)
    source_edge_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_nodes"] = [normalize_label(node_id) for node_id in self.source_node_ids]
        return payload


@dataclass(slots=True)
class TaskChecklistItem:
    slot_id: str
    kind: str
    text: str
    status: str = "open"
    evidence_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def satisfied(self) -> bool:
        if self.kind == "anchors":
            return self.status in {"retrieved", "supported", "verified"}
        return self.status in {"supported", "verified"}


@dataclass(slots=True)
class TaskFrame:
    question: str
    anchors: list[str]
    target: str
    constraints: list[str]
    bridges: list[str]
    hypothesis_template: str
    checklist: dict[str, list[TaskChecklistItem]] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, question: str, payload: dict[str, Any]) -> "TaskFrame":
        anchors = [str(item).strip() for item in payload.get("anchors", []) if str(item).strip()]
        target = str(payload.get("target", "")).strip()
        constraints = [str(item).strip() for item in payload.get("constraints", []) if str(item).strip()]
        bridges = [str(item).strip() for item in payload.get("bridges", []) if str(item).strip()]
        hypothesis_template = str(payload.get("hypothesis_template", "")).strip()
        if not target:
            target = question
        if not hypothesis_template:
            hypothesis_template = f"Answer the question '{question}' by connecting the most relevant evidence."

        checklist = {
            "anchors": [
                TaskChecklistItem(slot_id=f"anchor-{index}", kind="anchors", text=text)
                for index, text in enumerate(anchors)
            ],
            "target": [TaskChecklistItem(slot_id="target-0", kind="target", text=target)],
            "constraints": [
                TaskChecklistItem(slot_id=f"constraint-{index}", kind="constraints", text=text)
                for index, text in enumerate(constraints)
            ],
            "bridges": [
                TaskChecklistItem(slot_id=f"bridge-{index}", kind="bridges", text=text)
                for index, text in enumerate(bridges)
            ],
        }
        return cls(
            question=question,
            anchors=anchors,
            target=target,
            constraints=constraints,
            bridges=bridges,
            hypothesis_template=hypothesis_template,
            checklist=checklist,
        )

    def iter_slots(self) -> list[TaskChecklistItem]:
        slots: list[TaskChecklistItem] = []
        for key in ("anchors", "target", "constraints", "bridges"):
            slots.extend(self.checklist.get(key, []))
        return slots

    def get_open_slots(self) -> list[TaskChecklistItem]:
        return [slot for slot in self.iter_slots() if not slot.satisfied()]

    def find_slot(self, slot_id: str) -> TaskChecklistItem | None:
        for slot in self.iter_slots():
            if slot.slot_id == slot_id:
                return slot
        return None

    def mark_slot(self, slot_id: str, evidence_id: str | None = None, status: str = "supported", note: str | None = None) -> bool:
        slot = self.find_slot(slot_id)
        if slot is None:
            return False
        slot.status = status
        if evidence_id and evidence_id not in slot.evidence_ids:
            slot.evidence_ids.append(evidence_id)
        if note:
            slot.notes.append(note)
        return True

    def progress_snapshot(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "anchors": [asdict(item) for item in self.checklist.get("anchors", [])],
            "target": [asdict(item) for item in self.checklist.get("target", [])],
            "constraints": [asdict(item) for item in self.checklist.get("constraints", [])],
            "bridges": [asdict(item) for item in self.checklist.get("bridges", [])],
            "all_required_satisfied": self.is_satisfied(),
        }

    def is_satisfied(self) -> bool:
        anchor_ok = all(slot.satisfied() for slot in self.checklist.get("anchors", []))
        required_ok = all(
            slot.satisfied()
            for key in ("target", "constraints", "bridges")
            for slot in self.checklist.get(key, [])
        )
        return anchor_ok and required_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "anchors": self.anchors,
            "target": self.target,
            "constraints": self.constraints,
            "bridges": self.bridges,
            "hypothesis_template": self.hypothesis_template,
            "checklist": self.progress_snapshot(),
        }


@dataclass(slots=True)
class Grounding:
    anchor_texts: list[str] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        parts: list[str] = []
        if self.anchor_texts:
            parts.append("anchors: " + ", ".join(self.anchor_texts))
        if self.node_ids:
            parts.append("nodes: " + ", ".join(normalize_label(node_id) for node_id in self.node_ids[:8]))
        if self.chunk_ids:
            parts.append("chunks: " + ", ".join(self.chunk_ids[:8]))
        if self.notes:
            parts.append("notes: " + " | ".join(self.notes[:4]))
        return "; ".join(parts)


@dataclass(slots=True)
class ThoughtState:
    thought_id: str
    role: str
    content: str
    grounding: Grounding = field(default_factory=Grounding)
    score: float = 0.0
    status: str = "active"
    parent_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["grounding_text"] = self.grounding.to_text()
        return payload

    def brief(self) -> dict[str, Any]:
        return {
            "thought_id": self.thought_id,
            "role": self.role,
            "status": self.status,
            "score": round(self.score, 4),
            "content": self.content,
            "grounding_text": self.grounding.to_text(),
        }


@dataclass(slots=True)
class ThoughtGraph:
    question: str
    root_id: str
    frontier_ids: list[str] = field(default_factory=list)
    status: str = "running"
    thoughts: dict[str, ThoughtState] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    final_answer: dict[str, Any] | None = None
    termination_reason: str | None = None

    def add_thought(self, thought: ThoughtState) -> None:
        self.thoughts[thought.thought_id] = thought
        self.recompute_frontier()

    def get(self, thought_id: str) -> ThoughtState:
        return self.thoughts[thought_id]

    def set_status(self, thought_id: str, status: str) -> None:
        self.thoughts[thought_id].status = status
        self.recompute_frontier()

    def recompute_frontier(self) -> None:
        self.frontier_ids = [
            thought_id
            for thought_id, thought in self.thoughts.items()
            if thought.status == "active" and thought.role != "answer"
        ]

    def active_frontier(self) -> list[ThoughtState]:
        return [self.thoughts[thought_id] for thought_id in self.frontier_ids]

    def append_history(self, event_type: str, payload: dict[str, Any]) -> None:
        self.history.append({"event": event_type, "payload": payload})

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "root_id": self.root_id,
            "frontier_ids": list(self.frontier_ids),
            "status": self.status,
            "termination_reason": self.termination_reason,
            "final_answer": self.final_answer,
            "thoughts": {thought_id: thought.to_dict() for thought_id, thought in self.thoughts.items()},
            "history": list(self.history),
        }
