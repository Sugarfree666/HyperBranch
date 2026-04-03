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
class HyperedgeCandidate:
    hyperedge_id: str
    hyperedge_text: str
    score: float
    branch_kind: str
    branch_score: float = 0.0
    fused_score: float = 0.0
    coverage_gain: float = 0.0
    constraint_gain: float = 0.0
    relation_gain: float = 0.0
    connector_gain: float = 0.0
    novelty_gain: float = 0.0
    focus_gain: float = 0.0
    penalty: float = 0.0
    entity_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    matched_topic_entities: list[str] = field(default_factory=list)
    support_entities: list[str] = field(default_factory=list)
    supporting_chunks: list[str] = field(default_factory=list)
    score_breakdown: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["display_hyperedge"] = normalize_label(self.hyperedge_id)
        payload["entity_labels"] = [normalize_label(entity_id) for entity_id in self.entity_ids]
        payload["matched_topic_labels"] = [normalize_label(entity_id) for entity_id in self.matched_topic_entities]
        payload["support_entity_labels"] = [normalize_label(entity_id) for entity_id in self.support_entities]
        return payload


@dataclass(slots=True)
class EvidenceSubgraph:
    hyperedge_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    branch_support: dict[str, list[str]] = field(default_factory=dict)
    branch_answers: dict[str, dict[str, Any]] = field(default_factory=dict)
    frontier_history: list[dict[str, Any]] = field(default_factory=list)
    control_history: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def record_branch_result(
        self,
        branch_kind: str,
        candidates: list[HyperedgeCandidate],
        branch_result: dict[str, Any] | None = None,
    ) -> None:
        support_ids = self.branch_support.setdefault(branch_kind, [])
        for candidate in candidates:
            if candidate.hyperedge_id not in support_ids:
                support_ids.append(candidate.hyperedge_id)
        if branch_result is not None:
            self.branch_answers[branch_kind] = dict(branch_result)

    def add_frontier(
        self,
        iteration: int,
        candidates: list[HyperedgeCandidate],
        evidence_items: list[EvidenceItem],
        control_state: dict[str, Any],
    ) -> None:
        seen_hyperedges = set(self.hyperedge_ids)
        seen_entities = set(self.entity_ids)
        seen_chunks = set(self.chunk_ids)
        seen_evidence_ids = {item.evidence_id for item in self.evidence}

        for candidate in candidates:
            if candidate.hyperedge_id not in seen_hyperedges:
                self.hyperedge_ids.append(candidate.hyperedge_id)
                seen_hyperedges.add(candidate.hyperedge_id)
            for entity_id in candidate.entity_ids:
                if entity_id not in seen_entities:
                    self.entity_ids.append(entity_id)
                    seen_entities.add(entity_id)
            for chunk_id in candidate.chunk_ids:
                if chunk_id not in seen_chunks:
                    self.chunk_ids.append(chunk_id)
                    seen_chunks.add(chunk_id)

        for item in evidence_items:
            if item.evidence_id in seen_evidence_ids:
                continue
            self.evidence.append(item)
            seen_evidence_ids.add(item.evidence_id)
            if item.chunk_id and item.chunk_id not in seen_chunks:
                self.chunk_ids.append(item.chunk_id)
                seen_chunks.add(item.chunk_id)
            for node_id in item.source_node_ids:
                if node_id not in seen_entities and node_id not in seen_hyperedges:
                    self.entity_ids.append(node_id)
                    seen_entities.add(node_id)

        self.frontier_history.append(
            {
                "iteration": iteration,
                "hyperedge_ids": [candidate.hyperedge_id for candidate in candidates],
                "frontier": [candidate.to_dict() for candidate in candidates],
            }
        )
        self.control_history.append(dict(control_state))

    def to_text(self, limit: int = 5) -> str:
        parts: list[str] = []
        if self.hyperedge_ids:
            parts.append(
                "hyperedges: " + " | ".join(normalize_label(item) for item in self.hyperedge_ids[:limit])
            )
        if self.entity_ids:
            parts.append("entities: " + ", ".join(normalize_label(item) for item in self.entity_ids[:limit]))
        if self.evidence:
            parts.append("evidence: " + " | ".join(item.content[:220] for item in self.evidence[:limit]))
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hyperedge_ids": list(self.hyperedge_ids),
            "entity_ids": list(self.entity_ids),
            "chunk_ids": list(self.chunk_ids),
            "branch_support": dict(self.branch_support),
            "branch_answers": dict(self.branch_answers),
            "frontier_history": list(self.frontier_history),
            "control_history": list(self.control_history),
            "notes": list(self.notes),
            "evidence": [item.to_dict() for item in self.evidence],
            "summary_text": self.to_text(),
        }


@dataclass(slots=True)
class RetrievalControlState:
    iteration: int = 0
    branch_weights: dict[str, float] = field(
        default_factory=lambda: {"constraint": 1.0, "relation": 1.0, "anchor": 1.0}
    )
    missing_requirements: list[str] = field(default_factory=list)
    next_focus: list[str] = field(default_factory=list)
    branch_queries: dict[str, list[str]] = field(default_factory=dict)
    candidate_filters: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def current_focus(self) -> list[str]:
        focus: list[str] = []
        for item in [*self.next_focus, *self.missing_requirements]:
            text = str(item).strip()
            if text and text not in focus:
                focus.append(text)
        return focus

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "branch_weights": dict(self.branch_weights),
            "missing_requirements": list(self.missing_requirements),
            "next_focus": list(self.next_focus),
            "current_focus": self.current_focus(),
            "branch_queries": {key: list(value) for key, value in self.branch_queries.items()},
            "candidate_filters": dict(self.candidate_filters),
            "notes": list(self.notes),
        }


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
    topic_entities: list[str] = field(default_factory=list)
    answer_type_hint: str = ""
    relation_intent: str = ""
    hard_constraints: list[str] = field(default_factory=list)
    relation_skeleton: str = ""
    initial_entity_ids: list[str] = field(default_factory=list)
    initial_hyperedge_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    checklist: dict[str, list[TaskChecklistItem]] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, question: str, payload: dict[str, Any]) -> "TaskFrame":
        topic_entities = [
            str(item).strip()
            for item in payload.get("topic_entities", payload.get("anchors", []))
            if str(item).strip()
        ]
        answer_type_hint = str(payload.get("answer_type_hint", payload.get("target", ""))).strip()
        relation_intent = str(payload.get("relation_intent", "")).strip()
        hard_constraints = [
            str(item).strip()
            for item in payload.get("hard_constraints", payload.get("constraints", []))
            if str(item).strip()
        ]
        relation_skeleton = str(payload.get("relation_skeleton", "")).strip()

        anchors = [str(item).strip() for item in payload.get("anchors", topic_entities) if str(item).strip()]
        target = str(payload.get("target", answer_type_hint)).strip()
        constraints = [str(item).strip() for item in payload.get("constraints", hard_constraints) if str(item).strip()]
        bridges = [str(item).strip() for item in payload.get("bridges", []) if str(item).strip()]
        if relation_intent and relation_intent not in bridges:
            bridges = [relation_intent, *bridges]
        if not target:
            target = question
        if not answer_type_hint:
            answer_type_hint = target
        if not anchors:
            anchors = topic_entities
        if not topic_entities:
            topic_entities = anchors
        if not constraints:
            constraints = hard_constraints
        if not hard_constraints:
            hard_constraints = constraints

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
            topic_entities=topic_entities,
            answer_type_hint=answer_type_hint,
            relation_intent=relation_intent,
            hard_constraints=hard_constraints,
            relation_skeleton=relation_skeleton,
            initial_entity_ids=[
                str(item).strip() for item in payload.get("initial_entity_ids", []) if str(item).strip()
            ],
            initial_hyperedge_ids=[
                str(item).strip() for item in payload.get("initial_hyperedge_ids", []) if str(item).strip()
            ],
            metadata=dict(payload.get("metadata", {})),
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
            "topic_entities": list(self.topic_entities),
            "answer_type_hint": self.answer_type_hint,
            "relation_intent": self.relation_intent,
            "hard_constraints": list(self.hard_constraints),
            "relation_skeleton": self.relation_skeleton,
            "initial_entity_ids": list(self.initial_entity_ids),
            "initial_hyperedge_ids": list(self.initial_hyperedge_ids),
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
            "topic_entities": self.topic_entities,
            "answer_type_hint": self.answer_type_hint,
            "relation_intent": self.relation_intent,
            "hard_constraints": self.hard_constraints,
            "relation_skeleton": self.relation_skeleton,
            "initial_entity_ids": self.initial_entity_ids,
            "initial_hyperedge_ids": self.initial_hyperedge_ids,
            "metadata": self.metadata,
            "checklist": self.progress_snapshot(),
        }


@dataclass(slots=True)
class Grounding:
    anchor_texts: list[str] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def update_with_evidence(self, evidence_items: list[EvidenceItem]) -> None:
        existing = {item.evidence_id for item in self.evidence}
        for item in evidence_items:
            if item.evidence_id in existing:
                continue
            self.evidence.append(item)
            existing.add(item.evidence_id)
            if item.chunk_id and item.chunk_id not in self.chunk_ids:
                self.chunk_ids.append(item.chunk_id)
            for node_id in item.source_node_ids:
                if node_id not in self.node_ids:
                    self.node_ids.append(node_id)

    def to_text(self) -> str:
        parts: list[str] = []
        if self.anchor_texts:
            parts.append("anchors: " + ", ".join(self.anchor_texts))
        if self.node_ids:
            parts.append("nodes: " + ", ".join(normalize_label(node_id) for node_id in self.node_ids[:8]))
        if self.chunk_ids:
            parts.append("chunks: " + ", ".join(self.chunk_ids[:8]))
        if self.evidence:
            evidence_bits = []
            for item in self.evidence[:3]:
                evidence_bits.append(item.content[:220])
            parts.append("evidence: " + " | ".join(evidence_bits))
        if self.notes:
            parts.append("notes: " + " | ".join(self.notes[:4]))
        return "; ".join(parts)


@dataclass(slots=True)
class ThoughtState:
    thought_id: str
    kind: str
    content: str
    objective: str
    slot_id: str | None
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
            "kind": self.kind,
            "status": self.status,
            "score": round(self.score, 4),
            "content": self.content,
            "objective": self.objective,
            "slot_id": self.slot_id,
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
            if thought.status == "active" and thought.kind == "reasoning"
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
