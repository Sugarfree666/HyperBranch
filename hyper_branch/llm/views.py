from __future__ import annotations

from typing import Any

from ..models import EvidenceItem, EvidenceSubgraph, RetrievalControlState, TaskChecklistItem, TaskFrame, ThoughtGraph
from ..utils import normalize_label, short_text


FORBIDDEN_LLM_FIELDS = {
    "score",
    "branch_score",
    "fused_score",
    "coverage_gain",
    "constraint_gain",
    "relation_gain",
    "connector_gain",
    "novelty_gain",
    "focus_gain",
    "penalty",
    "frontier_history",
    "control_history",
    "branch_support",
    "branch_answers",
}


def build_llm_evidence_view(
    question: str,
    task_frame: TaskFrame,
    evidence_subgraph: EvidenceSubgraph,
    merge_result: dict[str, Any],
    control_state: RetrievalControlState,
    *,
    max_frontier_items: int = 3,
) -> dict[str, Any]:
    frontier_candidates = _frontier_candidates(merge_result, evidence_subgraph)[: max(1, min(max_frontier_items, 3))]
    branch_membership = _branch_membership(merge_result.get("branch_contributions", {}))
    missing_requirements = _dedupe_texts(
        [
            *(_clean_text_list(merge_result.get("missing_requirements", []), limit=4)),
            *(_clean_text_list(control_state.missing_requirements, limit=4)),
        ]
    )[:4]
    next_focus = _dedupe_texts(
        [
            *(_clean_text_list(merge_result.get("next_focus", []), limit=4)),
            *(_clean_text_list(control_state.next_focus, limit=4)),
        ]
    )[:4]
    current_focus = _dedupe_texts([*next_focus, *missing_requirements, *control_state.current_focus()])[:6]
    answer_hypotheses = _clean_text_list(merge_result.get("answer_hypotheses", []), limit=3)

    frontier_view: list[dict[str, Any]] = []
    for candidate in frontier_candidates:
        hyperedge_id = str(candidate.get("hyperedge_id", "")).strip()
        hyperedge_text = short_text(
            normalize_label(str(candidate.get("hyperedge_text") or hyperedge_id)),
            240,
        )
        core_evidence = _select_core_evidence(candidate, evidence_subgraph.evidence)
        frontier_view.append(
            {
                "hyperedge": hyperedge_text,
                "supporting_branches": branch_membership.get(hyperedge_id, []),
                "supporting_channels": _candidate_labels(candidate, "supporting_channel_labels", "channel_label", limit=4),
                "matched_topics": _candidate_labels(candidate, "matched_topic_labels", "matched_topic_entities", limit=4),
                "core_entities": _candidate_labels(
                    candidate,
                    "support_entity_labels",
                    "entity_labels",
                    "support_entities",
                    "entity_ids",
                    limit=4,
                ),
                "core_evidence": core_evidence,
            }
        )

    coverage_summary = {
        "topics": _slot_summary(task_frame.checklist.get("anchors", [])),
        "constraints": _slot_summary(task_frame.checklist.get("constraints", [])),
        "relations": {
            "intent": short_text(task_frame.relation_intent or task_frame.relation_skeleton or task_frame.target, 180),
            **_slot_summary(task_frame.checklist.get("bridges", [])),
        },
        "target": _target_summary(task_frame.checklist.get("target", [])),
        "answer_hypotheses": answer_hypotheses,
    }
    control_summary = {
        "iteration": int(control_state.iteration),
        "branch_weights": {
            branch_kind: round(float(control_state.branch_weights.get(branch_kind, 0.0) or 0.0), 3)
            for branch_kind in ("constraint", "relation", "anchor")
        },
        "preferred_branches": _clean_text_list(merge_result.get("preferred_branches", []), limit=3),
    }

    return {
        "question": question,
        "current_focus": current_focus,
        "missing_requirements": missing_requirements,
        "next_focus": next_focus,
        "frontier_hyperedges": frontier_view,
        "coverage_summary": coverage_summary,
        "control_summary": control_summary,
        "evidence_summary": _build_evidence_summary(frontier_view, coverage_summary, missing_requirements, answer_hypotheses),
    }


def build_llm_thought_graph_summary(thought_graph: ThoughtGraph, *, limit: int = 6) -> dict[str, Any]:
    recent_thoughts = list(thought_graph.thoughts.values())[-max(1, limit) :]
    return {
        "status": thought_graph.status,
        "termination_reason": thought_graph.termination_reason,
        "frontier_ids": list(thought_graph.frontier_ids[:4]),
        "recent_thoughts": [
            {
                "thought_id": thought.thought_id,
                "kind": thought.kind,
                "status": thought.status,
                "objective": short_text(thought.objective, 140),
                "slot_id": thought.slot_id,
                "grounding": short_text(thought.grounding.to_text(), 220),
            }
            for thought in recent_thoughts
        ],
    }


def _frontier_candidates(merge_result: dict[str, Any], evidence_subgraph: EvidenceSubgraph) -> list[dict[str, Any]]:
    frontier = merge_result.get("frontier", [])
    if isinstance(frontier, list) and frontier:
        return [candidate for candidate in frontier if isinstance(candidate, dict)]
    if evidence_subgraph.frontier_history:
        latest = evidence_subgraph.frontier_history[-1].get("frontier", [])
        if isinstance(latest, list):
            return [candidate for candidate in latest if isinstance(candidate, dict)]
    return []


def _branch_membership(branch_contributions: Any) -> dict[str, list[str]]:
    memberships: dict[str, list[str]] = {}
    if not isinstance(branch_contributions, dict):
        return memberships
    for branch_kind, supported_ids in branch_contributions.items():
        if not isinstance(supported_ids, list):
            continue
        for hyperedge_id in supported_ids:
            cleaned = str(hyperedge_id).strip()
            if not cleaned:
                continue
            memberships.setdefault(cleaned, [])
            branch_text = str(branch_kind).strip()
            if branch_text and branch_text not in memberships[cleaned]:
                memberships[cleaned].append(branch_text)
    return memberships


def _candidate_labels(candidate: dict[str, Any], *keys: str, limit: int) -> list[str]:
    labels: list[str] = []
    for key in keys:
        value = candidate.get(key, [])
        if not isinstance(value, list):
            continue
        for item in value:
            cleaned = normalize_label(str(item).strip())
            if cleaned and cleaned not in labels:
                labels.append(cleaned)
            if len(labels) >= limit:
                return labels
    return labels


def _select_core_evidence(candidate: dict[str, Any], evidence_items: list[EvidenceItem]) -> str:
    hyperedge_id = str(candidate.get("hyperedge_id", "")).strip()
    chunk_ids = {
        str(chunk_id).strip()
        for chunk_id in candidate.get("chunk_ids", [])
        if str(chunk_id).strip()
    }
    for item in evidence_items:
        if hyperedge_id and hyperedge_id in item.source_node_ids:
            return _trim_evidence_content(item.content)
        if item.chunk_id and item.chunk_id in chunk_ids:
            return _trim_evidence_content(item.content)
    supporting_chunks = candidate.get("supporting_chunks", [])
    if isinstance(supporting_chunks, list):
        for chunk_text in supporting_chunks:
            cleaned = short_text(str(chunk_text).strip(), 420)
            if cleaned:
                return cleaned
    return ""


def _trim_evidence_content(content: str) -> str:
    sections = [part.strip() for part in str(content).split("\n\n") if part.strip()]
    if not sections:
        return ""
    body = sections[-1] if len(sections) > 1 else sections[0]
    return short_text(body, 420)


def _slot_summary(slots: list[TaskChecklistItem]) -> dict[str, list[str]]:
    covered: list[str] = []
    missing: list[str] = []
    for slot in slots:
        text = short_text(str(slot.text).strip(), 140)
        if not text:
            continue
        target = covered if slot.satisfied() else missing
        if text not in target:
            target.append(text)
    return {
        "covered": covered[:4],
        "missing": missing[:4],
    }


def _target_summary(slots: list[TaskChecklistItem]) -> dict[str, Any]:
    if not slots:
        return {"text": "", "status": "open"}
    slot = slots[0]
    return {
        "text": short_text(str(slot.text).strip(), 140),
        "status": slot.status,
    }


def _build_evidence_summary(
    frontier_view: list[dict[str, Any]],
    coverage_summary: dict[str, Any],
    missing_requirements: list[str],
    answer_hypotheses: list[str],
) -> str:
    summary_parts: list[str] = []
    if answer_hypotheses:
        summary_parts.append("Current answer hypotheses: " + ", ".join(answer_hypotheses))
    if frontier_view:
        summary_parts.append(
            "Top frontier evidence: "
            + " | ".join(short_text(str(item.get("hyperedge", "")), 120) for item in frontier_view[:2])
        )
    open_constraints = coverage_summary.get("constraints", {}).get("missing", [])
    if open_constraints:
        summary_parts.append("Still-open constraints: " + ", ".join(open_constraints[:2]))
    elif missing_requirements:
        summary_parts.append("Remaining gaps: " + ", ".join(missing_requirements[:2]))
    if not summary_parts:
        return "Evidence view is sparse and does not yet isolate a grounded answer."
    return " ".join(summary_parts)


def _clean_text_list(values: Any, *, limit: int) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for value in values:
        text = short_text(normalize_label(str(value).strip()), 140)
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _dedupe_texts(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        cleaned = short_text(normalize_label(str(value).strip()), 140)
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return deduped
