from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from ..models import HyperedgeCandidate, TaskFrame, ThoughtGraph, ThoughtState
from ..utils import content_tokens, lexical_overlap_score, normalize_label, short_text
from .client import OpenAICompatibleClient
from .prompts import PromptManager


class ReasoningService(ABC):
    @abstractmethod
    def build_task_frame(self, question: str, dataset_summary: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def select_branch_candidates(
        self,
        question: str,
        task_frame: TaskFrame,
        branch_kind: str,
        candidate_hyperedges: list[HyperedgeCandidate],
        evidence_subgraph: dict[str, Any],
        top_k: int,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def reconcile_branches(
        self,
        question: str,
        task_frame: TaskFrame,
        branch_results: list[dict[str, Any]],
        evidence_subgraph: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def judge_sufficiency(
        self,
        question: str,
        task_frame: TaskFrame,
        branch_results: list[dict[str, Any]],
        merge_result: dict[str, Any],
        evidence_subgraph: dict[str, Any],
        iteration: int,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def synthesize_answer(
        self,
        question: str,
        task_frame: TaskFrame,
        thought_graph: ThoughtGraph,
        evidence_subgraph: dict[str, Any],
        merge_result: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError


class OpenAIReasoningService(ReasoningService):
    def __init__(self, client: OpenAICompatibleClient, prompts: PromptManager) -> None:
        self.client = client
        self.prompts = prompts

    def build_task_frame(self, question: str, dataset_summary: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "question": question,
            "dataset_summary": dataset_summary,
        }
        response = self.client.chat_json("task_frame", self.prompts.get("task_frame"), payload)
        response.setdefault("topic_entities", response.get("anchors", []))
        response.setdefault("answer_type_hint", response.get("target", question))
        response.setdefault("relation_intent", "")
        response.setdefault("hard_constraints", response.get("constraints", []))
        response.setdefault("relation_skeleton", "")
        return response

    def select_branch_candidates(
        self,
        question: str,
        task_frame: TaskFrame,
        branch_kind: str,
        candidate_hyperedges: list[HyperedgeCandidate],
        evidence_subgraph: dict[str, Any],
        top_k: int,
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "branch_kind": branch_kind,
            "top_k": top_k,
            "task_frame_progress": task_frame.progress_snapshot(),
            "evidence_subgraph": evidence_subgraph,
            "candidate_hyperedges": [candidate.to_dict() for candidate in candidate_hyperedges],
        }
        response = self.client.chat_json("branch_selector", self.prompts.get("branch_selector"), payload)
        valid_ids = {candidate.hyperedge_id for candidate in candidate_hyperedges}
        selected_ids = response.get("selected_hyperedge_ids", [])
        if not isinstance(selected_ids, list):
            selected_ids = []
        response["selected_hyperedge_ids"] = [item for item in selected_ids if item in valid_ids][:top_k]
        response.setdefault("candidate_answer", "")
        response.setdefault("supporting_facts", [])
        response.setdefault("missing_requirements", [])
        response.setdefault("confidence", 0.0)
        response.setdefault("notes", "")
        return response

    def reconcile_branches(
        self,
        question: str,
        task_frame: TaskFrame,
        branch_results: list[dict[str, Any]],
        evidence_subgraph: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "task_frame_progress": task_frame.progress_snapshot(),
            "branch_results": branch_results,
            "evidence_subgraph": evidence_subgraph,
        }
        response = self.client.chat_json("branch_reconcile", self.prompts.get("branch_reconcile"), payload)
        response.setdefault("consensus_answer", "")
        response.setdefault("agreement_groups", [])
        response.setdefault("conflicts", [])
        response.setdefault("preferred_branches", [])
        response.setdefault("missing_requirements", [])
        response.setdefault("notes", "")
        return response

    def judge_sufficiency(
        self,
        question: str,
        task_frame: TaskFrame,
        branch_results: list[dict[str, Any]],
        merge_result: dict[str, Any],
        evidence_subgraph: dict[str, Any],
        iteration: int,
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "iteration": iteration,
            "task_frame_progress": task_frame.progress_snapshot(),
            "branch_results": branch_results,
            "merge_result": merge_result,
            "evidence_subgraph": evidence_subgraph,
        }
        response = self.client.chat_json("evidence_judge", self.prompts.get("evidence_judge"), payload)
        response.setdefault("enough", False)
        response.setdefault("confidence", 0.0)
        response.setdefault("reason", "")
        response.setdefault("missing_requirements", [])
        response.setdefault("next_focus", [])
        return response

    def synthesize_answer(
        self,
        question: str,
        task_frame: TaskFrame,
        thought_graph: ThoughtGraph,
        evidence_subgraph: dict[str, Any],
        merge_result: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "task_frame_progress": task_frame.progress_snapshot(),
            "merge_result": merge_result,
            "evidence_subgraph": evidence_subgraph,
            "thought_graph_summary": {
                "status": thought_graph.status,
                "termination_reason": thought_graph.termination_reason,
                "thoughts": [thought.brief() for thought in thought_graph.thoughts.values()],
            },
        }
        response = self.client.chat_json("final_answer", self.prompts.get("final_answer"), payload)
        response.setdefault("answer", "")
        response.setdefault("reasoning_summary", "")
        response.setdefault("confidence", 0.0)
        response.setdefault("remaining_gaps", [])
        return response


class MockReasoningService(ReasoningService):
    def build_task_frame(self, question: str, dataset_summary: dict[str, Any]) -> dict[str, Any]:
        del dataset_summary
        phrases = _extract_topic_phrases(question)
        relation_intent = _infer_relation_intent(question)
        answer_type_hint = _infer_answer_type(question)
        hard_constraints = _infer_constraints(question)
        relation_skeleton = relation_intent if relation_intent else question
        return {
            "topic_entities": phrases[:4],
            "answer_type_hint": answer_type_hint,
            "relation_intent": relation_intent,
            "hard_constraints": hard_constraints,
            "relation_skeleton": relation_skeleton,
            "anchors": phrases[:4],
            "target": answer_type_hint,
            "constraints": hard_constraints,
            "bridges": [relation_intent] if relation_intent else [],
        }

    def select_branch_candidates(
        self,
        question: str,
        task_frame: TaskFrame,
        branch_kind: str,
        candidate_hyperedges: list[HyperedgeCandidate],
        evidence_subgraph: dict[str, Any],
        top_k: int,
    ) -> dict[str, Any]:
        del question, evidence_subgraph
        selected = candidate_hyperedges[:top_k]
        candidate_answer = _infer_candidate_answer(task_frame, selected)
        supporting_facts = [normalize_label(candidate.hyperedge_id) for candidate in selected[:2]]
        missing_requirements = [] if candidate_answer else [f"{branch_kind} branch still lacks a concrete answer hypothesis."]
        confidence = min(0.95, max((candidate.score for candidate in selected), default=0.0))
        return {
            "selected_hyperedge_ids": [candidate.hyperedge_id for candidate in selected],
            "candidate_answer": candidate_answer,
            "supporting_facts": supporting_facts,
            "missing_requirements": missing_requirements,
            "confidence": confidence,
            "notes": f"Mock {branch_kind} branch used the highest-ranked hyperedges.",
        }

    def reconcile_branches(
        self,
        question: str,
        task_frame: TaskFrame,
        branch_results: list[dict[str, Any]],
        evidence_subgraph: dict[str, Any],
    ) -> dict[str, Any]:
        del question, task_frame, evidence_subgraph
        answer_support: dict[str, list[str]] = defaultdict(list)
        confidences: dict[str, float] = defaultdict(float)
        for branch in branch_results:
            answer = str(branch.get("candidate_answer", "")).strip()
            branch_kind = str(branch.get("branch_kind", "")).strip()
            if not answer:
                continue
            answer_support[answer].append(branch_kind)
            confidences[answer] += float(branch.get("confidence", 0.0) or 0.0)

        ranked_answers = sorted(
            answer_support,
            key=lambda answer: (len(answer_support[answer]), confidences[answer]),
            reverse=True,
        )
        consensus = ranked_answers[0] if ranked_answers else ""
        preferred = answer_support.get(consensus, [])
        conflicts = []
        for answer in ranked_answers[1:]:
            conflicts.append({"answer": answer, "branches": answer_support[answer]})

        missing_requirements: list[str] = []
        for branch in branch_results:
            for item in branch.get("missing_requirements", []) or []:
                text = str(item).strip()
                if text and text not in missing_requirements:
                    missing_requirements.append(text)

        return {
            "consensus_answer": consensus,
            "agreement_groups": [preferred] if preferred else [],
            "conflicts": conflicts,
            "preferred_branches": preferred,
            "missing_requirements": missing_requirements,
            "notes": "Mock reconciliation prefers answers supported by multiple branches.",
        }

    def judge_sufficiency(
        self,
        question: str,
        task_frame: TaskFrame,
        branch_results: list[dict[str, Any]],
        merge_result: dict[str, Any],
        evidence_subgraph: dict[str, Any],
        iteration: int,
    ) -> dict[str, Any]:
        del question, task_frame
        consensus = str(merge_result.get("consensus_answer", "")).strip()
        preferred_branches = list(merge_result.get("preferred_branches", []))
        evidence_count = len(evidence_subgraph.get("hyperedge_ids", []))
        branch_answer_count = sum(1 for branch in branch_results if str(branch.get("candidate_answer", "")).strip())
        enough = bool(consensus) and (len(preferred_branches) >= 2 or evidence_count >= 4 or iteration >= 2 and branch_answer_count >= 2)
        missing_requirements = list(merge_result.get("missing_requirements", []))
        return {
            "enough": enough,
            "confidence": min(0.95, 0.35 + (0.15 * len(preferred_branches)) + (0.05 * evidence_count)),
            "reason": "Mock sufficiency requires a non-empty consensus answer plus multi-branch or evidence support.",
            "missing_requirements": [] if enough else missing_requirements,
            "next_focus": [] if enough else missing_requirements[:3],
        }

    def synthesize_answer(
        self,
        question: str,
        task_frame: TaskFrame,
        thought_graph: ThoughtGraph,
        evidence_subgraph: dict[str, Any],
        merge_result: dict[str, Any],
    ) -> dict[str, Any]:
        del task_frame, thought_graph
        answer = str(merge_result.get("consensus_answer", "")).strip()
        if not answer:
            branch_answers = evidence_subgraph.get("branch_answers", {})
            for payload in branch_answers.values():
                answer = str(payload.get("candidate_answer", "")).strip()
                if answer:
                    break
        if not answer:
            answer = f"No grounded answer was produced for: {question}"

        evidence_lines = []
        for item in evidence_subgraph.get("evidence", [])[:3]:
            if isinstance(item, dict):
                evidence_lines.append(short_text(str(item.get("content", "")), 180))
        reasoning_summary = " | ".join(evidence_lines) if evidence_lines else "Mock synthesis over accumulated evidence subgraph."
        remaining_gaps = list(merge_result.get("missing_requirements", [])) if not merge_result.get("consensus_answer") else []
        preferred_count = len(merge_result.get("preferred_branches", []))
        confidence = min(0.95, 0.4 + (0.12 * preferred_count))
        return {
            "answer": answer,
            "reasoning_summary": reasoning_summary,
            "confidence": confidence,
            "remaining_gaps": remaining_gaps,
        }


def _extract_topic_phrases(question: str) -> list[str]:
    cleaned = question.replace("?", " ").replace(",", " ").replace(";", " ")
    tokens = [token.strip() for token in cleaned.split() if token.strip()]
    capitalized: list[str] = []
    current: list[str] = []
    for token in tokens:
        if token[:1].isupper():
            current.append(token)
        elif current:
            capitalized.append(" ".join(current))
            current = []
    if current:
        capitalized.append(" ".join(current))
    if capitalized:
        return capitalized

    content = content_tokens(question)
    phrases: list[str] = []
    for index in range(len(content)):
        phrases.append(content[index])
        if index + 1 < len(content):
            phrases.append(f"{content[index]} {content[index + 1]}")
    deduped: list[str] = []
    for phrase in phrases:
        if phrase and phrase not in deduped:
            deduped.append(phrase)
    return deduped[:6]


def _infer_answer_type(question: str) -> str:
    lowered = question.lower()
    if lowered.startswith("when"):
        return "time or year"
    if lowered.startswith("where") or "what region" in lowered:
        return "location"
    if lowered.startswith("who"):
        return "person or organization"
    if "what farm animals" in lowered or "what animal" in lowered:
        return "animal or livestock type"
    if lowered.startswith("what"):
        return "entity, concept, or phrase"
    return "grounded short answer"


def _infer_relation_intent(question: str) -> str:
    lowered = question.lower()
    if "known for" in lowered:
        return "identify the entity that satisfies a set of defining properties"
    if "what concept" in lowered:
        return "find the shared concept across multiple evidence statements"
    if "what region" in lowered:
        return "find the location connected to multiple clues"
    if "what farm animals" in lowered:
        return "find the animal type linked to several properties"
    return "connect the topic entities and constraints to the missing answer"


def _infer_constraints(question: str) -> list[str]:
    lowered = question.lower()
    constraints: list[str] = []
    if "known for" in lowered:
        constraints.append("Answer must satisfy all listed descriptive clues.")
    if "both" in lowered:
        constraints.append("Evidence should connect both referenced situations.")
    if "and" in lowered:
        constraints.append("Prefer answers supported by multiple conjunctive facts.")
    return constraints


def _infer_candidate_answer(task_frame: TaskFrame, candidates: list[HyperedgeCandidate]) -> str:
    if not candidates:
        return ""
    excluded_texts = [*task_frame.topic_entities, *task_frame.hard_constraints, task_frame.relation_intent, task_frame.question]
    scores: dict[str, float] = defaultdict(float)
    for candidate in candidates:
        for entity_id in candidate.entity_ids:
            if lexical_overlap_score(excluded_texts, entity_id) > 0.5:
                continue
            scores[entity_id] += candidate.score
    if scores:
        ranked = sorted(scores, key=scores.get, reverse=True)
        return normalize_label(ranked[0])
    return normalize_label(candidates[0].hyperedge_id)
