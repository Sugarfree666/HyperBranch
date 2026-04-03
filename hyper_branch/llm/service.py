from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from ..models import TaskFrame, ThoughtGraph
from ..utils import content_tokens, lexical_overlap_score, normalize_label, short_text
from .client import OpenAICompatibleClient
from .prompts import PromptManager


class ReasoningService(ABC):
    @abstractmethod
    def build_task_frame(self, question: str, dataset_summary: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def judge_sufficiency(
        self,
        question: str,
        task_frame: TaskFrame,
        merge_result: dict[str, Any],
        evidence_subgraph: dict[str, Any],
        iteration: int,
        retrieval_control_state: dict[str, Any],
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

    def judge_sufficiency(
        self,
        question: str,
        task_frame: TaskFrame,
        merge_result: dict[str, Any],
        evidence_subgraph: dict[str, Any],
        iteration: int,
        retrieval_control_state: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "iteration": iteration,
            "task_frame_progress": task_frame.progress_snapshot(),
            "merge_result": merge_result,
            "evidence_subgraph": evidence_subgraph,
            "retrieval_control_state": retrieval_control_state,
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

    def judge_sufficiency(
        self,
        question: str,
        task_frame: TaskFrame,
        merge_result: dict[str, Any],
        evidence_subgraph: dict[str, Any],
        iteration: int,
        retrieval_control_state: dict[str, Any],
    ) -> dict[str, Any]:
        del question, task_frame, retrieval_control_state
        frontier = merge_result.get("frontier", [])
        answer_hypotheses = merge_result.get("answer_hypotheses", [])
        coverage = float(merge_result.get("coverage_summary", {}).get("topic_entity_coverage", 0.0) or 0.0)
        evidence_count = len(evidence_subgraph.get("evidence", []))
        enough = bool(frontier) and (coverage >= 0.5 or evidence_count >= 4 or (iteration >= 2 and bool(answer_hypotheses)))

        missing_requirements: list[str] = []
        next_focus: list[str] = []
        if not enough:
            if coverage < 0.5:
                missing_requirements.append("Need stronger anchor coverage across topic entities.")
                next_focus.append("bridge missing topic entities and improve anchor coverage")
            if not answer_hypotheses:
                missing_requirements.append("Need a more discriminative relation pattern to isolate the answer.")
                next_focus.append("relation closure around the most plausible frontier hyperedges")
            if evidence_count < 4:
                missing_requirements.append("Need more direct supporting evidence chunks.")
                next_focus.append("retrieve chunks that satisfy missing constraints more directly")

        return {
            "enough": enough,
            "confidence": min(0.95, 0.35 + (0.2 * coverage) + (0.08 * evidence_count)),
            "reason": "Mock sufficiency is based on frontier coverage, evidence volume, and whether the frontier already supports answer hypotheses.",
            "missing_requirements": [] if enough else missing_requirements,
            "next_focus": [] if enough else next_focus,
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
        answer = ""
        answer_hypotheses = merge_result.get("answer_hypotheses", [])
        if isinstance(answer_hypotheses, list):
            for hypothesis in answer_hypotheses:
                text = str(hypothesis).strip()
                if text:
                    answer = text
                    break
        if not answer:
            answer = f"No grounded answer was produced for: {question}"

        evidence_lines = []
        for item in evidence_subgraph.get("evidence", [])[:3]:
            if isinstance(item, dict):
                evidence_lines.append(short_text(str(item.get("content", "")), 180))
        reasoning_summary = " | ".join(evidence_lines) if evidence_lines else "Mock synthesis over accumulated evidence subgraph."
        remaining_gaps = list(merge_result.get("missing_requirements", []))
        confidence = min(
            0.95,
            0.3 + (0.15 * float(merge_result.get("coverage_summary", {}).get("topic_entity_coverage", 0.0) or 0.0)),
        )
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
