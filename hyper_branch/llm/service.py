from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from ..models import TaskFrame, ThoughtGraph
from ..utils import content_tokens, ensure_list, normalize_label, short_text
from .client import OpenAICompatibleClient
from .prompts import PromptManager
from .views import build_llm_thought_graph_summary


ANSWER_PREFIX_RE = re.compile(r"^(?:answer\s*:\s*|the answer is\s+|it is\s+|it's\s+|they are\s+|this is\s+)", re.IGNORECASE)
LEADING_ARTICLE_RE = re.compile(r"^(?:the|an?)\s+", re.IGNORECASE)
ABSTRACT_TARGET_HEAD_RE = re.compile(
    r"^(?:overall\s+)?(?:promotion|development|preservation|improvement|support|role|importance|quality|efficiency|empowerment|involvement)\s+of\s+",
    re.IGNORECASE,
)
ABSTRACT_TARGET_PAIR_HEAD_RE = re.compile(
    r"^(?:overall\s+)?(?:empowerment and involvement|support and development|growth and development)\s+of\s+",
    re.IGNORECASE,
)
TRAILING_CONTEXT_RE = re.compile(r"\b(?:for|in|across|within|among|throughout)\b", re.IGNORECASE)
ANSWER_RELATION_MARKERS = (
    "contribute to ",
    "contributes to ",
    "contributing to ",
    "lead to ",
    "leads to ",
    "leading to ",
    "result in ",
    "results in ",
    "resulting in ",
    "promote ",
    "promotes ",
    "promoting ",
    "preserve ",
    "preserves ",
    "preserving ",
    "develop ",
    "develops ",
    "developing ",
    "foster ",
    "fosters ",
    "fostering ",
)
ABSTRACT_TARGET_SUFFIXES = (
    " improvement",
    " improvements",
    " outcome",
    " outcomes",
)
GENERIC_ANSWER_TOKENS = {
    "answer",
    "concept",
    "entity",
    "grounded",
    "location",
    "organization",
    "outcome",
    "person",
    "phrase",
    "short",
    "time",
    "type",
    "year",
}


class ReasoningService(ABC):
    @abstractmethod
    def build_task_frame(self, question: str, dataset_summary: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def judge_sufficiency(
        self,
        question: str,
        task_frame: TaskFrame,
        llm_evidence_view: dict[str, Any],
        iteration: int,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def synthesize_answer(
        self,
        question: str,
        task_frame: TaskFrame,
        thought_graph: ThoughtGraph,
        llm_evidence_view: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def select_expansion_entities(
        self,
        question: str,
        task_frame: TaskFrame,
        candidate_entities: list[dict[str, Any]],
        control_state: Any,
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
        llm_evidence_view: dict[str, Any],
        iteration: int,
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "iteration": iteration,
            "question_goal": task_frame.answerability_snapshot(),
            "llm_evidence_view": llm_evidence_view,
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
        llm_evidence_view: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "task_frame_progress": task_frame.progress_snapshot(),
            "llm_evidence_view": llm_evidence_view,
            "thought_graph_summary": build_llm_thought_graph_summary(thought_graph),
        }
        response = self.client.chat_json("final_answer", self.prompts.get("final_answer"), payload)
        response["answer"] = _coerce_direct_answer(question, response.get("answer", ""), llm_evidence_view)
        response.setdefault("reasoning_summary", "")
        response.setdefault("confidence", 0.0)
        response.setdefault("remaining_gaps", [])
        return response

    def select_expansion_entities(
        self,
        question: str,
        task_frame: TaskFrame,
        candidate_entities: list[dict[str, Any]],
        control_state: Any,
    ) -> dict[str, Any]:
        if not candidate_entities:
            return {"selected_entity_ids": [], "reason": "No fresh expansion entities were available."}
        payload = {
            "question": question,
            "topic_entities": list(task_frame.topic_entities),
            "hard_constraints": list(task_frame.hard_constraints),
            "relation_intent": task_frame.relation_intent,
            "current_focus": list(control_state.current_focus()),
            "selection_limit": 2,
            "candidate_entities": candidate_entities,
        }
        response = self.client.chat_json("entity_frontier", self.prompts.get("entity_frontier"), payload)
        response.setdefault("selected_entity_ids", [])
        response.setdefault("reason", "")
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
        llm_evidence_view: dict[str, Any],
        iteration: int,
    ) -> dict[str, Any]:
        del question, task_frame
        frontier = llm_evidence_view.get("frontier_hyperedges", [])
        answer_hypotheses = llm_evidence_view.get("coverage_summary", {}).get("answer_hypotheses", [])
        covered_topics = llm_evidence_view.get("coverage_summary", {}).get("topics", {}).get("covered", [])
        total_topics = covered_topics + llm_evidence_view.get("coverage_summary", {}).get("topics", {}).get("missing", [])
        coverage = len(covered_topics) / max(len(total_topics), 1)
        evidence_count = len([item for item in frontier if str(item.get("core_evidence", "")).strip()])
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
        llm_evidence_view: dict[str, Any],
    ) -> dict[str, Any]:
        del task_frame, thought_graph
        answer = ""
        answer_hypotheses = llm_evidence_view.get("coverage_summary", {}).get("answer_hypotheses", [])
        if isinstance(answer_hypotheses, list):
            for hypothesis in answer_hypotheses:
                text = str(hypothesis).strip()
                if text:
                    answer = text
                    break
        if not answer:
            answer = f"No grounded answer was produced for: {question}"
        answer = _coerce_direct_answer(question, answer, llm_evidence_view)

        evidence_lines = []
        for item in llm_evidence_view.get("frontier_hyperedges", [])[:3]:
            if isinstance(item, dict):
                evidence_lines.append(short_text(str(item.get("core_evidence", "")), 180))
        reasoning_summary = " | ".join(evidence_lines) if evidence_lines else "Mock synthesis over compressed evidence view."
        remaining_gaps = list(llm_evidence_view.get("missing_requirements", []))
        covered_topics = llm_evidence_view.get("coverage_summary", {}).get("topics", {}).get("covered", [])
        total_topics = covered_topics + llm_evidence_view.get("coverage_summary", {}).get("topics", {}).get("missing", [])
        confidence = min(
            0.95,
            0.3 + (0.15 * (len(covered_topics) / max(len(total_topics), 1))),
        )
        return {
            "answer": answer,
            "reasoning_summary": reasoning_summary,
            "confidence": confidence,
            "remaining_gaps": remaining_gaps,
        }

    def select_expansion_entities(
        self,
        question: str,
        task_frame: TaskFrame,
        candidate_entities: list[dict[str, Any]],
        control_state: Any,
    ) -> dict[str, Any]:
        del question, task_frame, control_state
        return {
            "selected_entity_ids": [
                str(candidate.get("entity_id", "")).strip()
                for candidate in candidate_entities[:2]
                if str(candidate.get("entity_id", "")).strip()
            ],
            "reason": "Mock selector kept the top coarse-ranked fresh entities.",
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


def _coerce_direct_answer(question: str, answer: str, llm_evidence_view: dict[str, Any]) -> str:
    cleaned_answer = _clean_answer_text(answer)
    candidates = _direct_answer_candidates(question, llm_evidence_view)

    if cleaned_answer:
        tail_candidate = _extract_answer_relation_tail(cleaned_answer)
        if tail_candidate:
            for variant in _answer_variants(tail_candidate):
                _register_candidate(candidates, variant, "answer_tail", bonus=0.7)

    if not cleaned_answer:
        best = _best_direct_candidate("", candidates, llm_evidence_view)
        return best["text"] if best else ""

    if not _is_sentence_like_answer(cleaned_answer):
        best = _best_direct_candidate(cleaned_answer, candidates, llm_evidence_view)
        if best and _clean_answer_text(best["text"]).lower() in cleaned_answer.lower():
            return best["text"]
        return cleaned_answer

    best = _best_direct_candidate(cleaned_answer, candidates, llm_evidence_view)
    if best:
        return best["text"]

    fallback = _extract_short_answer_fallback(cleaned_answer)
    return fallback or short_text(cleaned_answer, 120)


def _direct_answer_candidates(question: str, llm_evidence_view: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    coverage_summary = llm_evidence_view.get("coverage_summary", {})
    how_question = question.lstrip().lower().startswith("how ")

    for hypothesis in ensure_list(coverage_summary.get("answer_hypotheses", [])):
        _register_candidate(candidates, hypothesis, "answer_hypothesis", bonus=0.55)

    target = coverage_summary.get("target", {})
    target_supported = str(target.get("status", "")).strip().lower() == "supported"
    target_bonus = 0.65 if how_question and target_supported else 0.45 if target_supported else 0.25
    for variant in _answer_variants(str(target.get("text", "") or "")):
        _register_candidate(candidates, variant, "coverage_target", bonus=target_bonus)

    for variant in _answer_variants(_extract_question_target(question)):
        _register_candidate(candidates, variant, "question_target", bonus=0.5 if how_question else 0.2)

    for item in ensure_list(llm_evidence_view.get("frontier_hyperedges", []))[:3]:
        if not isinstance(item, dict):
            continue
        for entity in ensure_list(item.get("core_entities", [])):
            _register_candidate(candidates, entity, "frontier_entity", bonus=0.25)
        for topic in ensure_list(item.get("matched_topics", [])):
            _register_candidate(candidates, topic, "frontier_topic", bonus=0.15)
    return candidates


def _register_candidate(
    candidates: dict[str, dict[str, Any]],
    text: str,
    source: str,
    *,
    bonus: float,
) -> None:
    cleaned = _clean_answer_text(text)
    if not cleaned:
        return
    if _is_generic_answer_candidate(cleaned):
        return
    token_count = len(content_tokens(cleaned)) or len(cleaned.split())
    if token_count > 8:
        return
    key = cleaned.lower()
    if key not in candidates or bonus > float(candidates[key].get("bonus", 0.0)):
        candidates[key] = {"text": cleaned, "source": source, "bonus": bonus}


def _best_direct_candidate(
    answer: str,
    candidates: dict[str, dict[str, Any]],
    llm_evidence_view: dict[str, Any],
) -> dict[str, Any] | None:
    if not candidates:
        return None

    reference_texts = [str(llm_evidence_view.get("evidence_summary", "") or "")]
    frontier = llm_evidence_view.get("frontier_hyperedges", [])
    if isinstance(frontier, list):
        for item in frontier[:3]:
            if not isinstance(item, dict):
                continue
            reference_texts.append(str(item.get("core_evidence", "") or ""))
            reference_texts.append(str(item.get("hyperedge", "") or ""))

    best: dict[str, Any] | None = None
    best_score = 0.0
    normalized_answer = _clean_answer_text(answer).lower()
    for candidate in candidates.values():
        candidate_text = candidate["text"]
        score = float(candidate.get("bonus", 0.0))
        if answer:
            score += 1.4 * _candidate_overlap_score([answer], candidate_text)
            if candidate_text.lower() in normalized_answer:
                score += 1.0
        score += 0.7 * _candidate_overlap_score(reference_texts, candidate_text)
        if len(content_tokens(candidate_text)) <= 4:
            score += 0.1
        if best is None or score > best_score:
            best = candidate
            best_score = score

    if not best:
        return None
    if best_score >= 0.65:
        return best
    if best_score >= 0.45 and str(best.get("source", "")) in {
        "coverage_target",
        "question_target",
        "answer_hypothesis",
        "answer_tail",
    }:
        return best
    return None


def _candidate_overlap_score(reference_texts: list[str], candidate_text: str) -> float:
    candidate_tokens = set(content_tokens(candidate_text))
    if not candidate_tokens:
        return 0.0

    scores: list[float] = []
    for reference in reference_texts:
        reference_tokens = set(content_tokens(reference))
        if not reference_tokens:
            continue
        overlap = len(reference_tokens & candidate_tokens) / max(len(candidate_tokens), 1)
        scores.append(overlap)
    return max(scores, default=0.0)


def _answer_variants(text: str) -> list[str]:
    cleaned = _clean_answer_text(text)
    if not cleaned:
        return []

    variants = [cleaned]
    article_stripped = LEADING_ARTICLE_RE.sub("", cleaned).strip()
    if article_stripped and article_stripped not in variants:
        variants.append(article_stripped)

    for candidate in list(variants):
        stripped_pair = ABSTRACT_TARGET_PAIR_HEAD_RE.sub("", candidate).strip()
        if stripped_pair and stripped_pair not in variants:
            variants.append(stripped_pair)
        stripped_head = ABSTRACT_TARGET_HEAD_RE.sub("", candidate).strip()
        if stripped_head and stripped_head not in variants:
            variants.append(stripped_head)

    expanded: list[str] = []
    for candidate in variants:
        lowered = candidate.lower()
        for suffix in ABSTRACT_TARGET_SUFFIXES:
            if lowered.endswith(suffix):
                trimmed = candidate[: -len(suffix)].strip()
                if trimmed:
                    expanded.append(trimmed)
        if " related to " in lowered:
            expanded.append(candidate.split(" related to ", 1)[1].strip())
        marker_match = TRAILING_CONTEXT_RE.search(candidate)
        if marker_match and marker_match.start() > 0:
            expanded.append(candidate[: marker_match.start()].strip())
    variants.extend(expanded)

    deduped: list[str] = []
    for candidate in variants:
        normalized = _clean_answer_text(candidate)
        token_count = len(content_tokens(normalized)) or len(normalized.split())
        if normalized and token_count <= 8 and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _extract_question_target(question: str) -> str:
    lowered = question.lower().strip()
    for marker in ANSWER_RELATION_MARKERS:
        index = lowered.rfind(marker)
        if index == -1:
            continue
        return question[index + len(marker) :].rstrip(" ?.")
    return ""


def _extract_answer_relation_tail(answer: str) -> str:
    lowered = answer.lower()
    for marker in ANSWER_RELATION_MARKERS:
        index = lowered.find(marker)
        if index == -1:
            continue
        tail = answer[index + len(marker) :].strip()
        if not tail:
            continue
        tail = re.split(r"[.;]|,\s+(?:which|and|but)\b|\b(?:which|because|by|through|while|whereas)\b", tail, maxsplit=1, flags=re.IGNORECASE)[0]
        return tail.strip(" ,")
    return ""


def _extract_short_answer_fallback(answer: str) -> str:
    extracted_tail = _extract_answer_relation_tail(answer)
    variants = _answer_variants(extracted_tail)
    if variants:
        return variants[0]

    first_clause = re.split(r"[.;]|,\s+(?:which|and|but)\b", answer, maxsplit=1, flags=re.IGNORECASE)[0]
    first_clause = _clean_answer_text(first_clause)
    if 0 < (len(content_tokens(first_clause)) or len(first_clause.split())) <= 8:
        return first_clause
    return ""


def _clean_answer_text(text: str) -> str:
    cleaned = short_text(normalize_label(str(text).strip()), 240)
    cleaned = ANSWER_PREFIX_RE.sub("", cleaned).strip()
    cleaned = cleaned.strip(" '\"")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.rstrip(" .;,")


def _is_generic_answer_candidate(text: str) -> bool:
    tokens = content_tokens(text)
    if not tokens:
        return True
    if len(tokens) <= 4 and all(token in GENERIC_ANSWER_TOKENS for token in tokens):
        return True
    return "missing answer" in text.lower()


def _is_sentence_like_answer(answer: str) -> bool:
    lowered = answer.lower()
    token_count = len(content_tokens(answer)) or len(answer.split())
    if token_count > 8:
        return True
    if any(punct in answer for punct in (".", ";")):
        return True
    if "," in answer and token_count > 5:
        return True
    if lowered.startswith(("by ", "because ", "through ", "these ", "this ", "they ", "initiatives ")):
        return True
    if token_count <= 4:
        return False
    return any(
        marker in lowered
        for marker in (
            " which ",
            " because ",
            " by ",
            " through ",
            " contribute ",
            " contributes ",
            " leading to ",
            " resulting in ",
            " improving ",
            " enhancing ",
            " reducing ",
            " building ",
            " fostering ",
            " supporting ",
        )
    )
