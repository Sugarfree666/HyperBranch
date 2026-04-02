from __future__ import annotations

import logging
from typing import Any

from ..config import ReasoningConfig
from ..models import EvidenceItem
from ..models import ThoughtState
from ..utils import cosine_similarity


class ThoughtScorer:
    def __init__(self, embedder: Any, config: ReasoningConfig, logger: logging.Logger) -> None:
        self.embedder = embedder
        self.config = config
        self.logger = logger

    def score_thoughts(
        self,
        question: str,
        thoughts: list[ThoughtState],
        evidence_by_thought: dict[str, list[EvidenceItem]] | None = None,
    ) -> list[ThoughtState]:
        if not thoughts:
            return []
        question_vector = self.embedder.embed_texts([question], stage="thought_scoring_question")[0]
        thought_vectors = self.embedder.embed_texts([thought.content for thought in thoughts], stage="thought_scoring_content")
        evidence_by_thought = evidence_by_thought or {}
        grounding_texts = [
            self._compose_grounding_text(thought, evidence_by_thought.get(thought.thought_id, []))
            for thought in thoughts
        ]
        grounding_vectors = self.embedder.embed_texts(
            [text if text else "ungrounded thought" for text in grounding_texts],
            stage="thought_scoring_grounding",
        )

        for thought, thought_vector, grounding_vector, grounding_text in zip(
            thoughts, thought_vectors, grounding_vectors, grounding_texts, strict=True
        ):
            task_score = max(cosine_similarity(thought_vector, question_vector), 0.0)
            grounding_score = max(cosine_similarity(thought_vector, grounding_vector), 0.0) if grounding_text else 0.0
            thought.score = task_score * (1.0 + grounding_score)
            thought.metadata["score_breakdown"] = {
                "task_score": task_score,
                "grounding_score": grounding_score,
            }
        self.logger.info("Scored %s thoughts", len(thoughts))
        return thoughts

    def _compose_grounding_text(self, thought: ThoughtState, evidence_items: list[EvidenceItem]) -> str:
        parts: list[str] = []
        base_grounding = thought.grounding.to_text()
        if base_grounding:
            parts.append(base_grounding)
        if evidence_items:
            evidence_bits = []
            for item in evidence_items[:3]:
                evidence_bits.append(item.content[:220])
            parts.append("evidence: " + " | ".join(evidence_bits))
        return "; ".join(parts)

    def shortlist(self, thoughts: list[ThoughtState]) -> list[ThoughtState]:
        runnable_roles = {"hypothesis", "constraint", "bridge"}
        candidates = [
            thought
            for thought in thoughts
            if thought.status == "active"
            and thought.score >= self.config.thought_score_threshold
            and thought.role in runnable_roles
        ]
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: self.config.coarse_top_k]
