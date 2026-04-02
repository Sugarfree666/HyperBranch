from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import EvidenceItem, TaskFrame, ThoughtGraph, ThoughtState
from ..utils import short_text
from .client import OpenAICompatibleClient
from .prompts import PromptManager


class ReasoningService(ABC):
    @abstractmethod
    def build_task_frame(self, question: str, dataset_summary: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def select_thoughts(
        self,
        question: str,
        task_frame: TaskFrame,
        candidate_thoughts: list[ThoughtState],
        top_k: int,
    ) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def decide_operation(
        self,
        question: str,
        task_frame: TaskFrame,
        thought: ThoughtState,
        evidence_items: list[EvidenceItem],
        related_thoughts: list[ThoughtState],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def synthesize_answer(
        self,
        question: str,
        task_frame: TaskFrame,
        thought_graph: ThoughtGraph,
        verified_reasoning: list[ThoughtState],
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
        return {
            "anchors": response.get("anchors", []),
            "target": response.get("target", question),
            "constraints": response.get("constraints", []),
            "bridges": response.get("bridges", []),
        }

    def select_thoughts(
        self,
        question: str,
        task_frame: TaskFrame,
        candidate_thoughts: list[ThoughtState],
        top_k: int,
    ) -> list[str]:
        payload = {
            "question": question,
            "top_k": top_k,
            "task_frame_progress": task_frame.progress_snapshot(),
            "candidate_thoughts": [thought.brief() for thought in candidate_thoughts],
        }
        response = self.client.chat_json("thought_selector", self.prompts.get("thought_selector"), payload)
        selected = response.get("selected_thought_ids", [])
        if not isinstance(selected, list):
            return []
        valid_ids = {thought.thought_id for thought in candidate_thoughts}
        return [thought_id for thought_id in selected if thought_id in valid_ids][:top_k]

    def decide_operation(
        self,
        question: str,
        task_frame: TaskFrame,
        thought: ThoughtState,
        evidence_items: list[EvidenceItem],
        related_thoughts: list[ThoughtState],
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "task_frame_progress": task_frame.progress_snapshot(),
            "thought": thought.to_dict(),
            "evidence_items": [item.to_dict() for item in evidence_items],
            "related_thoughts": [item.brief() for item in related_thoughts],
        }
        response = self.client.chat_json("thought_operations", self.prompts.get("thought_operations"), payload)
        response.setdefault("operation", "verify")
        response.setdefault("new_status", "active")
        response.setdefault("new_thoughts", [])
        response.setdefault(
            "verification",
            {"verdict": "insufficient", "confidence": 0.0, "evidence_ids": [], "notes": ""},
        )
        response.setdefault("merge_with_thought_ids", [])
        return response

    def synthesize_answer(
        self,
        question: str,
        task_frame: TaskFrame,
        thought_graph: ThoughtGraph,
        verified_reasoning: list[ThoughtState],
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "task_frame_progress": task_frame.progress_snapshot(),
            "thought_graph_summary": {
                "status": thought_graph.status,
                "frontier_ids": thought_graph.frontier_ids,
                "thoughts": [thought.brief() for thought in thought_graph.thoughts.values()],
            },
            "verified_reasoning": [thought.brief() for thought in verified_reasoning],
        }
        response = self.client.chat_json("final_answer", self.prompts.get("final_answer"), payload)
        response.setdefault("answer", "")
        response.setdefault("reasoning_summary", "")
        response.setdefault("confidence", 0.0)
        response.setdefault("remaining_gaps", [])
        return response


class MockReasoningService(ReasoningService):
    def build_task_frame(self, question: str, dataset_summary: dict[str, Any]) -> dict[str, Any]:
        tokens = [token.strip(" ,.?;:") for token in question.split() if token.strip(" ,.?;:")]
        anchors = tokens[: min(3, len(tokens))]
        target = question
        bridges = [f"Connect evidence that explains {anchors[0]}" for anchors in [anchors] if anchors]
        constraints = ["Use grounded evidence from retrieved chunks.", "Prefer evidence chains over unsupported claims."]
        return {
            "anchors": anchors,
            "target": target,
            "constraints": constraints,
            "bridges": bridges,
        }

    def select_thoughts(
        self,
        question: str,
        task_frame: TaskFrame,
        candidate_thoughts: list[ThoughtState],
        top_k: int,
    ) -> list[str]:
        return [thought.thought_id for thought in sorted(candidate_thoughts, key=lambda item: item.score, reverse=True)[:top_k]]

    def decide_operation(
        self,
        question: str,
        task_frame: TaskFrame,
        thought: ThoughtState,
        evidence_items: list[EvidenceItem],
        related_thoughts: list[ThoughtState],
    ) -> dict[str, Any]:
        if evidence_items and evidence_items[0].score >= 0.2:
            return {
                "operation": "verify",
                "reason": "Top evidence is sufficiently aligned with the current thought.",
                "new_status": "verified",
                "new_thoughts": [],
                "merge_with_thought_ids": [],
                "verification": {
                    "verdict": "supported",
                    "confidence": min(0.95, 0.4 + evidence_items[0].score),
                    "evidence_ids": [evidence_items[0].evidence_id],
                    "notes": short_text(evidence_items[0].content, 180),
                },
            }

        expansion = f"Refine this reasoning branch: {thought.content}"
        return {
            "operation": "expand",
            "reason": "Evidence is still weak, so the branch should be refined.",
            "new_status": "expanded",
            "new_thoughts": [
                {
                    "content": expansion,
                    "objective": thought.objective,
                    "slot_id": thought.slot_id,
                    "metadata": {"intent": thought.metadata.get("intent", "followup")},
                }
            ],
            "merge_with_thought_ids": [],
            "verification": {
                "verdict": "insufficient",
                "confidence": 0.2,
                "evidence_ids": [],
                "notes": "Need stronger evidence.",
            },
        }

    def synthesize_answer(
        self,
        question: str,
        task_frame: TaskFrame,
        thought_graph: ThoughtGraph,
        verified_reasoning: list[ThoughtState],
    ) -> dict[str, Any]:
        snippets = [
            short_text(thought.content, 160)
            for thought in verified_reasoning[:3]
        ]
        if not snippets:
            snippets = [short_text(thought.content, 160) for thought in thought_graph.thoughts.values() if thought.kind == "reasoning"][:3]
        answer = " ".join(snippets) if snippets else f"No grounded answer was produced for: {question}"
        return {
            "answer": answer,
            "reasoning_summary": f"Mock synthesis over {len(verified_reasoning)} verified reasoning thoughts.",
            "confidence": 0.55 if verified_reasoning else 0.25,
            "remaining_gaps": [] if verified_reasoning else ["No verified reasoning thoughts were available."],
        }
