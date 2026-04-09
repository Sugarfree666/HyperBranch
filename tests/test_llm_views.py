from __future__ import annotations

import unittest

from hyper_branch.llm.service import OpenAIReasoningService
from hyper_branch.llm.views import FORBIDDEN_LLM_FIELDS, build_llm_evidence_view
from hyper_branch.models import EvidenceItem, EvidenceSubgraph, RetrievalControlState, TaskFrame, ThoughtGraph, ThoughtState


class DummyClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def chat_json(self, stage: str, prompt: str, payload: dict) -> dict:
        self.calls.append((stage, payload))
        del prompt
        if stage == "evidence_judge":
            return {
                "enough": False,
                "confidence": 0.4,
                "reason": "stub",
                "missing_requirements": [],
                "next_focus": [],
            }
        return {
            "answer": "stub answer",
            "reasoning_summary": "stub reasoning",
            "confidence": 0.5,
            "remaining_gaps": [],
        }


class DummyPrompts:
    def get(self, name: str) -> str:
        return name


class LongAnswerClient(DummyClient):
    def chat_json(self, stage: str, prompt: str, payload: dict) -> dict:
        self.calls.append((stage, payload))
        del prompt
        if stage == "evidence_judge":
            return {
                "enough": True,
                "confidence": 0.8,
                "reason": "stub",
                "missing_requirements": [],
                "next_focus": [],
            }
        return {
            "answer": (
                "Initiatives aimed at improving neighborhoods and combating food insecurity "
                "significantly contribute to community health in St. Louis and across the nation "
                "by enhancing neighborhood aesthetics, reducing crime, and building community bonds."
            ),
            "reasoning_summary": "stub reasoning",
            "confidence": 0.7,
            "remaining_gaps": [],
        }


def contains_forbidden_key(payload) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in FORBIDDEN_LLM_FIELDS:
                return True
            if contains_forbidden_key(value):
                return True
    if isinstance(payload, list):
        return any(contains_forbidden_key(item) for item in payload)
    return False


class LlmEvidenceViewTest(unittest.TestCase):
    def _build_task_frame(self) -> TaskFrame:
        task_frame = TaskFrame.from_payload(
            "What region is known for its robust distribution network for local food and has a college operating a farm for over a hundred years?",
            {
                "anchors": ["region", "distribution network", "college"],
                "target": "region",
                "constraints": [
                    "robust distribution network for local food",
                    "college operating a farm for over a hundred years",
                ],
                "bridges": ["find the location connected to multiple clues"],
                "topic_entities": ["region", "distribution network", "local food", "college"],
                "answer_type_hint": "location",
                "relation_intent": "find the location connected to multiple clues",
                "hard_constraints": [
                    "robust distribution network for local food",
                    "college operating a farm for over a hundred years",
                ],
                "relation_skeleton": "region linked to food distribution and long-running college farm",
            },
        )
        task_frame.mark_slot("anchor-1", status="supported")
        task_frame.mark_slot("constraint-0", status="supported")
        task_frame.mark_slot("bridge-0", status="supported")
        return task_frame

    def test_build_llm_evidence_view_strips_noisy_fields(self) -> None:
        task_frame = self._build_task_frame()
        evidence_subgraph = EvidenceSubgraph(
            evidence=[
                EvidenceItem(
                    evidence_id="ev-1",
                    chunk_id="chunk-1",
                    content=(
                        "Though Jackson acknowledges that some communities may not have a good distribution network in place.\n\n"
                        "western North Carolina is endowed with a robust and growing infrastructure for local food."
                    ),
                    score=0.82,
                    source_node_ids=["he-1", '"WESTERN NORTH CAROLINA"'],
                ),
                EvidenceItem(
                    evidence_id="ev-2",
                    chunk_id="chunk-2",
                    content=(
                        "For more than a hundred years, Warren Wilson College in western North Carolina has operated a farm.\n\n"
                        "Warren Wilson College in western North Carolina has operated a farm of about 340 acres for over a hundred years."
                    ),
                    score=0.79,
                    source_node_ids=["he-2", '"WARREN WILSON COLLEGE"'],
                ),
            ]
        )
        merge_result = {
            "frontier": [
                {
                    "hyperedge_id": "he-1",
                    "hyperedge_text": "western North Carolina is endowed with a robust and growing infrastructure for local food.",
                    "matched_topic_labels": ["distribution network"],
                    "support_entity_labels": ["WESTERN NORTH CAROLINA", "DISTRIBUTION NETWORK"],
                    "chunk_ids": ["chunk-1"],
                    "supporting_chunks": ["western North Carolina is endowed with a robust and growing infrastructure."],
                    "score": 0.9,
                    "branch_score": 0.8,
                    "fused_score": 0.85,
                    "coverage_gain": 0.6,
                },
                {
                    "hyperedge_id": "he-2",
                    "hyperedge_text": "Warren Wilson College in western North Carolina has operated a farm for over a hundred years.",
                    "matched_topic_labels": ["college"],
                    "support_entity_labels": ["WARREN WILSON COLLEGE", "WESTERN NORTH CAROLINA"],
                    "chunk_ids": ["chunk-2"],
                    "supporting_chunks": ["Warren Wilson College has operated a farm for over a hundred years."],
                    "relation_gain": 0.7,
                    "penalty": 0.0,
                },
            ],
            "branch_contributions": {
                "constraint": ["he-1", "he-2"],
                "anchor": ["he-1"],
                "relation": ["he-2"],
            },
            "preferred_branches": ["constraint", "anchor"],
            "answer_hypotheses": ["WESTERN NORTH CAROLINA"],
            "missing_requirements": ["Need stronger direct support for the college clue."],
            "next_focus": ["look for college evidence tied to the same region"],
        }
        control_state = RetrievalControlState(
            iteration=2,
            branch_weights={"constraint": 0.5, "relation": 0.3, "anchor": 0.2},
            missing_requirements=["Need stronger direct support for the college clue."],
            next_focus=["look for college evidence tied to the same region"],
        )

        view = build_llm_evidence_view(
            question=task_frame.question,
            task_frame=task_frame,
            evidence_subgraph=evidence_subgraph,
            merge_result=merge_result,
            control_state=control_state,
        )

        self.assertEqual(view["question"], task_frame.question)
        self.assertEqual(len(view["frontier_hyperedges"]), 2)
        self.assertEqual(view["frontier_hyperedges"][0]["supporting_branches"], ["constraint", "anchor"])
        self.assertIn("western North Carolina", view["frontier_hyperedges"][0]["core_evidence"])
        self.assertEqual(view["coverage_summary"]["constraints"]["covered"], ["robust distribution network for local food"])
        self.assertEqual(
            view["coverage_summary"]["constraints"]["missing"],
            ["college operating a farm for over a hundred years"],
        )
        self.assertEqual(view["answerability_summary"]["candidate_answer"], "WESTERN NORTH CAROLINA")
        self.assertIn(
            "Checklist coverage is advisory",
            view["answerability_summary"]["note"],
        )
        self.assertTrue(view["answerability_summary"]["supporting_evidence"])
        self.assertFalse(contains_forbidden_key(view))

    def test_openai_reasoning_service_uses_compressed_views_only(self) -> None:
        client = DummyClient()
        service = OpenAIReasoningService(client=client, prompts=DummyPrompts())
        task_frame = self._build_task_frame()
        thought_graph = ThoughtGraph(question=task_frame.question, root_id="th-0001")
        thought_graph.add_thought(
            ThoughtState(
                thought_id="th-0001",
                kind="reasoning",
                content="constraint operator recommended hyperedges: he-1 (0.812)",
                objective="constraint hyperedge search operator",
                slot_id="target-0",
                status="searched",
            )
        )
        llm_evidence_view = {
            "question": task_frame.question,
            "current_focus": ["college clue"],
            "missing_requirements": ["Need stronger direct support for the college clue."],
            "next_focus": ["look for college evidence tied to the same region"],
            "frontier_hyperedges": [
                {
                    "hyperedge": "western North Carolina is endowed with a robust and growing infrastructure for local food.",
                    "supporting_branches": ["constraint", "anchor"],
                    "matched_topics": ["distribution network"],
                    "core_entities": ["WESTERN NORTH CAROLINA"],
                    "core_evidence": "western North Carolina is endowed with a robust and growing infrastructure for local food.",
                }
            ],
            "coverage_summary": {
                "topics": {"covered": ["distribution network"], "missing": ["college"]},
                "constraints": {"covered": ["robust distribution network for local food"], "missing": ["college operating a farm for over a hundred years"]},
                "relations": {"intent": "find the location connected to multiple clues", "covered": ["find the location connected to multiple clues"], "missing": []},
                "target": {"text": "region", "status": "open"},
                "answer_hypotheses": ["WESTERN NORTH CAROLINA"],
            },
            "control_summary": {
                "iteration": 2,
                "branch_weights": {"constraint": 0.5, "relation": 0.3, "anchor": 0.2},
                "preferred_branches": ["constraint"],
            },
            "evidence_summary": "Current answer hypotheses: WESTERN NORTH CAROLINA",
        }

        service.judge_sufficiency(
            question=task_frame.question,
            task_frame=task_frame,
            llm_evidence_view=llm_evidence_view,
            iteration=2,
        )
        service.synthesize_answer(
            question=task_frame.question,
            task_frame=task_frame,
            thought_graph=thought_graph,
            llm_evidence_view=llm_evidence_view,
        )

        judge_payload = client.calls[0][1]
        answer_payload = client.calls[1][1]

        self.assertIn("llm_evidence_view", judge_payload)
        self.assertIn("question_goal", judge_payload)
        self.assertNotIn("task_frame_progress", judge_payload)
        self.assertNotIn("evidence_subgraph", judge_payload)
        self.assertNotIn("merge_result", judge_payload)
        self.assertNotIn("retrieval_control_state", judge_payload)
        self.assertEqual(judge_payload["question_goal"]["target"], "region")
        self.assertIn("soft guidance", judge_payload["question_goal"]["guidance"])
        self.assertFalse(contains_forbidden_key(judge_payload["llm_evidence_view"]))
        self.assertNotIn("score", answer_payload["thought_graph_summary"]["recent_thoughts"][0])
        self.assertNotIn("content", answer_payload["thought_graph_summary"]["recent_thoughts"][0])

    def test_synthesize_answer_coerces_long_explanation_into_direct_answer(self) -> None:
        client = LongAnswerClient()
        service = OpenAIReasoningService(client=client, prompts=DummyPrompts())
        question = (
            "How do initiatives aimed at improving neighborhoods and combating food insecurity "
            "contribute to the overall community health in St. Louis and across the nation?"
        )
        task_frame = TaskFrame.from_payload(
            question,
            {
                "anchors": ["initiatives", "neighborhoods", "food insecurity", "community health"],
                "target": "overall community health improvement",
                "constraints": ["involves neighborhoods", "focus on combating food insecurity"],
                "bridges": ["impact of initiatives on community health"],
                "topic_entities": ["initiatives", "neighborhoods", "food insecurity", "community health"],
                "answer_type_hint": "concept or outcome",
                "relation_intent": "impact of initiatives on community health",
                "hard_constraints": ["involves neighborhoods", "focus on combating food insecurity"],
                "relation_skeleton": "initiatives linked to community health outcomes",
            },
        )
        task_frame.mark_slot("target-0", status="supported")
        thought_graph = ThoughtGraph(question=question, root_id="th-0001")
        llm_evidence_view = {
            "question": question,
            "current_focus": [],
            "missing_requirements": [],
            "next_focus": [],
            "frontier_hyperedges": [
                {
                    "hyperedge": "These initiatives improve community health while supporting food security.",
                    "supporting_branches": ["constraint", "relation"],
                    "matched_topics": ["community health"],
                    "core_entities": ["RESILIENT COMMUNITY"],
                    "core_evidence": "These initiatives improve community health while supporting food security.",
                }
            ],
            "coverage_summary": {
                "topics": {"covered": ["community health"], "missing": []},
                "constraints": {"covered": ["involves neighborhoods", "focus on combating food insecurity"], "missing": []},
                "relations": {"intent": "impact of initiatives on community health", "covered": ["impact of initiatives on community health"], "missing": []},
                "target": {"text": "overall community health improvement", "status": "supported"},
                "answer_hypotheses": ["RESILIENT COMMUNITY"],
            },
            "control_summary": {
                "iteration": 3,
                "branch_weights": {"constraint": 0.4, "relation": 0.35, "anchor": 0.25},
                "preferred_branches": ["constraint", "relation"],
            },
            "evidence_summary": "Supported outcome: community health.",
        }

        response = service.synthesize_answer(
            question=question,
            task_frame=task_frame,
            thought_graph=thought_graph,
            llm_evidence_view=llm_evidence_view,
        )

        self.assertEqual(response["answer"].lower(), "community health")
        self.assertLessEqual(len(response["answer"].split()), 3)


if __name__ == "__main__":
    unittest.main()
