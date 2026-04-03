from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from openai import APIConnectionError, OpenAI, RateLimitError, Timeout
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent


def _read_optional_text(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


@lru_cache(maxsize=1)
def _build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip() or _read_optional_text(PROJECT_ROOT / "PRoH-main" / "openai_api_key.txt")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for G-E evaluation.")

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or _read_optional_text(PROJECT_ROOT / "PRoH-main" / "openai_base_url.txt")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
def _judge_with_llm(prompt: str) -> str:
    client = _build_client()
    response = client.chat.completions.create(
        model=os.getenv("EVAL_OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return str(response.choices[0].message.content or "")


def cal_gen(question: str, answers: list[str], generation: str, f1_score: float) -> dict[str, Any]:
    if not generation.strip():
        return {"score": 0.0, "explanation": {"status": "empty_generation"}}

    def build_prompt(metric: str) -> str:
        descriptions = {
            "comprehensiveness": (
                "comprehensiveness",
                "whether the response covers the important aspects of the question",
                """Scoring guide (0-10):
- 10: Extremely thorough and complete.
- 8-9: Covers most key aspects with only minor omissions.
- 6-7: Partially complete, but misses some important details.
- 4-5: Limited coverage and shallow treatment.
- 1-3: Barely addresses the question.
- 0: Completely unhelpful.""",
            ),
            "knowledgeability": (
                "knowledgeability",
                "whether the response demonstrates domain-relevant knowledge",
                """Scoring guide (0-10):
- 10: Deep, accurate, and insightful knowledge.
- 8-9: Strong knowledge with minor gaps.
- 6-7: Some relevant knowledge, but limited depth.
- 4-5: Basic or partially flawed understanding.
- 1-3: Weak knowledge and poor insight.
- 0: No meaningful knowledge shown.""",
            ),
            "correctness": (
                "correctness",
                "whether the response is factually and logically correct",
                """Scoring guide (0-10):
- 10: Fully correct and logically sound.
- 8-9: Mostly correct with minor issues.
- 6-7: Partially correct with notable flaws.
- 4-5: Several clear errors.
- 1-3: Mostly incorrect.
- 0: Entirely wrong.""",
            ),
            "relevance": (
                "relevance",
                "whether the response directly addresses the question",
                """Scoring guide (0-10):
- 10: Fully focused and helpful.
- 8-9: Mostly relevant with minor digressions.
- 6-7: Generally relevant but somewhat diffuse.
- 4-5: Limited relevance.
- 1-3: Barely related.
- 0: Irrelevant.""",
            ),
            "logical_coherence": (
                "logical coherence",
                "whether the response is clear and internally consistent",
                """Scoring guide (0-10):
- 10: Very clear, coherent, and easy to follow.
- 8-9: Mostly coherent with small lapses.
- 6-7: Understandable but uneven.
- 4-5: Often disorganized or unclear.
- 1-3: Hard to follow.
- 0: Incoherent.""",
            ),
            "factuality": (
                "factuality",
                "whether the response relies on accurate and verifiable facts",
                """Scoring guide (0-10):
- 10: Accurate and verifiable throughout.
- 8-9: Mostly factual with minor issues.
- 6-7: Mixed factual quality.
- 4-5: Several unsupported or false claims.
- 1-3: Mostly unreliable.
- 0: Fabricated or entirely false.""",
            ),
        }

        title, goal, rubric = descriptions[metric]
        return f"""You are evaluating the {title} of a generated answer.

Question:
{question}

Golden answers:
{answers}

Response to evaluate:
{generation}

Evaluation goal:
Judge {goal}.

{rubric}

Return exactly:
<score>
INTEGER_0_TO_10
</score>
<explanation>
Brief justification.
</explanation>
"""

    def score_metric(metric: str) -> tuple[str, dict[str, Any]]:
        prompt = build_prompt(metric)
        content = _judge_with_llm(prompt)
        try:
            score_str = content.split("<score>")[1].split("</score>")[0].strip()
            explanation = content.split("<explanation>")[1].split("</explanation>")[0].strip()
            raw_score = int(score_str)
        except Exception as exc:
            raw_score = 5
            explanation = f"Failed to parse evaluator output. Defaulted to 5. Error: {exc}"

        normalized_score = max(0.0, min(1.0, raw_score / 10.0))
        blended_score = (normalized_score + f1_score) / 2
        return metric, {"score": blended_score, "explanation": explanation}

    explanations: dict[str, Any] = {}
    metrics = [
        "comprehensiveness",
        "knowledgeability",
        "correctness",
        "relevance",
        "logical_coherence",
        "factuality",
    ]

    for metric in metrics:
        metric_name, result = score_metric(metric)
        explanations[metric_name] = result

    overall_score = round(float(np.mean([explanations[metric]["score"] for metric in metrics])), 4)
    return {"score": overall_score, "explanation": explanations}
