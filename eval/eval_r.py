from __future__ import annotations

import re
import string
from functools import lru_cache


def normalize_answer(answer: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(answer.lower())))


@lru_cache(maxsize=1)
def _load_model():
    from simcse import SimCSE

    return SimCSE("princeton-nlp/sup-simcse-roberta-large")


def calculate_metric_scores_rsim(gold_answers: list[str], predicted_answers: list[str]) -> tuple[dict[str, float], list[dict[str, float]]]:
    assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

    model = _load_model()
    total = 0.0
    example_eval_results: list[dict[str, float]] = []

    for gold, predicted in zip(gold_answers, predicted_answers):
        normalized_gold = normalize_answer(str(gold))
        normalized_predicted = normalize_answer(str(predicted))
        if not normalized_gold or not normalized_predicted:
            score = 0.0
        else:
            similarity = model.similarity([normalized_gold], [normalized_predicted])
            score = float(similarity[0][0])
        total += score
        example_eval_results.append({"R-S": score})

    avg = total / len(gold_answers) if gold_answers else 0.0
    return {"R-S": avg}, example_eval_results


def cal_rsim(gold_answers: list[str], predicted_answers: list[str]) -> float:
    overall_result, _ = calculate_metric_scores_rsim(gold_answers=gold_answers, predicted_answers=predicted_answers)
    return overall_result["R-S"]
