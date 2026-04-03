from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

import numpy as np


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", re.DOTALL)
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "known",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def normalize_label(text: str) -> str:
    cleaned = text.strip()
    for prefix in ("<hyperedge>", "<synonyms>"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    cleaned = cleaned.replace("<SEP>", " / ")
    cleaned = cleaned.strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1]
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def split_source_ids(source_text: str) -> list[str]:
    if not source_text:
        return []
    return [part.strip() for part in source_text.split("<SEP>") if part.strip()]


def slugify(text: str, max_length: int = 64) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    if not normalized:
        normalized = "run"
    return normalized[:max_length].strip("-") or "run"


def short_text(text: str, limit: int = 300) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def extract_json_payload(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty response; expected JSON payload.")

    fence_match = JSON_BLOCK_RE.search(stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()

    for candidate in _json_candidates(stripped):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Unable to parse JSON payload from response: {short_text(text, 240)}")


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        candidates.append(text[first_obj : last_obj + 1])
    first_arr = text.find("[")
    last_arr = text.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        candidates.append(text[first_arr : last_arr + 1])
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_label(text).lower())


def content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]


def lexical_overlap_score(query_texts: list[str], candidate_text: str) -> float:
    candidate_tokens = set(content_tokens(candidate_text))
    if not candidate_tokens:
        return 0.0

    scores: list[float] = []
    for text in query_texts:
        query_tokens = set(content_tokens(text))
        if not query_tokens:
            continue
        overlap = len(query_tokens & candidate_tokens)
        if overlap == 0:
            scores.append(0.0)
            continue
        scores.append(overlap / max(len(query_tokens), 1))
    return max(scores, default=0.0)
