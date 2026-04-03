from __future__ import annotations

import json
import os
import socket
import ssl
import time
from typing import Any
from urllib import error, request

import numpy as np

from ..config import LLMConfig
from ..logging_utils import TraceStore
from ..utils import extract_json_payload


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig, trace_store: TraceStore | None = None) -> None:
        self.config = config
        self.trace_store = trace_store
        self.api_key = os.getenv(config.api_key_env, "").strip()
        self.base_url = os.getenv(config.base_url_env, "https://api.openai.com/v1").rstrip("/")
        self.embedding_cache: dict[str, np.ndarray] = {}
        self.response_cache: dict[str, Any] = {}
        if not self.api_key:
            raise RuntimeError(
                f"Environment variable {config.api_key_env} is required for online LLM calls."
            )

    def chat_json(self, stage: str, system_prompt: str, user_payload: dict[str, Any], max_tokens: int = 1400) -> dict[str, Any]:
        response_text = self.chat_text(stage=stage, system_prompt=system_prompt, user_payload=user_payload, max_tokens=max_tokens)
        parsed = extract_json_payload(response_text)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object from stage '{stage}', got {type(parsed).__name__}.")
        return parsed

    def chat_text(
        self,
        stage: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        max_tokens: int = 1400,
        temperature: float | None = None,
    ) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
            ],
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        response = self._post_json("/chat/completions", payload)
        content = response["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        if self.trace_store is not None:
            self.trace_store.log_llm_call(stage, payload, {"content": content})
        return str(content)

    def embed_texts(self, texts: list[str], stage: str) -> list[np.ndarray]:
        unique_texts = [text for text in texts if text not in self.embedding_cache]
        if unique_texts:
            payload = {
                "model": self.config.embedding_model,
                "input": unique_texts,
            }
            response = self._post_json("/embeddings", payload)
            vectors = sorted(response["data"], key=lambda item: item["index"])
            for text, item in zip(unique_texts, vectors, strict=True):
                self.embedding_cache[text] = np.asarray(item["embedding"], dtype=np.float32)
            if self.trace_store is not None:
                self.trace_store.log_llm_call(
                    stage,
                    {"model": self.config.embedding_model, "count": len(unique_texts)},
                    {"count": len(vectors)},
                )
        return [self.embedding_cache[text] for text in texts]

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        raw_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        attempts = max(1, self.config.max_retries + 1)

        for attempt in range(1, attempts + 1):
            req = request.Request(url=url, data=raw_payload, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    body = resp.read().decode("utf-8")
                return json.loads(body)
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if self._should_retry_http(exc.code) and attempt < attempts:
                    self._before_retry(endpoint, attempt, attempts, f"HTTP {exc.code}")
                    continue
                raise RuntimeError(
                    f"LLM request failed with HTTP {exc.code} after {attempt} attempt(s): {body}"
                ) from exc
            except (error.URLError, TimeoutError, socket.timeout, ssl.SSLError) as exc:
                reason = exc.reason if isinstance(exc, error.URLError) else exc
                if self._should_retry_transport(reason) and attempt < attempts:
                    self._before_retry(endpoint, attempt, attempts, reason)
                    continue
                raise RuntimeError(
                    f"LLM request failed after {attempt} attempt(s): {reason}"
                ) from exc

        raise RuntimeError(f"LLM request failed after {attempts} attempt(s): exhausted retries.")

    def _before_retry(self, endpoint: str, attempt: int, attempts: int, reason: object) -> None:
        if self.trace_store is not None:
            self.trace_store.log_event(
                "llm_request_retry",
                {
                    "endpoint": endpoint,
                    "attempt": attempt,
                    "max_attempts": attempts,
                    "reason": str(reason),
                },
            )
        delay_seconds = self.config.retry_backoff_seconds * (2 ** (attempt - 1))
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    def _should_retry_http(self, status_code: int) -> bool:
        return status_code in {408, 409, 425, 429, 500, 502, 503, 504}

    def _should_retry_transport(self, reason: object) -> bool:
        if isinstance(reason, (TimeoutError, socket.timeout, ssl.SSLError, ConnectionResetError)):
            return True
        text = str(reason).lower()
        transient_markers = (
            "timed out",
            "timeout",
            "temporarily unavailable",
            "temporary failure",
            "connection reset",
            "connection aborted",
            "connection refused",
            "handshake operation timed out",
            "ssl",
        )
        return any(marker in text for marker in transient_markers)


class LocalHashEmbeddingClient:
    def __init__(self, dimension: int = 1536) -> None:
        self.dimension = dimension
        self.cache: dict[str, np.ndarray] = {}

    def embed_texts(self, texts: list[str], stage: str) -> list[np.ndarray]:
        del stage
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = [token for token in text.lower().split() if token]
        if not tokens:
            self.cache[text] = vector
            return vector
        for token in tokens:
            slot = hash(token) % self.dimension
            vector[slot] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        self.cache[text] = vector
        return vector
