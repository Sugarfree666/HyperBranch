from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..models import VectorMatch


class VectorStore:
    def __init__(self, name: str, rows: list[dict[str, Any]], matrix: np.ndarray, label_fields: tuple[str, ...]) -> None:
        self.name = name
        self.rows = rows
        self.label_fields = label_fields
        self.row_ids = [str(row.get("__id__", index)) for index, row in enumerate(rows)]
        self.id_to_index = {row_id: index for index, row_id in enumerate(self.row_ids)}
        self.matrix = self._normalize_matrix(matrix.astype(np.float32))

    @classmethod
    def from_json(
        cls,
        path: Path,
        name: str,
        label_fields: tuple[str, ...],
    ) -> "VectorStore":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = list(payload.get("data", []))
        dim = int(payload.get("embedding_dim", 0))
        matrix_payload = payload.get("matrix")
        matrix = cls._decode_matrix(matrix_payload, len(rows), dim)
        return cls(name=name, rows=rows, matrix=matrix, label_fields=label_fields)

    @staticmethod
    def _decode_matrix(matrix_payload: Any, row_count: int, dim: int) -> np.ndarray:
        if isinstance(matrix_payload, str):
            raw = base64.b64decode(matrix_payload)
            matrix = np.frombuffer(raw, dtype="<f4")
            expected = row_count * dim
            if matrix.size != expected:
                raise ValueError(f"Decoded matrix has {matrix.size} values; expected {expected}.")
            return matrix.reshape(row_count, dim)
        if isinstance(matrix_payload, list):
            matrix = np.asarray(matrix_payload, dtype=np.float32)
            if matrix.ndim != 2:
                raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}.")
            return matrix
        raise TypeError(f"Unsupported matrix payload type: {type(matrix_payload).__name__}")

    @staticmethod
    def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def query(
        self,
        query_vector: np.ndarray,
        top_k: int,
        allowed_ids: set[str] | None = None,
    ) -> list[VectorMatch]:
        if top_k <= 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query = query / query_norm
        scores = self.matrix @ query

        if allowed_ids is not None:
            allowed_indices = np.array(
                [self.id_to_index[row_id] for row_id in allowed_ids if row_id in self.id_to_index],
                dtype=np.int32,
            )
            if allowed_indices.size == 0:
                return []
            filtered_scores = scores[allowed_indices]
            order = np.argsort(filtered_scores)[::-1][:top_k]
            indices = allowed_indices[order]
        else:
            indices = np.argsort(scores)[::-1][:top_k]

        matches: list[VectorMatch] = []
        for index in indices:
            row = self.rows[int(index)]
            row_id = self.row_ids[int(index)]
            matches.append(
                VectorMatch(
                    item_id=row_id,
                    label=self._label_for_row(row, row_id),
                    score=float(scores[int(index)]),
                    metadata=row,
                )
            )
        return matches

    def similarity(self, query_vector: np.ndarray, row_id: str) -> float:
        index = self.id_to_index.get(row_id)
        if index is None:
            return 0.0
        query = np.asarray(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return 0.0
        query = query / query_norm
        return float(np.dot(self.matrix[index], query))

    def _label_for_row(self, row: dict[str, Any], fallback: str) -> str:
        for field in self.label_fields:
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                return value
        return fallback
