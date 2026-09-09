from __future__ import annotations

import json
import base64
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hyper_branch.database import HypergraphDatabase, _lookup_key


class HypergraphDatabaseTest(unittest.TestCase):
    def test_lookup_key_normalizes_accents_and_apostrophe_variants(self) -> None:
        self.assertEqual(_lookup_key("Beyonc\u00e9"), _lookup_key("Beyonce"))
        self.assertEqual(_lookup_key("O\u2019Connor"), _lookup_key("O'Connor"))
        self.assertEqual(_lookup_key("J. R. R. Tolkien"), _lookup_key("J R R Tolkien"))

    def test_missing_entity_name_vdb_does_not_fall_back_to_entity_description_vdb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "graph_chunk_entity_relation.graphml").write_text("<graphml />", encoding="utf-8")
            (root / "kv_store_text_chunks.json").write_text(json.dumps({}), encoding="utf-8")
            (root / "kv_store_full_docs.json").write_text(json.dumps({}), encoding="utf-8")
            (root / "vdb_entities.json").write_text(json.dumps({}), encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "vdb_entity_names.json"):
                HypergraphDatabase(root)

    def test_source_chunk_entities_extend_two_hops_and_candidates_are_ranked_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "graph_chunk_entity_relation.graphml").write_text(
                GRAPHML,
                encoding="utf-8",
            )
            (root / "kv_store_text_chunks.json").write_text(
                json.dumps(
                    {
                        "C1": {"content": "A and B occur in the same source chunk."},
                        "C2": {"content": "B connects to the second hyperedge."},
                    }
                ),
                encoding="utf-8",
            )
            _write_vectors(
                root / "vdb_entity_names.json",
                "entity_name",
                [("A", [1.0, 0.0]), ("B", [0.0, 1.0])],
            )
            _write_vectors(
                root / "vdb_hyperedges.json",
                "hyperedge_name",
                [("H1", [1.0, 0.0]), ("H2", [0.0, 1.0]), ("H3", [0.4, 0.9165])],
            )
            _write_vectors(
                root / "vdb_chunks.json",
                "__id__",
                [("C1", [1.0, 0.0]), ("C2", [0.0, 1.0])],
            )

            database = HypergraphDatabase(root)
            candidates = database.candidate_pool(["A"], FixedEmbedder([0.0, 1.0]))
            vector_linked_candidates = database.candidate_pool(
                ["alias for A"],
                FixedEmbedder([1.0, 0.0]),
            )
            ranked = database.rank(
                "Which relation is relevant?",
                candidates,
                FixedEmbedder([0.0, 1.0]),
            )
            path_ranked = database.rank(
                "Which path is relevant?",
                {"H2": {"H1", "H3"}, "H3": set()},
                FixedEmbedder([1.0, 0.0]),
            )

        self.assertEqual(candidates, {"H1": set(), "H2": {"H1"}})
        self.assertEqual(vector_linked_candidates, candidates)
        self.assertNotIn("Generic", database.chunk_to_entities["C1"])
        self.assertEqual(database.link_entity_ids(["A."], FailingEmbedder()), {"A.": ["A"]})
        self.assertEqual(database._link_entities("A.", FailingEmbedder()), ["A"])
        self.assertEqual(database._link_entities("A (film)", FailingEmbedder()), ["A"])
        database._entity_ids_by_name[_lookup_key("A")].append("A alias")
        self.assertEqual(database._link_entities("A.", FailingEmbedder()), ["A", "A alias"])
        database._entity_ids_by_name[_lookup_key("A")].append("B")
        self.assertEqual(database._link_entities("A (film)", FixedEmbedder([0.0, 1.0])), ["B"])
        self.assertEqual([item["id"] for item in ranked], ["H2", "H1"])
        self.assertEqual(
            ranked[0]["chunks"],
            [
                ("C1", "A and B occur in the same source chunk."),
                ("C2", "B connects to the second hyperedge."),
            ],
        )
        self.assertEqual(ranked[0]["first_hop_texts"], ["H1"])
        self.assertNotIn("H3", [item["id"] for item in ranked])
        self.assertEqual([item["id"] for item in path_ranked], ["H2", "H3"])
        self.assertEqual(path_ranked[0]["first_hop_texts"], ["H1"])
        self.assertEqual(
            path_ranked[0]["chunks"],
            [
                ("C1", "A and B occur in the same source chunk."),
                ("C2", "B connects to the second hyperedge."),
            ],
        )


class FixedEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        return self.vector


class FailingEmbedder:
    def embed_text(self, text: str) -> np.ndarray:
        raise AssertionError("punctuation-insensitive exact matching should not embed")


def _write_vectors(
    path: Path,
    label_field: str,
    rows: list[tuple[str, list[float]]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "embedding_dim": 2,
                "matrix": base64.b64encode(
                    np.asarray([vector for _label, vector in rows], dtype="<f4").tobytes()
                ).decode("ascii"),
                "data": [
                    {"__id__": label, label_field: label}
                    for label, _vector in rows
                ],
            }
        ),
        encoding="utf-8",
    )


GRAPHML = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="role" for="all" attr.name="role" attr.type="string"/>
  <key id="source" for="node" attr.name="source_id" attr.type="string"/>
  <key id="type" for="node" attr.name="entity_type" attr.type="string"/>
  <graph id="G" edgedefault="undirected">
    <node id="A"><data key="role">entity</data><data key="source">C1</data></node>
    <node id="B"><data key="role">entity</data><data key="source">C1</data></node>
    <node id="Generic"><data key="role">entity</data><data key="source">C1</data><data key="type">CONCEPT</data></node>
    <node id="H1"><data key="role">hyperedge</data><data key="source">C1</data></node>
    <node id="H2"><data key="role">hyperedge</data><data key="source">C2</data></node>
    <node id="H3"><data key="role">hyperedge</data><data key="source">C2</data></node>
    <edge source="A" target="H1"><data key="role">link</data></edge>
    <edge source="B" target="H2"><data key="role">link</data></edge>
    <edge source="Generic" target="H3"><data key="role">link</data></edge>
  </graph>
</graphml>
"""


if __name__ == "__main__":
    unittest.main()
