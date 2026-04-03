from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hyper_branch.cli import _load_question_from_file
from hyper_branch.utils import extract_json_payload, normalize_label


class UtilsTest(unittest.TestCase):
    def test_extract_json_payload_from_fenced_block(self) -> None:
        payload = extract_json_payload('```json\n{"a": 1, "b": [2]}\n```')
        self.assertEqual(payload["a"], 1)
        self.assertEqual(payload["b"], [2])

    def test_normalize_label(self) -> None:
        self.assertEqual(
            normalize_label('<hyperedge>"Urban farms build trust through transparency."'),
            "Urban farms build trust through transparency.",
        )

    def test_load_question_from_json_list_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "query_test.json"
            path.write_text(
                json.dumps(
                    [
                        {"question": "First question?"},
                        {"question": "Second question?"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self.assertEqual(_load_question_from_file(path, 1), "Second question?")


if __name__ == "__main__":
    unittest.main()

