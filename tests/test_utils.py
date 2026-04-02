from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from goth_hyper.cli import _load_question_from_file
from goth_hyper.utils import extract_json_payload, load_dotenv, normalize_label


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

    def test_load_dotenv_reads_key_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / ".env"
            path.write_text(
                '\n'.join(
                    [
                        '# comment',
                        'OPENAI_API_KEY="test-key"',
                        'OPENAI_BASE_URL=https://api.openai.com/v1',
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                loaded = load_dotenv(path)
                self.assertEqual(loaded["OPENAI_API_KEY"], "test-key")
                self.assertEqual(os.environ["OPENAI_API_KEY"], "test-key")
                self.assertEqual(os.environ["OPENAI_BASE_URL"], "https://api.openai.com/v1")


if __name__ == "__main__":
    unittest.main()
