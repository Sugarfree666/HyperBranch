from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib import error

from hyper_branch import cli
from hyper_branch.config import Config, DatasetConfig, LLMConfig, PromptConfig, ReasoningConfig, RetrievalConfig, RuntimeConfig
from hyper_branch.llm.client import OpenAICompatibleClient


class FakeHTTPResponse:
    def __init__(self, body: str) -> None:
        self.body = body

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False

    def read(self) -> bytes:
        return self.body.encode("utf-8")


class LLMClientRetryTest(unittest.TestCase):
    def test_transport_timeout_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = LLMConfig(timeout_seconds=1, max_retries=2, retry_backoff_seconds=0.0)
            client = OpenAICompatibleClient(config=config)

            responses = [
                error.URLError(TimeoutError("The handshake operation timed out")),
                FakeHTTPResponse('{"ok": true}'),
            ]

            with patch("hyper_branch.llm.client.request.urlopen", side_effect=responses) as mocked_urlopen, patch(
                "hyper_branch.llm.client.time.sleep"
            ):
                payload = client._post_json("/chat/completions", {"ping": "pong"})

            self.assertEqual(payload["ok"], True)
            self.assertEqual(mocked_urlopen.call_count, 2)


class CLIAllowFailureTest(unittest.TestCase):
    def test_allow_failure_returns_zero_and_writes_error_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_run_dir = Path(tmp_dir) / "runs"
            config = Config(
                project_root=Path(tmp_dir),
                dataset=DatasetConfig(root=Path(tmp_dir)),
                runtime=RuntimeConfig(base_run_dir=base_run_dir),
                retrieval=RetrievalConfig(),
                reasoning=ReasoningConfig(),
                llm=LLMConfig(use_mock=True),
                prompts=PromptConfig(directory=Path(tmp_dir)),
            )
            fake_pipeline = MagicMock()
            fake_pipeline.run.side_effect = RuntimeError("The handshake operation timed out")

            stdout = io.StringIO()
            with patch("hyper_branch.cli.load_config", return_value=config), patch(
                "hyper_branch.cli.HyperBranchPipeline", return_value=fake_pipeline
            ), patch(
                "sys.argv",
                ["cli.py", "--question", "Test question?", "--allow-failure"],
            ), redirect_stdout(stdout):
                exit_code = cli.main()

            self.assertEqual(exit_code, 0)
            run_dirs = sorted(base_run_dir.iterdir())
            self.assertEqual(len(run_dirs), 1)
            error_artifact = run_dirs[0] / "artifacts" / "error.json"
            self.assertTrue(error_artifact.exists())
            self.assertIn("status=failed", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()

