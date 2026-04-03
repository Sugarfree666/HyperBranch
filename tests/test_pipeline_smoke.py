from __future__ import annotations

import unittest
from pathlib import Path

from goth_hyper.config import load_config
from goth_hyper.logging_utils import TraceStore, configure_logging, create_run_dir
from goth_hyper.pipeline import GoTHyperPipeline


class PipelineSmokeTest(unittest.TestCase):
    def test_mock_pipeline_runs_end_to_end(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_config(project_root / "configs" / "agriculture.yaml", project_root)
        config.llm.use_mock = True

        question = "How can urban farms build community support while dealing with lead contamination in soil?"
        run_dir = create_run_dir(config.runtime.base_run_dir, "test smoke run")
        logger = configure_logging(run_dir, config.runtime.log_level)
        trace_store = TraceStore(run_dir)

        pipeline = GoTHyperPipeline(config=config, run_dir=run_dir, logger=logger, trace_store=trace_store)
        result = pipeline.run(question)

        self.assertTrue(result["final_answer"]["answer"])
        self.assertTrue((run_dir / "artifacts" / "final_answer.json").exists())
        self.assertTrue((run_dir / "artifacts" / "evidence_subgraph.json").exists())
        self.assertTrue((run_dir / "run.log").exists())


if __name__ == "__main__":
    unittest.main()
