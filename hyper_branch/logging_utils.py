from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .utils import pretty_json, slugify


class TraceStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._event_path = run_dir / "events.jsonl"
        self._llm_path = run_dir / "llm_calls.jsonl"
        self._lock = Lock()

    def log_event(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "payload": payload,
        }
        self._append_jsonl(self._event_path, record)

    def log_llm_call(self, stage: str, request_payload: dict[str, Any], response_payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage": stage,
            "request": request_payload,
            "response": response_payload,
        }
        self._append_jsonl(self._llm_path, record)

    def save_artifact(self, relative_path: str, payload: Any) -> Path:
        target = self.run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, str):
            target.write_text(payload, encoding="utf-8")
        else:
            target.write_text(pretty_json(payload), encoding="utf-8")
        return target

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_run_dir(base_dir: Path, question: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{timestamp}_{slugify(question, 48)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def configure_logging(run_dir: Path, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("hyper_branch")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger
