from __future__ import annotations

from pathlib import Path


class PromptManager:
    def __init__(self, prompt_dir: Path) -> None:
        self.prompt_dir = prompt_dir
        self._cache: dict[str, str] = {}

    def get(self, name: str) -> str:
        if name not in self._cache:
            path = self.prompt_dir / f"{name}.md"
            self._cache[name] = path.read_text(encoding="utf-8")
        return self._cache[name]
