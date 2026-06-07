from __future__ import annotations

import re


class ProgressStream:
    """Convert console/tqdm output into concise progress messages."""

    def __init__(self, callback) -> None:
        self.callback = callback
        self.buffer = ""
        self.last_message = ""

    def write(self, text: str) -> int:
        self.buffer += text
        parts = re.split(r"[\r\n]+", self.buffer)
        self.buffer = parts.pop()
        for part in parts:
            self._emit(part)
        if "%" in self.buffer:
            self._emit(self.buffer)
            self.buffer = ""
        return len(text)

    def flush(self) -> None:
        if self.buffer.strip():
            self._emit(self.buffer)
            self.buffer = ""

    def _emit(self, text: str) -> None:
        message = " ".join(text.strip().split())
        if not message or message == self.last_message:
            return
        self.last_message = message
        self.callback(message)
