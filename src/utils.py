from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True)
class PerformanceStats:
    fps: float = 0.0
    inference_ms: float = 0.0
    frame_count: int = 0
    started_at: float = 0.0
    _last_frame_at: float = 0.0

    def start(self) -> None:
        now = time.monotonic()
        self.started_at = now
        self._last_frame_at = now

    def mark_frame(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_frame_at
        self.frame_count += 1
        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            self.fps = instant_fps if self.fps == 0 else (self.fps * 0.85 + instant_fps * 0.15)
        self._last_frame_at = now

    @property
    def uptime(self) -> float:
        if self.started_at == 0:
            return 0.0
        return time.monotonic() - self.started_at

