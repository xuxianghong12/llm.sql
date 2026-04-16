"""Phase-level timing utilities for inference profiling."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Dict, Generator, List


@dataclass
class PhaseRecord:
    name: str
    start: float = 0.0
    end: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        return self.end - self.start

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_seconds * 1000.0


class PerfTimer:
    """Accumulating multi-phase timer."""

    def __init__(self) -> None:
        self._records: List[PhaseRecord] = []

    @contextlib.contextmanager
    def phase(self, name: str) -> Generator[PhaseRecord, None, None]:
        rec = PhaseRecord(name=name, start=time.perf_counter())
        self._records.append(rec)
        try:
            yield rec
        finally:
            rec.end = time.perf_counter()

    @property
    def records(self) -> List[PhaseRecord]:
        return list(self._records)

    def total_seconds(self) -> float:
        return sum(record.elapsed_seconds for record in self._records)

    def summary(self) -> str:
        lines = ["Phase Timing Summary", "-" * 50]
        for record in self._records:
            lines.append(f"  {record.name:<30s} {record.elapsed_ms:>10.2f} ms")
        lines.append("-" * 50)
        lines.append(f"  {'TOTAL':<30s} {self.total_seconds() * 1000:>10.2f} ms")
        return "\n".join(lines)

    def to_rows(self) -> List[Dict[str, object]]:
        return [
            {
                "phase": record.name,
                "elapsed_seconds": record.elapsed_seconds,
                "elapsed_ms": record.elapsed_ms,
            }
            for record in self._records
        ]

    def record_duration(self, name: str, seconds: float) -> PhaseRecord:
        rec = PhaseRecord(name=name, start=0.0, end=seconds)
        self._records.append(rec)
        return rec

    def reset(self) -> None:
        self._records.clear()


@contextlib.contextmanager
def phase_timer(name: str = "unnamed") -> Generator[PhaseRecord, None, None]:
    rec = PhaseRecord(name=name, start=time.perf_counter())
    try:
        yield rec
    finally:
        rec.end = time.perf_counter()