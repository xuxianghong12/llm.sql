"""Memory and page-fault profiling utilities."""

from __future__ import annotations

import os
import resource
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List


_IS_LINUX = os.path.isfile("/proc/self/status")


@dataclass
class MemorySnapshot:
    label: str
    vm_peak_kb: int = 0
    vm_rss_kb: int = 0
    vm_swap_kb: int = 0
    ru_maxrss_kb: int = 0
    ru_minflt: int = 0
    ru_majflt: int = 0
    ru_inblock: int = 0
    ru_oublock: int = 0
    ru_nvcsw: int = 0
    ru_nivcsw: int = 0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    io_read_syscalls: int = 0
    io_write_syscalls: int = 0


def _read_proc_status() -> Dict[str, int]:
    result: Dict[str, int] = {}
    if not _IS_LINUX:
        return result
    try:
        with open("/proc/self/status") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                tokens = parts[1].strip().split()
                if not tokens:
                    continue
                try:
                    result[key] = int(tokens[0])
                except ValueError:
                    pass
    except OSError:
        pass
    return result


def _read_proc_io() -> Dict[str, int]:
    result: Dict[str, int] = {}
    if not _IS_LINUX:
        return result
    try:
        with open("/proc/self/io") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                try:
                    result[key] = int(parts[1].strip())
                except ValueError:
                    pass
    except (OSError, PermissionError):
        pass
    return result


def take_snapshot(label: str = "") -> MemorySnapshot:
    status = _read_proc_status()
    io_info = _read_proc_io()
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return MemorySnapshot(
        label=label,
        vm_peak_kb=status.get("VmPeak", 0),
        vm_rss_kb=status.get("VmRSS", 0),
        vm_swap_kb=status.get("VmSwap", 0),
        ru_maxrss_kb=ru.ru_maxrss,
        ru_minflt=ru.ru_minflt,
        ru_majflt=ru.ru_majflt,
        ru_inblock=ru.ru_inblock,
        ru_oublock=ru.ru_oublock,
        ru_nvcsw=ru.ru_nvcsw,
        ru_nivcsw=ru.ru_nivcsw,
        io_read_bytes=io_info.get("rchar", 0),
        io_write_bytes=io_info.get("wchar", 0),
        io_read_syscalls=io_info.get("syscr", 0),
        io_write_syscalls=io_info.get("syscw", 0),
    )


@dataclass
class MemoryDiff:
    from_label: str
    to_label: str
    delta_rss_kb: int = 0
    delta_swap_kb: int = 0
    delta_minflt: int = 0
    delta_majflt: int = 0
    delta_inblock: int = 0
    delta_oublock: int = 0
    delta_nvcsw: int = 0
    delta_nivcsw: int = 0
    delta_io_read_bytes: int = 0
    delta_io_write_bytes: int = 0


def diff_snapshots(a: MemorySnapshot, b: MemorySnapshot) -> MemoryDiff:
    return MemoryDiff(
        from_label=a.label,
        to_label=b.label,
        delta_rss_kb=b.vm_rss_kb - a.vm_rss_kb,
        delta_swap_kb=b.vm_swap_kb - a.vm_swap_kb,
        delta_minflt=b.ru_minflt - a.ru_minflt,
        delta_majflt=b.ru_majflt - a.ru_majflt,
        delta_inblock=b.ru_inblock - a.ru_inblock,
        delta_oublock=b.ru_oublock - a.ru_oublock,
        delta_nvcsw=b.ru_nvcsw - a.ru_nvcsw,
        delta_nivcsw=b.ru_nivcsw - a.ru_nivcsw,
        delta_io_read_bytes=b.io_read_bytes - a.io_read_bytes,
        delta_io_write_bytes=b.io_write_bytes - a.io_write_bytes,
    )


class MemoryTracker:
    """Collects named snapshots and produces a simple summary."""

    def __init__(self) -> None:
        self._snapshots: List[MemorySnapshot] = []

    def snapshot(self, label: str = "") -> MemorySnapshot:
        snap = take_snapshot(label)
        self._snapshots.append(snap)
        return snap

    @property
    def snapshots(self) -> List[MemorySnapshot]:
        return list(self._snapshots)

    def diffs(self) -> List[MemoryDiff]:
        return [
            diff_snapshots(self._snapshots[i], self._snapshots[i + 1])
            for i in range(len(self._snapshots) - 1)
        ]

    def summary(self) -> str:
        lines = [
            "Memory Profile Summary",
            "=" * 80,
            f"  {'label':<25s} {'RSS_kB':>10s} {'Swap_kB':>10s} {'MinFlt':>10s} {'MajFlt':>10s}",
            "-" * 80,
        ]
        for snapshot in self._snapshots:
            lines.append(
                f"  {snapshot.label:<25s} {snapshot.vm_rss_kb:>10d} {snapshot.vm_swap_kb:>10d} "
                f"{snapshot.ru_minflt:>10d} {snapshot.ru_majflt:>10d}"
            )

        if len(self._snapshots) >= 2:
            lines.append("")
            lines.append("Deltas (consecutive)")
            lines.append("-" * 80)
            lines.append(
                f"  {'from → to':<25s} {'ΔRSS_kB':>10s} {'ΔSwap_kB':>10s} {'ΔMinFlt':>10s} {'ΔMajFlt':>10s}"
            )
            lines.append("-" * 80)
            for diff in self.diffs():
                tag = f"{diff.from_label}→{diff.to_label}"
                lines.append(
                    f"  {tag:<25s} {diff.delta_rss_kb:>10d} {diff.delta_swap_kb:>10d} "
                    f"{diff.delta_minflt:>10d} {diff.delta_majflt:>10d}"
                )
        lines.append("=" * 80)
        return "\n".join(lines)

    def to_rows(self) -> List[Dict[str, object]]:
        return [
            {
                "label": snapshot.label,
                "vm_rss_kb": snapshot.vm_rss_kb,
                "vm_peak_kb": snapshot.vm_peak_kb,
                "vm_swap_kb": snapshot.vm_swap_kb,
                "ru_maxrss_kb": snapshot.ru_maxrss_kb,
                "ru_minflt": snapshot.ru_minflt,
                "ru_majflt": snapshot.ru_majflt,
                "ru_inblock": snapshot.ru_inblock,
                "ru_oublock": snapshot.ru_oublock,
                "ru_nvcsw": snapshot.ru_nvcsw,
                "ru_nivcsw": snapshot.ru_nivcsw,
                "io_read_bytes": snapshot.io_read_bytes,
                "io_write_bytes": snapshot.io_write_bytes,
            }
            for snapshot in self._snapshots
        ]

    def reset(self) -> None:
        self._snapshots.clear()


@dataclass
class StagePeakMemory:
    label: str
    elapsed_sec: float = 0.0
    start_rss_kb: int = 0
    end_rss_kb: int = 0
    peak_rss_kb: int = 0
    start_vmsize_kb: int = 0
    end_vmsize_kb: int = 0
    peak_vmsize_kb: int = 0
    start_swap_kb: int = 0
    end_swap_kb: int = 0
    peak_swap_kb: int = 0
    sample_count: int = 0


def _read_proc_memory_kb() -> Dict[str, int]:
    status = _read_proc_status()
    return {
        "rss_kb": status.get("VmRSS", 0),
        "vmsize_kb": status.get("VmSize", 0),
        "swap_kb": status.get("VmSwap", 0),
    }


def _format_kb_as_mib(value_kb: int) -> str:
    return f"{value_kb / 1024.0:.1f}"


class _StagePeakSampler:
    def __init__(self, label: str, sample_interval_sec: float) -> None:
        self._label = label
        self._sample_interval_sec = sample_interval_sec
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_metrics = _read_proc_memory_kb()
        self._peak_metrics = dict(self._start_metrics)
        self._sample_count = 1
        self._start_time = 0.0

    def start(self) -> None:
        self._start_time = time.perf_counter()
        if _IS_LINUX and self._sample_interval_sec > 0:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _observe(self, metrics: Dict[str, int]) -> None:
        self._sample_count += 1
        self._peak_metrics["rss_kb"] = max(
            self._peak_metrics["rss_kb"], metrics["rss_kb"]
        )
        self._peak_metrics["vmsize_kb"] = max(
            self._peak_metrics["vmsize_kb"], metrics["vmsize_kb"]
        )
        self._peak_metrics["swap_kb"] = max(
            self._peak_metrics["swap_kb"], metrics["swap_kb"]
        )

    def _run(self) -> None:
        while not self._stop_event.wait(self._sample_interval_sec):
            self._observe(_read_proc_memory_kb())

    def finish(self) -> StagePeakMemory:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

        end_metrics = _read_proc_memory_kb()
        self._observe(end_metrics)
        return StagePeakMemory(
            label=self._label,
            elapsed_sec=time.perf_counter() - self._start_time,
            start_rss_kb=self._start_metrics["rss_kb"],
            end_rss_kb=end_metrics["rss_kb"],
            peak_rss_kb=self._peak_metrics["rss_kb"],
            start_vmsize_kb=self._start_metrics["vmsize_kb"],
            end_vmsize_kb=end_metrics["vmsize_kb"],
            peak_vmsize_kb=self._peak_metrics["vmsize_kb"],
            start_swap_kb=self._start_metrics["swap_kb"],
            end_swap_kb=end_metrics["swap_kb"],
            peak_swap_kb=self._peak_metrics["swap_kb"],
            sample_count=self._sample_count,
        )


class StagePeakMemoryTracker:
    """Track peak process memory for named pipeline stages."""

    def __init__(self, sample_interval_sec: float = 0.05) -> None:
        self._sample_interval_sec = sample_interval_sec
        self._records: List[StagePeakMemory] = []

    @contextmanager
    def track(self, label: str) -> Iterator[None]:
        sampler = _StagePeakSampler(label, self._sample_interval_sec)
        sampler.start()
        try:
            yield
        finally:
            self._records.append(sampler.finish())

    @property
    def records(self) -> List[StagePeakMemory]:
        return list(self._records)

    def to_rows(self) -> List[Dict[str, object]]:
        return [
            {
                "label": record.label,
                "elapsed_sec": record.elapsed_sec,
                "start_rss_kb": record.start_rss_kb,
                "end_rss_kb": record.end_rss_kb,
                "peak_rss_kb": record.peak_rss_kb,
                "start_vmsize_kb": record.start_vmsize_kb,
                "end_vmsize_kb": record.end_vmsize_kb,
                "peak_vmsize_kb": record.peak_vmsize_kb,
                "start_swap_kb": record.start_swap_kb,
                "end_swap_kb": record.end_swap_kb,
                "peak_swap_kb": record.peak_swap_kb,
                "sample_count": record.sample_count,
            }
            for record in self._records
        ]

    def summary(self) -> str:
        lines = [
            "Stage Peak Memory Summary",
            "=" * 110,
            (
                f"  {'label':<20s} {'elapsed_s':>10s} {'start_rss_mib':>14s} "
                f"{'peak_rss_mib':>13s} {'end_rss_mib':>12s} {'peak_vms_mib':>13s} {'peak_swap_mib':>14s}"
            ),
            "-" * 110,
        ]
        for record in self._records:
            lines.append(
                f"  {record.label:<20s} {record.elapsed_sec:>10.2f} "
                f"{_format_kb_as_mib(record.start_rss_kb):>14s} "
                f"{_format_kb_as_mib(record.peak_rss_kb):>13s} "
                f"{_format_kb_as_mib(record.end_rss_kb):>12s} "
                f"{_format_kb_as_mib(record.peak_vmsize_kb):>13s} "
                f"{_format_kb_as_mib(record.peak_swap_kb):>14s}"
            )
        lines.append("=" * 110)
        return "\n".join(lines)

    def reset(self) -> None:
        self._records.clear()