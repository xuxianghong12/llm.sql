"""Memory and page-fault profiling utilities."""

from __future__ import annotations

import os
import resource
from dataclasses import dataclass
from typing import Dict, List


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