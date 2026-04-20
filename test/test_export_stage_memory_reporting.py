'''
Author: xuxianghong12
Date: 2026-04-18 00:09:56
LastEditors: xuxianghong12
LastEditTime: 2026-04-18 00:09:56
SPDX-License-Identifier: Apache-2.0
'''
from scripts.export_qwen2_05b import _format_stage_peak
from src.utils.memory_profiler import StagePeakMemory


def test_format_stage_peak_reports_mib_values() -> None:
    record = StagePeakMemory(
        label="load_model",
        elapsed_sec=12.345,
        peak_rss_kb=2048,
        peak_vmsize_kb=4096,
        peak_swap_kb=512,
    )

    formatted = _format_stage_peak(record)

    assert "[export][memory]" in formatted
    assert "load_model" in formatted
    assert "elapsed=12.35s" in formatted
    assert "peak_rss=2.0MiB" in formatted
    assert "peak_vms=4.0MiB" in formatted
    assert "peak_swap=0.5MiB" in formatted