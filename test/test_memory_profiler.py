'''
Author: xuxianghong12
Date: 2026-04-18 00:08:18
LastEditors: xuxianghong12
LastEditTime: 2026-04-18 00:08:19
SPDX-License-Identifier: Apache-2.0
'''
import time

from src.utils.memory_profiler import StagePeakMemoryTracker


def test_stage_peak_memory_tracker_records_stage_summary() -> None:
    tracker = StagePeakMemoryTracker(sample_interval_sec=0.005)

    with tracker.track("alloc_buffer"):
        payload = bytearray(4 * 1024 * 1024)
        payload[0] = 1
        time.sleep(0.03)

    rows = tracker.to_rows()

    assert len(rows) == 1
    assert rows[0]["label"] == "alloc_buffer"
    assert rows[0]["peak_rss_kb"] >= rows[0]["start_rss_kb"]
    assert rows[0]["peak_vmsize_kb"] >= rows[0]["start_vmsize_kb"]
    assert "alloc_buffer" in tracker.summary()