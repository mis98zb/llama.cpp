import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "gpu_timeline", Path(__file__).resolve().parents[1] / "scripts" / "gpu_timeline.py"
)
assert SPEC is not None and SPEC.loader is not None
timeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = timeline
SPEC.loader.exec_module(timeline)

OWNER = (1 << 48) | (123 << 24)
THREAD = OWNER | 7


def capture_db(path, nvtx_table="NVTX_EVENTS"):
    db = sqlite3.connect(path)
    db.executescript(f"""
        CREATE TABLE StringIds(id INTEGER, value TEXT);
        INSERT INTO StringIds VALUES (1, 'kernel'), (2, 'llama.prefill'), (3, 'llama.prefill.success');
        CREATE TABLE "{nvtx_table}"(start INTEGER, end INTEGER, text TEXT, textId INTEGER, globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
            start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER,
            streamId INTEGER, globalPid INTEGER, demangledName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
            start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER,
            streamId INTEGER, globalPid INTEGER, copyKind INTEGER, bytes INTEGER);
        CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER, label TEXT);
        INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (1, 'HtoD'), (2, 'DtoH'), (8, 'DtoD');
    """)
    db.execute(f'INSERT INTO "{nvtx_table}" VALUES(0,100,NULL,2,?)', (THREAD,))
    db.execute(f'INSERT INTO "{nvtx_table}" VALUES(99,NULL,NULL,3,?)', (THREAD,))
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(10,40,0,1,1,?,1)", (OWNER,))
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(0,20,0,1,2,?,1,100)", (OWNER,))
    db.commit()
    return db


def test_union_and_intersection():
    assert timeline.union([(20, 30), (0, 10), (5, 20), (5, 6), (40, 40)]) == [(0, 30)]
    assert timeline.intersection([(0, 10), (20, 30)], [(5, 25)]) == [(5, 10), (20, 25)]
    assert timeline.gaps([(5, 10), (20, 30)], 0, 40) == [(0, 5), (10, 20), (30, 40)]
    with pytest.raises(ValueError, match="ends before"):
        timeline.union([(10, 9)])


def test_concurrent_streams_are_not_double_counted():
    events = [
        timeline.Activity(10, 30, 0, 1, 1, "kernel", "a"),
        timeline.Activity(20, 40, 0, 1, 2, "kernel", "b"),
        timeline.Activity(0, 20, 0, 1, 3, "h2d", "copy", 100),
        timeline.Activity(10, 30, 0, 1, 4, "h2d", "copy", 200),
        timeline.Activity(50, 60, 0, 1, 4, "d2h", "copy", 100),
        timeline.Activity(70, 80, 0, 1, 4, "d2d", "copy", 100),
        timeline.Activity(80, 90, 0, 1, 4, "memset", "clear"),
    ]
    row = timeline.summarize(events, 0, 100)[0]
    assert row["kernel_active_ns"] == 30
    assert row["h2d_active_ns"] == 30
    assert row["h2d_kernel_overlap_ns"] == 20
    assert row["h2d_without_kernel_ns"] == 10
    assert row["h2d_overlap_ratio"] == pytest.approx(2 / 3)
    assert row["no_traced_gpu_activity_ns"] == 30
    assert row["h2d_bytes"] == 300
    assert row["h2d_count"] == 2
    assert row["host_transfer_active_ns"] == 40
    assert row["host_transfer_overlap_ratio"] == 0.5


def test_devices_do_not_overlap_each_other():
    events = [
        timeline.Activity(0, 10, 0, 1, 1, "kernel", "a"),
        timeline.Activity(0, 10, 1, 1, 1, "h2d", "copy", 100),
    ]
    rows = timeline.summarize(events, 0, 10)
    assert all(row["h2d_kernel_overlap_ns"] == 0 for row in rows)
    assert rows[0]["h2d_overlap_ratio"] is None
    assert rows[1]["h2d_overlap_ratio"] == 0
    assert rows[1]["kernel_h2d_overlap_ratio"] is None


@pytest.mark.parametrize("table", ["NVTX_EVENTS", "NvtxEvents"])
def test_sqlite_scope_and_export(tmp_path, table):
    source = tmp_path / "capture.sqlite"
    with capture_db(source, table) as db:
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(0,100,0,1,2,?,1)", (OWNER + (1 << 24),))
    output = tmp_path / "out"
    result = timeline.analyze(source, output)
    row = result["devices"][0]
    assert row["kernel_active_ns"] == 30
    assert row["h2d_kernel_overlap_ns"] == 10
    assert row["h2d_overlap_ratio"] == 0.5
    trace = json.loads((tmp_path / "out.trace.json").read_text())
    assert any(event["name"] == "kernel" and event["cat"] == "kernel" for event in trace["traceEvents"])
    assert any(event["name"] == "llama.prefill" for event in trace["traceEvents"])
    with pytest.raises(ValueError, match="overwrite"):
        timeline.analyze(source, output)


def test_failure_or_missing_completion_is_not_success(tmp_path):
    source = tmp_path / "capture.sqlite"
    with capture_db(source) as db:
        db.execute("DELETE FROM NVTX_EVENTS WHERE textId = 3")
    with pytest.raises(ValueError, match="successful completion"):
        timeline.load_capture(source)
    with sqlite3.connect(source) as db:
        db.execute("INSERT INTO NVTX_EVENTS VALUES(99,NULL,'llama.prefill.failed',NULL,?)", (THREAD,))
    with pytest.raises(ValueError, match="successful completion"):
        timeline.load_capture(source)


def test_cuda_logical_ids_do_not_name_the_wrong_physical_gpu(tmp_path):
    source = tmp_path / "capture.sqlite"
    with capture_db(source) as db:
        db.executescript("""
            CREATE TABLE TARGET_INFO_GPU(id INTEGER, name TEXT);
            INSERT INTO TARGET_INFO_GPU VALUES(0, 'wrong card'), (2, 'T1000');
            CREATE TABLE TARGET_INFO_CUDA_DEVICE(cudaId INTEGER, gpuId INTEGER, pid INTEGER);
            INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES(0, 2, 123);
        """)
    result = timeline.analyze(source, tmp_path / "out")
    assert result["devices"][0]["device_name"] == "T1000"
    assert result["devices"][0]["nsight_gpu_id"] == 2


def test_multiple_ranges_require_selection(tmp_path):
    source = tmp_path / "capture.sqlite"
    with capture_db(source) as db:
        db.execute("INSERT INTO NVTX_EVENTS VALUES(200,300,'llama.prefill',NULL,?)", (THREAD,))
        db.execute("INSERT INTO NVTX_EVENTS VALUES(299,NULL,'llama.prefill.success',NULL,?)", (THREAD,))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(210,220,0,1,1,?,1)", (OWNER,))
    with pytest.raises(ValueError, match="Multiple"):
        timeline.load_capture(source)
    window, events, _ = timeline.load_capture(source, 1)
    assert window["start"] == 200
    assert timeline.summarize(events, 200, 300)[0]["kernel_active_ns"] == 10


def test_partial_copy_and_reported_loss_are_rejected(tmp_path):
    source = tmp_path / "capture.sqlite"
    with capture_db(source) as db:
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET start = -1")
    with pytest.raises(ValueError, match="attribute its bytes"):
        timeline.load_capture(source)
    with sqlite3.connect(source) as db:
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET start = 0")
        db.executescript("CREATE TABLE DIAGNOSTIC_EVENT(text TEXT); INSERT INTO DIAGNOSTIC_EVENT VALUES('CUPTI dropped activity records');")
    with pytest.raises(ValueError, match="Incomplete CUDA capture"):
        timeline.load_capture(source)


def test_no_kernel_or_unknown_copy_kind_is_not_a_valid_measurement(tmp_path):
    source = tmp_path / "capture.sqlite"
    with capture_db(source) as db:
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET copyKind = 99")
    with pytest.raises(ValueError, match="Unknown CUDA memcpy"):
        timeline.load_capture(source)
    with sqlite3.connect(source) as db:
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET copyKind = 1")
        db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")
    with pytest.raises(ValueError, match="No kernel-level"):
        timeline.load_capture(source)


def test_batched_copy_envelopes_are_not_treated_as_dma_busy_time(tmp_path):
    source = tmp_path / "capture.sqlite"
    with capture_db(source) as db:
        db.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_MEMCPY ADD COLUMN copyCount INTEGER")
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET copyCount = 3")
    with pytest.raises(ValueError, match="Batched memcpy envelope"):
        timeline.load_capture(source)


def test_unknown_process_copies_do_not_silently_become_zero_transfer(tmp_path):
    source = tmp_path / "capture.sqlite"
    with capture_db(source) as db:
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET globalPid = NULL")
    with pytest.raises(ValueError, match="Unattributed GPU activity"):
        timeline.load_capture(source)


def test_chrome_timestamps_preserve_small_offsets():
    origin = 1 << 60
    events = [timeline.Activity(origin + 1, origin + 2, 0, 1, 1, "kernel", "a")]
    trace = timeline.chrome_trace({"start": origin, "end": origin + 100}, events, [])
    event = next(event for event in trace["traceEvents"] if event.get("cat") == "kernel")
    assert event["ts"] == 0.001
    assert event["dur"] == 0.001
    assert event["args"]["start_ns"] == origin + 1
    aggregates = [event for event in trace["traceEvents"] if event.get("cat") == "aggregate"]
    assert sum(event["dur"] for event in aggregates if event["name"] == "No traced GPU activity") == pytest.approx(0.099)
