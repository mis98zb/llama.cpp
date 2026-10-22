#!/usr/bin/env python3
"""Capture a layered prefill with Nsight Systems and analyze actual GPU activities."""

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Activity:
    start: int
    end: int
    device: int
    context: int
    stream: int
    kind: str
    name: str
    size: int = 0


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Activity ends before it starts")
        if end == start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def duration(intervals):
    return sum(end - start for start, end in intervals)


def intersection(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def gaps(intervals, start, end):
    result = []
    cursor = start
    for left, right in intervals:
        if left > cursor:
            result.append((cursor, left))
        cursor = right
    if cursor < end:
        result.append((cursor, end))
    return result


def summarize(activities, start, end):
    if end <= start:
        raise ValueError("Invalid prefill window")
    activities = [event for event in activities if event.start < end and event.end > start]
    devices = []
    for device in sorted({event.device for event in activities}):
        selected = [event for event in activities if event.device == device]
        groups = {}
        for kind in ("kernel", "h2d", "d2h", "d2d", "other_copy", "memset"):
            groups[kind] = union(
                (max(start, event.start), min(end, event.end))
                for event in selected
                if event.kind == kind and event.start < end and event.end > start
            )
        compute = groups["kernel"]
        h2d_overlap = intersection(compute, groups["h2d"])
        d2h_overlap = intersection(compute, groups["d2h"])
        host_transfers = union(groups["h2d"] + groups["d2h"])
        host_overlap = intersection(compute, host_transfers)
        busy = union(interval for group in groups.values() for interval in group)
        h2d_ns = duration(groups["h2d"])
        compute_ns = duration(compute)
        d2h_ns = duration(groups["d2h"])
        row = {
            "device_id": device,
            "window_ns": end - start,
            "kernel_active_ns": compute_ns,
            "h2d_active_ns": h2d_ns,
            "d2h_active_ns": d2h_ns,
            "d2d_active_ns": duration(groups["d2d"]),
            "memset_active_ns": duration(groups["memset"]),
            "h2d_kernel_overlap_ns": duration(h2d_overlap),
            "d2h_kernel_overlap_ns": duration(d2h_overlap),
            "h2d_without_kernel_ns": h2d_ns - duration(h2d_overlap),
            "h2d_overlap_ratio": duration(h2d_overlap) / h2d_ns if h2d_ns else None,
            "d2h_overlap_ratio": duration(d2h_overlap) / d2h_ns if d2h_ns else None,
            "host_transfer_active_ns": duration(host_transfers),
            "host_transfer_kernel_overlap_ns": duration(host_overlap),
            "host_transfer_overlap_ratio": duration(host_overlap) / duration(host_transfers) if host_transfers else None,
            "kernel_h2d_overlap_ratio": duration(h2d_overlap) / compute_ns if compute_ns else None,
            "traced_gpu_active_ns": duration(busy),
            "no_traced_gpu_activity_ns": end - start - duration(busy),
        }
        for kind in ("h2d", "d2h", "d2d", "other_copy"):
            copies = [event for event in selected if event.kind == kind]
            row[f"{kind}_bytes"] = sum(event.size for event in copies)
            row[f"{kind}_count"] = len(copies)
        devices.append(row)
    return devices


def columns(db, table):
    return {row["name"] for row in db.execute(f'PRAGMA table_info("{table}")')}


def nvtx_events(db, tables):
    table = next((name for name in ("NVTX_EVENTS", "NvtxEvents") if name in tables), None)
    if not table:
        raise ValueError("Missing NVTX data; capture with --trace=cuda,nvtx")
    fields = columns(db, table)
    if not {"start", "end", "globalTid"} <= fields:
        raise ValueError("Unsupported NvtxEvents schema")
    expressions = []
    join = ""
    if "text" in fields:
        expressions.append("NULLIF(n.text, '')")
    if "textId" in fields and "StringIds" in tables:
        join = "LEFT JOIN StringIds s ON s.id = n.textId"
        expressions.append("s.value")
    if not expressions:
        raise ValueError("Cannot resolve NVTX names")
    name = expressions[0] if len(expressions) == 1 else f"COALESCE({', '.join(expressions)})"
    return [
        dict(row)
        for row in db.execute(
            f'SELECT n.start, n.end, n.globalTid, {name} AS name FROM "{table}" n {join}'
        )
        if row["name"] and row["name"].startswith("llama.")
    ]


def load_capture(path, range_index=None):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        labels = nvtx_events(db, tables)
        windows = sorted(
            (row for row in labels if row["name"] == "llama.prefill" and row["end"] is not None),
            key=lambda row: row["start"],
        )
        if not windows:
            raise ValueError("No complete llama.prefill range; use --profile-gpu")
        if range_index is None and len(windows) != 1:
            raise ValueError("Multiple prefill ranges; select --range-index explicitly")
        index = 0 if range_index is None else range_index
        if not 0 <= index < len(windows):
            raise ValueError("Prefill range index out of bounds")
        window = windows[index]
        start, end, thread = window["start"], window["end"], window["globalTid"]
        if end <= start or thread is None:
            raise ValueError("Invalid prefill NVTX timestamps or owner")
        markers = {
            row["name"]
            for row in labels
            if row["globalTid"] == thread and start <= row["start"] <= end
        }
        if "llama.prefill.success" not in markers or "llama.prefill.failed" in markers:
            raise ValueError("Prefill did not report successful completion")
        owner = thread & ~0xFFFFFF
        window["device_info"] = {}
        if {"TARGET_INFO_CUDA_DEVICE", "TARGET_INFO_GPU"} <= tables:
            fields = columns(db, "TARGET_INFO_CUDA_DEVICE")
            if {"cudaId", "gpuId", "pid"} <= fields:
                query = """
                    SELECT c.cudaId, c.gpuId, g.name
                    FROM TARGET_INFO_CUDA_DEVICE c LEFT JOIN TARGET_INFO_GPU g ON g.id = c.gpuId
                    WHERE c.pid = ?
                """
                for row in db.execute(query, ((owner >> 24) & 0xFFFFFF,)):
                    window["device_info"][row["cudaId"]] = {
                        "device_name": row["name"], "nsight_gpu_id": row["gpuId"]
                    }
        events = []
        for table, kind in (
            ("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
            ("CUPTI_ACTIVITY_KIND_MEMCPY", "copy"),
            ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"),
        ):
            if table not in tables:
                continue
            fields = columns(db, table)
            required = {"start", "end", "deviceId", "contextId", "streamId", "globalPid"}
            if not required <= fields:
                raise ValueError(f"Unsupported {table} schema: missing {sorted(required - fields)}")
            if db.execute(
                f'SELECT 1 FROM "{table}" WHERE start < ? AND end > ? AND globalPid IS NULL LIMIT 1',
                (end, start),
            ).fetchone():
                raise ValueError(f"Unattributed GPU activity in {table}; process-scoped measurement is incomplete")
            names = {}
            if kind == "kernel":
                name_column = next((key for key in ("demangledName", "shortName", "nameId") if key in fields), None)
                if name_column and "StringIds" in tables:
                    names = {
                        row[0]: row[1]
                        for row in db.execute(
                            f'SELECT DISTINCT s.id, s.value FROM "{table}" e JOIN StringIds s ON s.id = e."{name_column}"'
                        )
                    }
            elif kind == "copy":
                if not {"copyKind", "bytes"} <= fields or "ENUM_CUDA_MEMCPY_OPER" not in tables:
                    raise ValueError("Cannot identify GPU copy directions and sizes")
                names = dict(db.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER"))
            query = f'SELECT * FROM "{table}" WHERE start < ? AND end > ? AND globalPid = ? LIMIT 2000001'
            for number, row in enumerate(db.execute(query, (end, start, owner))):
                if number >= 2_000_000:
                    raise ValueError("Capture exceeds two million activities; profile a shorter request")
                if any(not isinstance(row[key], int) for key in required):
                    raise ValueError(f"Invalid GPU activity fields in {table}")
                if row["end"] < row["start"]:
                    raise ValueError("Invalid GPU activity timestamps")
                category = kind
                name = kind
                size = 0
                if kind == "kernel":
                    name = names.get(row[name_column], "CUDA kernel") if name_column else "CUDA kernel"
                elif kind == "copy":
                    if "copyCount" in fields and row["copyCount"] not in (None, 0, 1):
                        raise ValueError("Batched memcpy envelope is not an individual GPU activity interval")
                    if row["copyKind"] not in names:
                        raise ValueError(f"Unknown CUDA memcpy kind {row['copyKind']}")
                    name = names[row["copyKind"]]
                    direction = re.sub("[^a-z]", "", name.lower())
                    if direction in ("htoh", "hosttohost", "unknown"):
                        raise ValueError("Host-only or unknown memcpy is not a measured GPU activity")
                    category = {"htod": "h2d", "hosttodevice": "h2d", "htoa": "h2d", "hosttoarray": "h2d",
                                "dtoh": "d2h", "devicetohost": "d2h", "atoh": "d2h", "arraytohost": "d2h",
                                "dtod": "d2d", "devicetodevice": "d2d", "atoa": "d2d", "arraytoarray": "d2d",
                                "atod": "d2d", "arraytodevice": "d2d", "dtoa": "d2d", "devicetoarray": "d2d"}.get(
                        direction, "other_copy"
                    )
                    size = row["bytes"]
                    if not isinstance(size, int) or size < 0:
                        raise ValueError("Negative memcpy byte count")
                    if size and row["start"] == row["end"]:
                        raise ValueError("Nonempty copy has no measured GPU duration")
                    if row["start"] < start or row["end"] > end:
                        raise ValueError("Copy crosses the prefill window; cannot attribute its bytes exactly")
                events.append(Activity(row["start"], row["end"], row["deviceId"], row["contextId"],
                                       row["streamId"], category, name, size))
        if not any(event.kind == "kernel" and event.end > event.start for event in events):
            raise ValueError("No kernel-level GPU activities for this process; use --cuda-graph-trace=node")
        for table in ("DIAGNOSTIC_EVENT", "DiagnosticMessages"):
            if table not in tables:
                continue
            fields = columns(db, table)
            text_field = next((key for key in ("message", "text") if key in fields), None)
            if text_field:
                for row in db.execute(f'SELECT "{text_field}" FROM "{table}"'):
                    message = str(row[0]).lower()
                    if ("cuda" in message or "cupti" in message) and ("dropped" in message or "lost" in message):
                        raise ValueError(f"Incomplete CUDA capture: {row[0]}")
        selected_labels = [
            row for row in labels
            if row["globalTid"] is not None and (row["globalTid"] & ~0xFFFFFF) == owner
            and start <= row["start"] <= end
        ]
        return window, events, selected_labels


def chrome_trace(window, events, labels):
    origin = window["start"]
    trace = [{"name": "process_name", "ph": "M", "pid": 0, "tid": 0, "args": {"name": "Host NVTX"}}]
    lanes = {}
    devices = {device: number + 1 for number, device in enumerate(sorted({event.device for event in events}))}
    for device, pid in devices.items():
        name = window.get("device_info", {}).get(device, {}).get("device_name") or "GPU"
        trace.append({"name": "process_name", "ph": "M", "pid": pid, "tid": 0,
                      "args": {"name": f"{name} (CUDA device {device})"}})
        selected = [event for event in events if event.device == device]
        compute = union((max(origin, event.start), min(window["end"], event.end))
                        for event in selected if event.kind == "kernel")
        copies = union((event.start, event.end) for event in selected if event.kind == "h2d")
        busy = union((max(origin, event.start), min(window["end"], event.end)) for event in selected)
        aggregates = (
            ("Kernel active (union)", compute),
            ("H2D active (union)", copies),
            ("H2D + kernel overlap", intersection(compute, copies)),
            ("No traced GPU activity", gaps(busy, origin, window["end"])),
        )
        for tid, (label, intervals) in enumerate(aggregates, 1):
            trace.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid, "args": {"name": label}})
            for start, end in intervals:
                trace.append({"name": label, "cat": "aggregate", "ph": "X", "pid": pid, "tid": tid,
                              "ts": (start - origin) / 1000, "dur": (end - start) / 1000})
    for event in events:
        key = (event.device, event.context, event.stream)
        if key not in lanes:
            lanes[key] = len(lanes) + 100
            trace.append({"name": "thread_name", "ph": "M", "pid": devices[event.device], "tid": lanes[key],
                          "args": {"name": f"Context {event.context}, stream {event.stream}"}})
        start = max(origin, event.start)
        end = min(window["end"], event.end)
        trace.append({"name": event.name, "cat": event.kind, "ph": "X", "pid": devices[event.device],
                      "tid": lanes[key], "ts": (start - origin) / 1000, "dur": (end - start) / 1000,
                      "args": {"bytes": event.size, "start_ns": event.start, "end_ns": event.end}})
    for label in labels:
        entry = {"name": label["name"], "cat": "host.nvtx", "pid": 0,
                 "tid": label["globalTid"] & 0xFFFFFF, "ts": (label["start"] - origin) / 1000}
        if label["end"] is not None and label["end"] > label["start"]:
            entry.update(ph="X", dur=(min(window["end"], label["end"]) - label["start"]) / 1000)
        else:
            entry.update(ph="i", s="t")
        trace.append(entry)
    return {"traceEvents": trace, "displayTimeUnit": "ms"}


def analyze(path, output, range_index=None):
    if not path.is_file():
        raise ValueError(f"Missing SQLite capture: {path}")
    window, events, labels = load_capture(path, range_index)
    summary = {
        "source": str(path),
        "measurement": "Nsight GPU activity intervals, not CPU/stream envelopes",
        "scope": "successful llama.prefill NVTX range, owner process, per device",
        "note": "Concurrent activity coverage is not a counterfactual critical-path hiding percentage.",
        "busy_activity_kinds": ["kernel", "memcpy", "memset"],
        "owner_pid": (window["globalTid"] >> 24) & 0xFFFFFF,
        "start_ns": window["start"], "end_ns": window["end"],
        "devices": summarize(events, window["start"], window["end"]),
    }
    for row in summary["devices"]:
        row.update(window.get("device_info", {}).get(row["device_id"], {}))
    outputs = (Path(str(output) + ".summary.json"), Path(str(output) + ".trace.json"))
    if any(target.exists() for target in outputs):
        raise ValueError("Refusing to overwrite existing timeline outputs")
    for target, data in zip(outputs, (summary, chrome_trace(window, events, labels))):
        with target.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=True, indent=2 if target == outputs[0] else None)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--nsys", default="nsys")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("command", nargs=argparse.REMAINDER)
    offline = commands.add_parser("analyze")
    offline.add_argument("sqlite", type=Path)
    offline.add_argument("--output", type=Path, required=True)
    offline.add_argument("--range-index", type=int)
    args = parser.parse_args()
    try:
        if args.action == "capture":
            executable = shutil.which(str(args.nsys))
            if not executable:
                raise ValueError("Nsight Systems CLI not found; supply --nsys with its full path")
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            if not command:
                raise ValueError("Supply the llama-layered-prefill command after --")
            if "--profile-gpu" not in command:
                command.append("--profile-gpu")
            if any(Path(str(args.output) + suffix).exists() for suffix in
                   (".nsys-rep", ".sqlite", ".summary.json", ".trace.json", ".capture.log")):
                raise ValueError("Refusing to overwrite an existing capture")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            invocation = [executable, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                          "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
                          "--capture-range-end=stop", "--export=sqlite", "--force-overwrite=false",
                          "--show-output=true",
                          "--output", str(args.output), *command]
            with Path(str(args.output) + ".capture.log").open("x", encoding="utf-8") as log:
                result = subprocess.run(invocation, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                log.write(result.stdout)
            print(result.stdout, end="")
            if result.returncode:
                raise ValueError(f"Nsight capture failed with exit code {result.returncode}")
            if re.search(r"(?im)(?:cuda|cupti)[^\n]*(?:dropped|lost)", result.stdout):
                raise ValueError("Nsight reported CUDA data loss; capture is incomplete")
            summary = analyze(Path(str(args.output) + ".sqlite"), args.output)
        else:
            summary = analyze(args.sqlite, args.output, args.range_index)
        print(json.dumps(summary, indent=2))
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"GPU timeline: {error}\n")


if __name__ == "__main__":
    main()
