#!/usr/bin/env python3
import json
import re
import sys
from bisect import bisect_right
from pathlib import Path

TASKS = ["task_2a", "task_2b", "task_3"]
TASK_LABELS = {
    "task_2a": "Task 2A",
    "task_2b": "Task 2B",
    "task_3": "Task 3",
}
TRACE_FILE_RE = re.compile(r"trace_(\d+)\.json$")
TASK_3_ITERATION_END_NAME = (
    "autograd::engine::evaluate_function: torch::autograd::AccumulateGrad"
)
TASK_3_ITERATION_START_TOKEN = "foreach_norm"


def load_trace(json_path: Path) -> dict:
    with json_path.open("r") as f:
        return json.load(f)


def sum_gloo_comm_ms(trace: dict) -> float:
    events = trace.get("traceEvents", [])
    total_dur_us = 0.0

    for ev in events:
        if ev.get("ph") != "X":
            continue
        if ev.get("cat") != "user_annotation":
            continue
        if not str(ev.get("name", "")).startswith("gloo:"):
            continue

        dur = ev.get("dur")
        if isinstance(dur, (int, float)):
            total_dur_us += dur

    return total_dur_us / 1e3


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []

    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return [(start, end) for start, end in merged]


def interval_length_ms(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals) / 1e3


def subtract_intervals(
    source: list[tuple[float, float]], subtract: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    if not source:
        return []
    if not subtract:
        return source

    result = []
    j = 0
    for start, end in source:
        curr = start
        while j < len(subtract) and subtract[j][1] <= start:
            j += 1

        k = j
        while k < len(subtract) and subtract[k][0] < end:
            sub_start, sub_end = subtract[k]
            if sub_start > curr:
                result.append((curr, min(sub_start, end)))
            curr = max(curr, sub_end)
            if curr >= end:
                break
            k += 1

        if curr < end:
            result.append((curr, end))

    return result


def task_3_iteration_gap_comm_ms(trace: dict) -> float:
    start_events: list[tuple[float, int | None]] = []
    end_events_by_tid: dict[int | None, list[float]] = {}

    for ev in trace.get("traceEvents", []):
        ts = ev.get("ts")
        dur = ev.get("dur")
        if ev.get("ph") != "X" or not isinstance(ts, (int, float)) or not isinstance(
            dur, (int, float)
        ):
            continue

        name = str(ev.get("name", ""))
        tid = ev.get("tid") if isinstance(ev.get("tid"), int) else None
        if TASK_3_ITERATION_START_TOKEN in name:
            start_events.append((ts, tid))
        elif name == TASK_3_ITERATION_END_NAME:
            end_events_by_tid.setdefault(tid, []).append(ts + dur)

    total_gap_us = 0.0
    for start_ts, tid in start_events:
        end_times = end_events_by_tid.get(tid, [])
        prev_end_idx = bisect_right(end_times, start_ts) - 1
        if prev_end_idx >= 0:
            total_gap_us += max(0.0, start_ts - end_times[prev_end_idx])

    return total_gap_us / 1e3


def total_profiled_ms(trace: dict) -> float:
    events = trace.get("traceEvents", [])
    timestamps = [ev["ts"] for ev in events if isinstance(ev.get("ts"), (int, float))]
    if not timestamps:
        return 0.0

    start_ts = min(timestamps)
    end_ts = max(timestamps)
    for ev in events:
        ts = ev.get("ts")
        dur = ev.get("dur")
        if isinstance(ts, (int, float)) and isinstance(dur, (int, float)):
            end_ts = max(end_ts, ts + dur)

    return (end_ts - start_ts) / 1e3


def extract_rank(trace_path: Path) -> int:
    match = TRACE_FILE_RE.search(trace_path.name)
    if not match:
        raise ValueError(f"Could not extract rank from file name: {trace_path}")
    return int(match.group(1))


def collect_metrics(data_dir: Path) -> dict[str, dict[int, dict[str, float]]]:
    metrics: dict[str, dict[int, dict[str, float]]] = {}

    for task in TASKS:
        task_dir = data_dir / task
        metrics[task] = {}
        for trace_path in sorted(task_dir.glob("trace_*.json")):
            trace = load_trace(trace_path)
            rank = extract_rank(trace_path)
            if task == "task_3":
                communication_ms = task_3_iteration_gap_comm_ms(trace)
            else:
                communication_ms = sum_gloo_comm_ms(trace)
            total_ms = total_profiled_ms(trace)
            percentage = (communication_ms / total_ms * 100.0) if total_ms else 0.0
            metrics[task][rank] = {
                "communication": communication_ms,
                "total": total_ms,
                "percentage": percentage,
            }

    return metrics


def format_metric(value: float, metric_name: str) -> str:
    if metric_name == "percentage":
        return f"{value:.2f}\\%"
    return f"{value / 1e3:.3f}"


def latex_table(metrics: dict[str, dict[int, dict[str, float]]]) -> str:
    ranks = sorted({rank for task_metrics in metrics.values() for rank in task_metrics})
    metric_specs = [
        ("communication", "Communication"),
        ("total", "Total"),
        ("percentage", "Percentage"),
    ]
    task_order = [task for task in TASKS if task in metrics]

    lines = [
        "\\begin{table}[h]",
        "  \\centering",
        "  \\caption{Per-rank communication time, total profiled time, and communication percentage across tasks. Times are reported in seconds.}",
        "  \\label{tab:trace_communication}",
        f"  \\begin{{tabular}}{{c{'c' * (len(metric_specs) * len(task_order))}}}",
        "    \\toprule",
        "    \\multirow{2}{*}{Rank} & "
        + " & ".join(
            f"\\multicolumn{{{len(task_order)}}}{{c}}{{{label}}}"
            for _, label in metric_specs
        )
        + " \\\\",
        "    \\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}",
        "    & "
        + " & ".join(TASK_LABELS[task] for _metric, _label in metric_specs for task in task_order)
        + " \\\\",
        "    \\midrule",
    ]

    for rank in ranks:
        cells = []
        for metric_name, _label in metric_specs:
            for task in task_order:
                value = metrics.get(task, {}).get(rank, {}).get(metric_name)
                cells.append("--" if value is None else format_metric(value, metric_name))
        lines.append(f"    {rank} & " + " & ".join(cells) + " \\\\")

    avg_cells = []
    for metric_name, _label in metric_specs:
        for task in task_order:
            values = [
                metrics.get(task, {}).get(rank, {}).get(metric_name)
                for rank in ranks
                if metrics.get(task, {}).get(rank, {}).get(metric_name) is not None
            ]
            avg_value = sum(values) / len(values) if values else None
            avg_cells.append("--" if avg_value is None else format_metric(avg_value, metric_name))
    lines.append("    \\midrule")
    lines.append("    Average & " + " & ".join(avg_cells) + " \\\\")

    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def print_single_trace_metrics(trace_path: Path) -> None:
    trace = load_trace(trace_path)
    if trace_path.parent.name == "task_3":
        communication_ms = task_3_iteration_gap_comm_ms(trace)
    else:
        communication_ms = sum_gloo_comm_ms(trace)
    total_ms = total_profiled_ms(trace)
    percentage = (communication_ms / total_ms * 100.0) if total_ms else 0.0

    print(f"trace = {trace_path}")
    print(f"communication_s = {communication_ms / 1e3:.5f}")
    print(f"total_s = {total_ms / 1e3:.5f}")
    print(f"percentage = {percentage:.6f}%")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./RTE")
    if not target.exists():
        print(f"Path not found: {target}")
        sys.exit(1)

    if target.is_file():
        print_single_trace_metrics(target)
    else:
        metrics = collect_metrics(target)
        print(latex_table(metrics))
