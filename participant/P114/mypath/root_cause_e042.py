"""E042：把 E040 的现象标签收成互斥根因。

父桶账本仍然互斥。这里另外做两件事：
1. 同一条损失如果还满足被 if 盖住的条件，单独记重叠，但不加进总分。
2. 开局够不着的步，用 E040 正方形可达结果覆盖一维「硬不可达」标签。

分数换算沿用 e039_bucket_refine 的配对公式：每条 score_loss 是 1/3，
除以 6 得到相对满分 1000 的分。不是 evaluate_one，不改 entry.py。
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

EVENTS = Path("/Users/lqz/Desktop/robocup_changsha/outputs/P114/e039-bucket-refine/events.csv")
SQUARE = Path("/Users/lqz/Desktop/robocup_changsha/outputs/P114/reachability-refine-e040/samples.csv")
BUCKET = Path("/Users/lqz/Desktop/robocup_changsha/outputs/P114/e039-bucket-refine/summary.json")
OUT = Path("/Users/lqz/Desktop/robocup_changsha/outputs/P114/root-cause-e042")

# 事件表里的 score_loss 之和，换成配对分要除以 6：
# 每条 1/3，再 /horizon=10，再 500*(basic+coop) 对 300 个种子取平均。
POINT_DIVISOR = 6.0

MOTION = {
    "TARGET_ASSISTED_REACHABLE",
    "TARGET_ESCAPED",
    "TARGET_ESCAPE",
    "TARGET_MOVED_AWAY",
}
GEOMETRY = {
    "HARD_UNREACHABLE",
    "MARGINAL_UNREACHABLE",
    "RESOURCE_SHORTAGE",
    "CONTROL_LIMITED",
    "ACTION_SATURATION",
    "HORIZON_END",
}
TIMING = {
    "TARGET_SWITCH",
    "LATE_START",
    "ONE_STEP_LATE",
    "INSUFFICIENT_PROGRESS",
    "WRONG_HEADING",
    "BAD_INITIAL_ASSIGNMENT",
    "OBSERVATION_DELAY",
}
ALLOCATION = {
    "AVOIDABLE_ALLOCATION_CONFLICT",
    "MULTI_TARGET_OVERLAP",
    "GEOMETRIC_CONFLICT",
    "CROWD",
}
SQUARE_GEOMETRY = {"hard_unreachable", "marginal_unreachable"}
SQUARE_MOTION = {"target_assisted_reachable"}
SQUARE_TIMING = {"frozen_2d_already_reachable"}


def _points(share: float) -> float:
    return share / POINT_DIVISOR


def _f(row: dict[str, str], key: str) -> float | None:
    text = row.get(key, "")
    if text == "":
        return None
    return float(text)


def _late(age: float | None, step: int) -> bool:
    if age is None:
        return False
    return step >= 3 and age <= max(1, step // 3)


def _root(cause: str, square: str | None, earlier: set[str], target_away: float) -> str:
    if square in SQUARE_GEOMETRY:
        return "geometry"
    if square in SQUARE_MOTION:
        return "target_motion"
    if square in SQUARE_TIMING:
        return "decision_timing"
    if cause in GEOMETRY:
        return "geometry"
    if cause in MOTION:
        return "target_motion"
    if cause in ALLOCATION:
        return "allocation"
    motion_linked = target_away > 0.5 or bool(earlier & MOTION)
    if cause in TIMING and motion_linked:
        return "target_motion"
    if cause in TIMING:
        return "decision_timing"
    return "unassigned"


def _round2(values: list[float]) -> float:
    return float(sum(round(value, 2) for value in values))


def main() -> None:
    with EVENTS.open(encoding="utf-8") as handle:
        events = list(csv.DictReader(handle))
    with SQUARE.open(encoding="utf-8") as handle:
        squares = list(csv.DictReader(handle))
    bucket = json.loads(BUCKET.read_text(encoding="utf-8"))

    square_of = {
        (row["seed"], row["group"], row["step"], row["target"]): row["class_name"] for row in squares
    }
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in events:
        groups[(row["seed"], row["layout"], row["target_id"])].append(row)
    for rows in groups.values():
        rows.sort(key=lambda item: int(item["step"]))

    by_cause: dict[str, float] = defaultdict(float)
    by_root: dict[str, float] = defaultdict(float)
    root_members: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    square_cross: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    unjoined_unreachable = 0.0
    switch_also_late = 0.0
    switch_also_away = 0.0
    switch_late_and_away = 0.0
    one_after_timing = 0.0
    one_after_motion = 0.0
    one_plain = 0.0
    timing_moved_because_away = 0.0
    timing_moved_because_earlier = 0.0

    for key, rows in groups.items():
        seen: set[str] = set()
        for row in rows:
            loss = float(row["score_loss"])
            cause = row["cause"]
            by_cause[cause] += loss
            square = None
            if row["coarse"] == "unreachable":
                square = square_of.get((row["seed"], row["layout"], row["step"], row["target_id"]))
                if square is None:
                    unjoined_unreachable += loss
                else:
                    square_cross[cause][square] += loss
            away = _f(row, "target_away") or 0.0
            if cause == "TARGET_SWITCH":
                late = _late(_f(row, "pursuit_age"), int(row["step"]))
                if late and away > 0.5:
                    switch_late_and_away += loss
                elif late:
                    switch_also_late += loss
                elif away > 0.5:
                    switch_also_away += loss
            if cause == "ONE_STEP_LATE":
                if seen & {"TARGET_SWITCH", "LATE_START"}:
                    one_after_timing += loss
                elif seen & MOTION:
                    one_after_motion += loss
                else:
                    one_plain += loss
            naive_timing = cause in TIMING
            root = _root(cause, square, seen, away)
            if naive_timing and root == "target_motion":
                if away > 0.5:
                    timing_moved_because_away += loss
                else:
                    timing_moved_because_earlier += loss
            by_root[root] += loss
            root_members[root][cause] += loss
            seen.add(cause)

    square_direct: dict[str, float] = defaultdict(float)
    for row in squares:
        square_direct[row["class_name"]] += 1.0 / 18.0

    collision = float(bucket["coarse_points"]["collision"])
    lost = float(bucket["lost"])
    display_parts = [
        float(bucket["panels"]["1_dynamic_reachability"]["causes"]["HARD_UNREACHABLE"]),
        float(bucket["refined_total"]["reachability_error"]),
        float(bucket["coarse_points"]["near"]),
        float(bucket["coarse_points"]["overlap"]),
        float(bucket["coarse_points"]["travel"]),
        float(bucket["coarse_points"]["crowd"]),
        collision,
    ]
    event_points = _points(sum(by_cause.values()))
    roots = {name: _points(by_root[name]) for name in ("geometry", "target_motion", "decision_timing", "allocation")}
    roots["allocation"] += collision

    summary = {
        "note": "脚本自算，不是 evaluate_one。父桶互斥所以账能平；细标签是现象。根因按正方形可达和同一目标的更早现象重贴，每条损失仍只进一个根因。",
        "exact_parent_sum": sum(display_parts),
        "display_rounded_sum": _round2(display_parts),
        "lost": lost,
        "event_points_without_collision": event_points,
        "by_cause_points": {name: _points(value) for name, value in sorted(by_cause.items())},
        "switch_points": _points(by_cause["TARGET_SWITCH"]),
        "switch_also_late_only": _points(switch_also_late),
        "switch_also_away_only": _points(switch_also_away),
        "switch_late_and_away": _points(switch_late_and_away),
        "late_start_points": _points(by_cause["LATE_START"]),
        "one_step_points": _points(by_cause["ONE_STEP_LATE"]),
        "one_step_after_switch_or_late": _points(one_after_timing),
        "one_step_after_earlier_motion": _points(one_after_motion),
        "one_step_without_those": _points(one_plain),
        "timing_relabeled_motion_because_away": _points(timing_moved_because_away),
        "timing_relabeled_motion_because_earlier": _points(timing_moved_because_earlier),
        "square_file_points": dict(square_direct),
        "unreachable_share_missing_square": unjoined_unreachable,
        "square_cross_points": {
            cause: {label: _points(value) for label, value in labels.items()}
            for cause, labels in square_cross.items()
        },
        "exclusive_roots": roots,
        "root_members": {
            root: {cause: _points(value) for cause, value in causes.items()}
            for root, causes in root_members.items()
        },
        "root_sum": sum(roots.values()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
