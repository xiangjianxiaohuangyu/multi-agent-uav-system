"""经验库统计与 CSV 导出。

实现：
- 数量、平均 PDR / Delay / Score
- 参数分布（6 字段各自打 4 桶）
- 场景分布（关键 4 字段：speed / energy / neighbor_count / traffic_load）
- CSV 导出（两段：SUMMARY + EXPERIENCES）
"""

from __future__ import annotations

import csv
import io
import logging
from collections import Counter
from typing import Any

from experience.repository import ExperienceRepository
from experience.scoring import PARAM_FIELDS, SCENE_FIELD_ORDER

log = logging.getLogger(__name__)


# 关键场景字段（避免响应过大）
SCENE_KEY_FIELDS: tuple[str, ...] = (
    "speed",
    "energy",
    "neighbor_count",
    "traffic_load",
)


def _bucket(value: float, edges: list[float]) -> str:
    """把数值分到 ``[low, mid, high, very_high]`` 桶。"""
    if value is None:
        return "unknown"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if len(edges) < 3:
        return f"{v:.3f}"
    # edges 是 3 个分位数（q1, q2, q3）
    if v < edges[0]:
        return "low"
    if v < edges[1]:
        return "mid"
    if v < edges[2]:
        return "high"
    return "very_high"


def _quartiles(values: list[float]) -> list[float]:
    """简单分位数（不含插值）。空列表返回 [0, 0, 0]。"""
    if not values:
        return [0.0, 0.0, 0.0]
    s = sorted(values)
    n = len(s)
    q1 = s[n // 4]
    q2 = s[n // 2]
    q3 = s[(3 * n) // 4]
    return [float(q1), float(q2), float(q3)]


def _safe_mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


# ---------------------------------------------------------------------- #
# 统计
# ---------------------------------------------------------------------- #


def compute_statistics(
    repo: ExperienceRepository,
    page_size: int = 1000,
) -> dict[str, Any]:
    """聚合整个经验库的统计信息（分页扫描以避免一次性加载全表）。"""
    all_rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        chunk = repo.list_paginated(offset=offset, limit=page_size)
        if not chunk:
            break
        all_rows.extend(chunk)
        if len(chunk) < page_size:
            break
        offset += page_size

    count = len(all_rows)
    pdr_vals = [float(r.get("result", {}).get("e2e_pdr", 0.0)) for r in all_rows]
    delay_vals = [float(r.get("result", {}).get("e2e_delay", 0.0)) for r in all_rows]
    score_vals = [float(r.get("score", 0.0)) for r in all_rows]

    # 桶分位数（基于当前数据）
    pdr_edges = _quartiles(pdr_vals) if pdr_vals else [0.0, 0.0, 0.0]
    delay_edges = _quartiles(delay_vals) if delay_vals else [0.0, 0.0, 0.0]
    score_edges = _quartiles(score_vals) if score_vals else [0.0, 0.0, 0.0]

    # 参数分布
    param_dist: dict[str, dict[str, int]] = {}
    for fld in PARAM_FIELDS:
        vals = []
        for r in all_rows:
            p = r.get("parameter", {})
            v = p.get(fld)
            if v is not None:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        edges = _quartiles(vals) if vals else [0.0, 0.0, 0.0]
        counter = Counter(_bucket(v, edges) for v in vals)
        param_dist[fld] = dict(counter)

    # 场景分布（仅关键字段）
    scene_dist: dict[str, dict[str, int]] = {}
    for fld in SCENE_KEY_FIELDS:
        vals = []
        for r in all_rows:
            s = r.get("scene", {})
            v = s.get(fld)
            if v is not None:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        edges = _quartiles(vals) if vals else [0.0, 0.0, 0.0]
        counter = Counter(_bucket(v, edges) for v in vals)
        scene_dist[fld] = dict(counter)

    return {
        "count": count,
        "avg_pdr": _safe_mean(pdr_vals),
        "avg_delay": _safe_mean(delay_vals),
        "avg_score": _safe_mean(score_vals),
        "quartiles": {
            "pdr": pdr_edges,
            "delay": delay_edges,
            "score": score_edges,
        },
        "parameter_distribution": param_dist,
        "scene_distribution": scene_dist,
    }


# ---------------------------------------------------------------------- #
# CSV
# ---------------------------------------------------------------------- #


def to_csv(
    stats: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    """生成两段 CSV 文本：SUMMARY + EXPERIENCES。"""
    buf = io.StringIO()
    writer = csv.writer(buf)

    # ---- SUMMARY ----
    writer.writerow(["# SECTION", "SUMMARY"])
    writer.writerow(["key", "value"])
    writer.writerow(["count", stats.get("count", 0)])
    writer.writerow(["avg_pdr", stats.get("avg_pdr", "")])
    writer.writerow(["avg_delay", stats.get("avg_delay", "")])
    writer.writerow(["avg_score", stats.get("avg_score", "")])

    q = stats.get("quartiles", {})
    writer.writerow(["pdr_q1_q2_q3", q.get("pdr", [])])
    writer.writerow(["delay_q1_q2_q3", q.get("delay", [])])
    writer.writerow(["score_q1_q2_q3", q.get("score", [])])

    pd = stats.get("parameter_distribution", {})
    for fld, buckets in pd.items():
        writer.writerow([f"param_dist.{fld}", str(buckets)])

    sd = stats.get("scene_distribution", {})
    for fld, buckets in sd.items():
        writer.writerow([f"scene_dist.{fld}", str(buckets)])

    # ---- EXPERIENCES ----
    writer.writerow([])
    writer.writerow(["# SECTION", "EXPERIENCES"])
    writer.writerow(
        [
            "experience_id",
            "score",
            "created_time",
            "scene.speed",
            "scene.energy",
            "scene.queue_length",
            "scene.neighbor_count",
            "scene.distance_to_destination",
            "scene.forward_candidate_ratio",
            "scene.avg_neighbor_distance",
            "scene.relative_speed_mean",
            "scene.link_stability",
            "scene.link_lifetime_mean",
            "scene.traffic_load",
            "parameter.hello_interval",
            "parameter.candidate_num",
            "parameter.w_distance",
            "parameter.w_linktime",
            "parameter.w_energy",
            "parameter.w_queue",
            "result.e2e_pdr",
            "result.e2e_delay",
            "result.routing_overhead",
            "result.energy_consumption",
        ]
    )
    for r in rows:
        scene = r.get("scene", {})
        param = r.get("parameter", {})
        res = r.get("result", {})
        writer.writerow(
            [
                r.get("experience_id", ""),
                r.get("score", ""),
                r.get("created_time", ""),
                *[scene.get(k, "") for k in SCENE_FIELD_ORDER],
                *[param.get(k, "") for k in PARAM_FIELDS],
                res.get("e2e_pdr", ""),
                res.get("e2e_delay", ""),
                res.get("routing_overhead", ""),
                res.get("energy_consumption", ""),
            ]
        )

    return buf.getvalue()
