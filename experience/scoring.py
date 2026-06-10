"""经验库的纯函数模块：场景向量构造、Score 计算、payload 字段映射。

本模块是 ``compute_score`` / ``scene_to_vector`` / ``payload_to_experience``
的单一真相源。所有可被单元测试覆盖的逻辑都集中在这里，零 I/O。
"""

from __future__ import annotations

import logging
from typing import Any

from experience.config import ExperienceConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# 固定字段顺序（spec 强制要求）
# ---------------------------------------------------------------------- #

SCENE_FIELD_ORDER: tuple[str, ...] = (
    "speed",
    "energy",
    "queue_length",
    "neighbor_count",
    "distance_to_destination",
    "forward_candidate_ratio",
    "avg_neighbor_distance",
    "relative_speed_mean",
    "link_stability",
    "link_lifetime_mean",
    "traffic_load",
)
SCENE_DIMENSION: int = len(SCENE_FIELD_ORDER)

PARAM_FIELDS: tuple[str, ...] = (
    "hello_interval",
    "candidate_num",
    "w_distance",
    "w_linktime",
    "w_energy",
    "w_queue",
)

RESULT_FIELDS: tuple[str, ...] = (
    "e2e_pdr",
    "e2e_delay",
    "routing_overhead",
    "energy_consumption",
)


# ---------------------------------------------------------------------- #
# 基础工具
# ---------------------------------------------------------------------- #


def _clamp01(x: float) -> float:
    """把 ``x`` 截断到 ``[0, 1]``。"""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _to_float(v: Any, default: float = 0.0) -> float:
    """尽力把任意值转成 ``float``；失败回退默认值。"""
    if v is None or v == "":
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return f


def _to_int(v: Any, default: int = 0) -> int:
    """尽力把任意值转成 ``int``；失败回退默认值。"""
    if v is None or v == "":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default


# ---------------------------------------------------------------------- #
# 公开 API
# ---------------------------------------------------------------------- #


def scene_to_vector(scene: dict[str, Any]) -> list[float]:
    """把场景字典转换为 11 维定长 float 向量。

    严格按 ``SCENE_FIELD_ORDER`` 顺序取值；缺失键 → ``0.0``；非数值 → ``0.0``。
    """
    if not isinstance(scene, dict):
        return [0.0] * SCENE_DIMENSION
    return [_to_float(scene.get(k)) for k in SCENE_FIELD_ORDER]


def compute_score(
    pdr: float,
    delay_ms: float,
    energy: float,
    cfg: ExperienceConfig,
) -> float:
    """按规范计算经验评分。

    score = w_pdr   * clamp(pdr, 0, 1)
          + w_delay * (1 - clamp(delay_ms / max_delay_ms, 0, 1))
          + w_energy* (1 - clamp(energy     / max_energy,   0, 1))
    """
    pdr_n = _clamp01(_to_float(pdr))
    delay_n = 1.0 - _clamp01(_to_float(delay_ms) / cfg.max_delay_ms)
    energy_n = 1.0 - _clamp01(_to_float(energy) / cfg.max_energy)
    w = cfg.score_weights
    return w.pdr * pdr_n + w.delay * delay_n + w.energy * energy_n


def build_score(
    pdr: float,
    delay_ms: float,
    energy: float,
    cfg: ExperienceConfig,
) -> float:
    """``compute_score`` 的别名；API 更直观。"""
    return compute_score(pdr, delay_ms, energy, cfg)


def validate_scene(scene: dict[str, Any]) -> None:
    """校验 ``scene`` 字典。

    Raises:
        ValueError: scene 不是字典或包含非数值的 key（严格模式，用于 API 入参）。
    """
    if not isinstance(scene, dict):
        raise ValueError("scene must be a dict")
    for k in SCENE_FIELD_ORDER:
        v = scene.get(k, 0.0)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"scene.{k} must be numeric, got {type(v).__name__}")


def validate_parameter(parameter: dict[str, Any]) -> None:
    """校验 ``parameter`` 字典。"""
    if not isinstance(parameter, dict):
        raise ValueError("parameter must be a dict")
    for k in PARAM_FIELDS:
        v = parameter.get(k)
        if v is None:
            raise ValueError(f"parameter.{k} is required")
    # weights 应在 [0, 1]
    for wkey in ("w_distance", "w_linktime", "w_energy", "w_queue"):
        wv = _to_float(parameter.get(wkey, 0.0))
        if wv < 0.0 or wv > 1.0:
            raise ValueError(f"parameter.{wkey} must be in [0, 1], got {wv}")


def validate_result(result: dict[str, Any]) -> None:
    """校验 ``result`` 字典。"""
    if not isinstance(result, dict):
        raise ValueError("result must be a dict")
    for k in RESULT_FIELDS:
        if k not in result:
            raise ValueError(f"result.{k} is required")


# ---------------------------------------------------------------------- #
# Payload → (scene, parameter, result) 映射
# ---------------------------------------------------------------------- #

# 归一化用分母（与原始字段量级匹配）
_LINK_LIFETIME_REF_S = 60.0   # 60s 视为完全稳定
_QUEUE_LOAD_REF = 10.0        # 队列长度 10 视为满载


def payload_to_experience(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """把 callback ``payload`` 映射为 ``(scene_dict, parameter_dict, result_dict)``。

    Returns:
        ``(scene, parameter, result)`` 三元组；``scene_params`` 回调或字段全缺失
        时返回 ``None``。
    """
    if not isinstance(payload, dict):
        return None
    if payload.get("type") == "scene_params":
        return None

    nodes = payload.get("nodes") or []
    if not nodes or not isinstance(nodes[0], dict):
        return None
    node = nodes[0]

    scene = node.get("scene_info") or {}
    if not scene:
        return None
    m = scene.get("m_info") or {}
    nb = scene.get("neighbor_info") or {}
    para = node.get("para_info") or {}
    weights = para.get("weights") or {}
    res = node.get("result_info") or {}

    # ---- scene（11 维） ----
    scene_dict: dict[str, Any] = {
        "speed": _to_float(m.get("speed")),
        "energy": _to_float(m.get("energy")),
        "queue_length": _to_int(m.get("queue_length", 0)),
        "neighbor_count": _to_int(m.get("neighbor_count", 0)),
        "distance_to_destination": _to_float(m.get("distance_to_destination")),
        "forward_candidate_ratio": _to_float(nb.get("forward_candidate_ratio")),
        "avg_neighbor_distance": _to_float(nb.get("distance_to_me_mean")),
        "relative_speed_mean": _to_float(nb.get("relative_speed_mean")),
        "link_stability": _clamp01(
            _to_float(nb.get("link_lifetime_mean")) / _LINK_LIFETIME_REF_S
        ),
        "link_lifetime_mean": _to_float(nb.get("link_lifetime_mean")),
        "traffic_load": _clamp01(
            _to_float(nb.get("queue_length_mean")) / _QUEUE_LOAD_REF
        ),
    }

    # ---- parameter（6 字段） ----
    param_dict: dict[str, Any] = {
        "hello_interval": _to_float(para.get("hello_interval")),
        "candidate_num": _to_int(para.get("path_num", para.get("multipathCount", 0))),
        "w_distance": _to_float(weights.get("w_distance", weights.get("distance"))),
        "w_linktime": _to_float(weights.get("w_linktime", weights.get("w_linkTime", weights.get("linkTime")))),
        "w_energy": _to_float(weights.get("w_energy", 0.0)),
        "w_queue": _to_float(weights.get("w_queue", 0.0)),
    }

    # ---- result（4 字段） ----
    result_dict: dict[str, Any] = {
        "e2e_pdr": _to_float(res.get("e2e_pdr", res.get("avg_pdr"))),
        "e2e_delay": _to_float(res.get("e2e_delay", res.get("avg_delay"))),
        "routing_overhead": _to_float(res.get("routing_overhead", res.get("control_packets"))),
        "energy_consumption": _to_float(
            res.get("energy_consumption", res.get("e2e_energy"))
        ),
    }

    return scene_dict, param_dict, result_dict
