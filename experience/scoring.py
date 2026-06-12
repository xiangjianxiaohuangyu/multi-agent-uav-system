"""经验库的纯函数模块：场景向量构造、payload 字段映射、字典校验。

本模块是 ``scene_to_vector`` / ``payload_to_experience`` / ``validate_*``
的单一真相源。所有可被单元测试覆盖的逻辑都集中在这里，零 I/O。
"""

from __future__ import annotations

import logging
from typing import Any

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
    "distance_to_me_mean",
    "distance_to_me_std",
    "distance_to_destination_mean",
    "distance_to_destination_std",
    "distance_to_destination_min",
    "relative_speed_mean",
    "relative_speed_std",
    "link_lifetime_mean",
    "link_lifetime_std",
    "neighbor_degree_mean",
    "neighbor_degree_std",
    "queue_length_mean",
    "queue_length_std",
    "queue_length_max",
    "energy_mean",
    "energy_std",
    "energy_min",
)
SCENE_DIMENSION: int = len(SCENE_FIELD_ORDER)

PARAM_FIELDS: tuple[str, ...] = (
    "hello_interval",
    "path_num",
    "w_distance",
    "w_linkTime",
    "w_relVelocity",
    "w_neighborCount",
)

RESULT_FIELDS: tuple[str, ...] = (
    "avg_pdr",
    "avg_delay",
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
    """把场景字典转换为 23 维定长 float 向量。

    严格按 ``SCENE_FIELD_ORDER`` 顺序取值；缺失键 → ``0.0``；非数值 → ``0.0``。
    """
    if not isinstance(scene, dict):
        return [0.0] * SCENE_DIMENSION
    return [_to_float(scene.get(k)) for k in SCENE_FIELD_ORDER]


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
    for wkey in ("w_distance", "w_linkTime", "w_relVelocity", "w_neighborCount"):
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


def payload_to_experience(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """把 callback ``payload`` 映射为 ``(scene_dict, parameter_dict, result_dict)``。

    支持的 payload 结构（NS3 当前发送的扁平格式，字段全部位于顶层）：
        {
            "speed": ..., "energy": ..., "queue_length": ...,
            "neighbor_count": ..., "distance_to_destination": ...,
            "forward_candidate_ratio": ..., "distance_to_me_*": ...,
            "distance_to_destination_*": ..., "relative_speed_*": ...,
            "link_lifetime_*": ..., "neighbor_degree_*": ...,
            "queue_length_*": ..., "energy_*": ...,
            "hello_interval": ..., "path_num": ...,
            "w_distance": ..., "w_linkTime": ...,
            "w_relVelocity": ..., "w_neighborCount": ...,
            "avg_pdr": ..., "avg_delay": ...
        }

    Returns:
        ``(scene, parameter, result)`` 三元组；``scene_params`` 回调或
        ``payload`` 为空时返回 ``None``。

    注意：
    - parameter / result 的键名严格对齐 NS3 payload。
    """
    if not isinstance(payload, dict):
        return None
    if payload.get("type") == "scene_params":
        return None
    if not payload:
        return None

    # ---- scene（23 维） ----
    scene_dict: dict[str, Any] = {
        "speed": _to_float(payload.get("speed")),
        "energy": _to_float(payload.get("energy")),
        "queue_length": _to_int(payload.get("queue_length", 0)),
        "neighbor_count": _to_int(payload.get("neighbor_count", 0)),
        "distance_to_destination": _to_float(payload.get("distance_to_destination")),
        "forward_candidate_ratio": _to_float(payload.get("forward_candidate_ratio")),
        "distance_to_me_mean": _to_float(payload.get("distance_to_me_mean")),
        "distance_to_me_std": _to_float(payload.get("distance_to_me_std")),
        "distance_to_destination_mean": _to_float(payload.get("distance_to_destination_mean")),
        "distance_to_destination_std": _to_float(payload.get("distance_to_destination_std")),
        "distance_to_destination_min": _to_float(payload.get("distance_to_destination_min")),
        "relative_speed_mean": _to_float(payload.get("relative_speed_mean")),
        "relative_speed_std": _to_float(payload.get("relative_speed_std")),
        "link_lifetime_mean": _to_float(payload.get("link_lifetime_mean")),
        "link_lifetime_std": _to_float(payload.get("link_lifetime_std")),
        "neighbor_degree_mean": _to_float(payload.get("neighbor_degree_mean")),
        "neighbor_degree_std": _to_float(payload.get("neighbor_degree_std")),
        "queue_length_mean": _to_float(payload.get("queue_length_mean")),
        "queue_length_std": _to_float(payload.get("queue_length_std")),
        "queue_length_max": _to_int(payload.get("queue_length_max", 0)),
        "energy_mean": _to_float(payload.get("energy_mean")),
        "energy_std": _to_float(payload.get("energy_std")),
        "energy_min": _to_float(payload.get("energy_min")),
    }

    # ---- parameter（6 字段，键名对齐 NS3） ----
    param_dict: dict[str, Any] = {
        "hello_interval": _to_float(payload.get("hello_interval")),
        "path_num": _to_int(payload.get("path_num", 0)),
        "w_distance": _to_float(payload.get("w_distance")),
        "w_linkTime": _to_float(payload.get("w_linkTime")),
        "w_relVelocity": _to_float(payload.get("w_relVelocity")),
        "w_neighborCount": _to_float(payload.get("w_neighborCount")),
    }

    # ---- result（2 字段，键名对齐 NS3） ----
    result_dict: dict[str, Any] = {
        "avg_pdr": _to_float(payload.get("avg_pdr")),
        "avg_delay": _to_float(payload.get("avg_delay")),
    }

    return scene_dict, param_dict, result_dict
