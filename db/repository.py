"""仓储层：把 callback payload 转为一行 ``SimulationRecord`` 并写入 MySQL。

核心不变量：
    ``write_simulation_record`` 永不抛出异常。MySQL 写入失败仅打 WARNING，
    不会影响调用方的 JSON 落盘或 HTTP 响应。
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from db.config import is_mysql_disabled
from db.engine import get_session
from db.models import SimulationRecord

log = logging.getLogger(__name__)

# DECIMAL(38,4) 的绝对值上限（约 9.99e33，34 个 9 加 .9999）。
# 任何超过此值的样本都按此截断。
DEC_38_4_MAX = Decimal("9999999999999999999999999999999999.9999")


# ----------------------- 类型转换辅助 ----------------------- #

def _to_decimal(v: Any, default: Decimal = Decimal("0")) -> Decimal:
    """尽力把任意值转成 ``Decimal``，失败回退默认值。"""
    if v is None or v == "":
        return default
    try:
        d = Decimal(str(v))
        if not d.is_finite():
            return default
        return d
    except Exception:
        return default


def _to_int(v: Any, default: int = 0) -> int:
    """尽力把任意值转成 ``int``，失败回退默认值。"""
    if v is None or v == "":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(Decimal(str(v)))
        except Exception:
            return default


def _clamp(v: Decimal, hi: Decimal = DEC_38_4_MAX, lo: Decimal = -DEC_38_4_MAX) -> Decimal:
    """把 ``Decimal`` 截断到 ``[lo, hi]`` 范围内。"""
    if v > hi:
        log.warning("value %s exceeds DECIMAL(38,4) cap; clamping to %s", v, hi)
        return hi
    if v < lo:
        log.warning("value %s below DECIMAL(38,4) cap; clamping to %s", v, lo)
        return lo
    return v


def _truncate(s: str | None, max_len: int, default: str) -> str:
    """截断字符串到 ``max_len``，空值用 ``default``。"""
    if s is None:
        return default
    text = str(s).strip()
    if not text:
        return default
    return text[:max_len]


# ----------------------- 字段提取 ----------------------- #

def _first_node(payload: dict) -> dict:
    """取 ``payload['nodes'][0]``，缺失则返回空 dict。"""
    nodes = payload.get("nodes")
    if not nodes:
        return {}
    first = nodes[0]
    return first if isinstance(first, dict) else {}


def _extract_row(payload: dict) -> SimulationRecord | None:
    """从 callback payload 构造 ``SimulationRecord``。

    返回 ``None`` 的情况：
    - ``type == "scene_params"``（场景参数回调，无仿真指标）
    - 必需字段 ``scene_info`` 完全缺失
    """
    if not isinstance(payload, dict):
        return None
    if payload.get("type") == "scene_params":
        return None

    node = _first_node(payload)
    scene = node.get("scene_info") or {}
    m = scene.get("m_info") or {}
    nb = scene.get("neighbor_info") or {}
    para = node.get("para_info") or {}
    w = para.get("weights") or {}
    res = node.get("result_info") or {}

    # 必需字段缺失则直接跳过（保持 JSON 落盘路径不动）
    if not scene and not para and not res:
        return None

    task_id = _truncate(str(payload.get("task_id") or ""), 64, "default")
    raw_id = payload.get("node_identity", payload.get("node_id"))
    device_id = _truncate(str(raw_id) if raw_id is not None else "", 64, "unknown")
    sim_time = _to_decimal(payload.get("simulation_time", 0))

    # 权重键：实际 JSON 是 w_*，但有些旧数据可能是 distance/linkTime 等
    weight_distance = _to_decimal(w.get("w_distance", w.get("distance")))
    weight_link_time = _to_decimal(w.get("w_linkTime", w.get("linkTime")))
    weight_rel_velocity = _to_decimal(w.get("w_relVelocity", w.get("relVelocity")))
    weight_neighbor_count = _to_decimal(w.get("w_neighborCount", w.get("neighborCount")))

    return SimulationRecord(
        task_id=task_id,
        device_id=device_id,
        simulation_time=sim_time,
        # m_info
        m_speed=_to_decimal(m.get("speed")),
        m_energy=_to_decimal(m.get("energy")),
        m_queue_length=_to_int(m.get("queue_length", 0)),
        m_neighbor_count=_to_int(m.get("neighbor_count", 0)),
        m_distance_to_destination=_to_decimal(m.get("distance_to_destination")),
        # neighbor_info
        nb_forward_candidate_ratio=_to_decimal(nb.get("forward_candidate_ratio")),
        nb_distance_to_me_mean=_to_decimal(nb.get("distance_to_me_mean")),
        nb_distance_to_me_std=_to_decimal(nb.get("distance_to_me_std")),
        nb_distance_to_destination_mean=_to_decimal(nb.get("distance_to_destination_mean")),
        nb_distance_to_destination_std=_to_decimal(nb.get("distance_to_destination_std")),
        nb_distance_to_destination_min=_to_decimal(nb.get("distance_to_destination_min")),
        nb_relative_speed_mean=_clamp(_to_decimal(nb.get("relative_speed_mean"))),
        nb_relative_speed_std=_clamp(_to_decimal(nb.get("relative_speed_std"))),
        nb_link_lifetime_mean=_to_decimal(nb.get("link_lifetime_mean")),
        nb_link_lifetime_std=_to_decimal(nb.get("link_lifetime_std")),
        nb_neighbor_degree_mean=_to_decimal(nb.get("neighbor_degree_mean")),
        nb_neighbor_degree_std=_to_decimal(nb.get("neighbor_degree_std")),
        nb_queue_length_mean=_to_decimal(nb.get("queue_length_mean")),
        nb_queue_length_std=_to_decimal(nb.get("queue_length_std")),
        nb_queue_length_max=_to_int(nb.get("queue_length_max", 0)),
        nb_energy_mean=_to_decimal(nb.get("energy_mean")),
        nb_energy_std=_to_decimal(nb.get("energy_std")),
        nb_energy_min=_to_decimal(nb.get("energy_min")),
        # para_info
        param_hello_interval=_to_decimal(para.get("hello_interval")),
        param_path_num=_to_int(para.get("path_num", 0)),
        weight_distance=weight_distance,
        weight_link_time=weight_link_time,
        weight_rel_velocity=weight_rel_velocity,
        weight_neighbor_count=weight_neighbor_count,
        # result_info
        res_avg_pdr=_to_decimal(res.get("avg_pdr")),
        res_avg_delay=_to_decimal(res.get("avg_delay")),
        res_energy_consumption=_to_decimal(res.get("energy_consumption")),
        res_control_packets=_to_int(res.get("control_packets", 0)),
        res_distance_progress=_to_decimal(res.get("distance_progress")),
    )


# ----------------------- 对外主入口 ----------------------- #

def write_simulation_record(payload: dict) -> bool:
    """把 callback payload 写入 MySQL。

    行为契约：
    - MySQL 被禁用（``MYSQL_DISABLED=1``）→ 返回 ``False``，不抛
    - 数据格式不合法（scene_params / 缺字段）→ 返回 ``False``，不抛
    - MySQL 不可达 / 写入失败 → 打 WARNING 日志，返回 ``False``，不抛
    - 成功 → 返回 ``True``

    调用方可以放心地把这个函数嵌在 HTTP 处理流程中，不必再包 try/except。
    """
    if is_mysql_disabled():
        log.debug("mysql disabled by env, skip write")
        return False

    try:
        row = _extract_row(payload)
        if row is None:
            return False
        with get_session() as session:
            session.add(row)
            session.commit()
        return True
    except SQLAlchemyError as e:
        log.warning("mysql write failed: %s", e)
        return False
    except Exception as e:
        # 兜底：任何未预期异常都不应阻断 JSON 写路径
        log.exception("unexpected error in write_simulation_record: %s", e)
        return False
