"""数据解析模块。

负责解析从 ns3 推送过来的仿真数据，并存储历史数据用于预测。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeInfo:
    """节点信息。"""
    position: list = field(default_factory=list)
    velocity: list = field(default_factory=list)
    energy_percentage: float = 0.0
    hello_interval: float = 0.0
    weight_distance: float = 0.0
    weight_link_time: float = 0.0
    weight_rel_velocity: float = 0.0
    weight_neighbor_count: float = 0.0
    multi_path_count: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "NodeInfo":
        """从字典创建节点信息。"""
        weights = data.get("weights", {})
        return cls(
            position=data.get("position", []),
            velocity=data.get("velocity", []),
            energy_percentage=data.get("energy_percentage", 0.0),
            hello_interval=data.get("hello_interval", 0.0),
            weight_distance=weights.get("distance", 0.0),
            weight_link_time=weights.get("linkTime", 0.0),
            weight_rel_velocity=weights.get("relVelocity", 0.0),
            weight_neighbor_count=weights.get("neighborCount", 0.0),
            multi_path_count=data.get("multipathCount", 0),
        )


@dataclass
class SimulationData:
    """仿真数据解析结果。"""
    status: str
    nodes: list[NodeInfo] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationData":
        """解析 JSON 字典为 SimulationData。"""
        nodes = [NodeInfo.from_dict(n) for n in data.get("nodes", [])]
        return cls(
            status=data.get("status", ""),
            nodes=nodes,
        )


@dataclass
class CommunicationPair:
    """通信对。"""
    source: int
    destination: int


@dataclass
class RawNodeSnapshot:
    """单条仿真回调的扁平化快照，对应 ``simulation_records`` 一行。

    与 ``NodeInfo`` 的区别：
    - ``NodeInfo`` 是 LLM agent 用的字段集（轻量、聚焦参数调优）
    - ``RawNodeSnapshot`` 是持久化层用的字段集（完整、聚焦指标落库）

    引入这个类是为了不破坏 ``NodeInfo`` 的下游消费者（agent.py / tool schemas）。
    """

    # m_info
    m_speed: float = 0.0
    m_energy: float = 0.0
    m_queue_length: int = 0
    m_neighbor_count: int = 0
    m_distance_to_destination: float = 0.0

    # neighbor_info
    nb_forward_candidate_ratio: float = 0.0
    nb_distance_to_me_mean: float = 0.0
    nb_distance_to_me_std: float = 0.0
    nb_distance_to_destination_mean: float = 0.0
    nb_distance_to_destination_std: float = 0.0
    nb_distance_to_destination_min: float = 0.0
    nb_relative_speed_mean: float = 0.0
    nb_relative_speed_std: float = 0.0
    nb_link_lifetime_mean: float = 0.0
    nb_link_lifetime_std: float = 0.0
    nb_neighbor_degree_mean: float = 0.0
    nb_neighbor_degree_std: float = 0.0
    nb_queue_length_mean: float = 0.0
    nb_queue_length_std: float = 0.0
    nb_queue_length_max: int = 0
    nb_energy_mean: float = 0.0
    nb_energy_std: float = 0.0
    nb_energy_min: float = 0.0

    # para_info
    param_hello_interval: float = 0.0
    param_path_num: int = 0
    weight_distance: float = 0.0
    weight_link_time: float = 0.0
    weight_rel_velocity: float = 0.0
    weight_neighbor_count: float = 0.0

    # result_info
    res_avg_pdr: float = 0.0
    res_avg_delay: float = 0.0

    @classmethod
    def from_payload(cls, payload: dict) -> "RawNodeSnapshot | None":
        """从 callback payload 构造快照。``scene_params`` 回调返回 None。

        NS3 当前发送扁平 payload（字段全在顶层），不再有 ``nodes`` / ``scene_info`` / ``m_info`` 等嵌套。
        """
        if not isinstance(payload, dict) or payload.get("type") == "scene_params":
            return None
        if not payload:
            return None

        return cls(
            m_speed=float(payload.get("speed") or 0.0),
            m_energy=float(payload.get("energy") or 0.0),
            m_queue_length=int(payload.get("queue_length") or 0),
            m_neighbor_count=int(payload.get("neighbor_count") or 0),
            m_distance_to_destination=float(payload.get("distance_to_destination") or 0.0),
            nb_forward_candidate_ratio=float(payload.get("forward_candidate_ratio") or 0.0),
            nb_distance_to_me_mean=float(payload.get("distance_to_me_mean") or 0.0),
            nb_distance_to_me_std=float(payload.get("distance_to_me_std") or 0.0),
            nb_distance_to_destination_mean=float(payload.get("distance_to_destination_mean") or 0.0),
            nb_distance_to_destination_std=float(payload.get("distance_to_destination_std") or 0.0),
            nb_distance_to_destination_min=float(payload.get("distance_to_destination_min") or 0.0),
            nb_relative_speed_mean=float(payload.get("relative_speed_mean") or 0.0),
            nb_relative_speed_std=float(payload.get("relative_speed_std") or 0.0),
            nb_link_lifetime_mean=float(payload.get("link_lifetime_mean") or 0.0),
            nb_link_lifetime_std=float(payload.get("link_lifetime_std") or 0.0),
            nb_neighbor_degree_mean=float(payload.get("neighbor_degree_mean") or 0.0),
            nb_neighbor_degree_std=float(payload.get("neighbor_degree_std") or 0.0),
            nb_queue_length_mean=float(payload.get("queue_length_mean") or 0.0),
            nb_queue_length_std=float(payload.get("queue_length_std") or 0.0),
            nb_queue_length_max=int(payload.get("queue_length_max") or 0),
            nb_energy_mean=float(payload.get("energy_mean") or 0.0),
            nb_energy_std=float(payload.get("energy_std") or 0.0),
            nb_energy_min=float(payload.get("energy_min") or 0.0),
            param_hello_interval=float(payload.get("hello_interval") or 0.0),
            param_path_num=int(payload.get("path_num") or 0),
            weight_distance=float(
                payload.get("w_distance", payload.get("distance")) or 0.0
            ),
            weight_link_time=float(
                payload.get("w_linkTime", payload.get("linkTime")) or 0.0
            ),
            weight_rel_velocity=float(
                payload.get("w_relVelocity", payload.get("relVelocity")) or 0.0
            ),
            weight_neighbor_count=float(
                payload.get("w_neighborCount", payload.get("neighborCount")) or 0.0
            ),
            res_avg_pdr=float(payload.get("avg_pdr") or 0.0),
            res_avg_delay=float(payload.get("avg_delay") or 0.0),
        )


@dataclass
class SceneParamsData:
    """场景参数数据。"""
    task_id: str
    node_count: int
    communication_pairs: list[CommunicationPair] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SceneParamsData":
        """解析 JSON 字典为 SceneParamsData。"""
        pairs = [CommunicationPair(**p) for p in data.get("communication_pairs", [])]
        return cls(
            task_id=data.get("task_id", ""),
            node_count=data.get("node_count", 0),
            communication_pairs=pairs,
        )


def parse_scene_params_data(data: dict[str, Any]) -> SceneParamsData:
    """解析场景参数数据并存储到全局数据存储。

    Args:
        data: 从 ns3 推送过来的场景参数字典

    Returns:
        解析后的 SceneParamsData 对象
    """
    scene_data = SceneParamsData.from_dict(data)
    _data_store.add_scene_params_data(scene_data)
    return scene_data


class DataStore:
    """数据存储类，保存历史节点状态用于预测。"""

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        # node_history 以节点在 sim_data.nodes 列表中的下标为 key
        # （NS3 不再传 node_id 后无法用业务 ID 区分节点）
        self.node_history: dict[int, list[NodeInfo]] = defaultdict(list)
        self.task_history: list[SimulationData] = []
        self.scene_params_history: list[SceneParamsData] = []

    def add_simulation_data(self, sim_data: SimulationData):
        """添加仿真数据到历史记录。"""
        self.task_history.append(sim_data)
        if len(self.task_history) > self.max_history:
            self.task_history.pop(0)

        for idx, node in enumerate(sim_data.nodes):
            self.node_history[idx].append(node)
            if len(self.node_history[idx]) > self.max_history:
                self.node_history[idx].pop(0)

        # print(f"[DataStore] node_history: {self.node_history}", flush=True)
        # print()
        # print(f"[DataStore] task_history: {self.task_history}", flush=True)
        # print()

    def add_scene_params_data(self, scene_data: SceneParamsData):
        """添加场景参数数据到历史记录。"""
        self.scene_params_history.append(scene_data)
        if len(self.scene_params_history) > self.max_history:
            self.scene_params_history.pop(0)


# 全局数据存储实例
_data_store = DataStore()


def get_data_store() -> DataStore:
    """获取全局数据存储实例。"""
    return _data_store


def parse_simulation_data(data: dict[str, Any]) -> SimulationData:
    """解析仿真数据的入口函数。

    Args:
        data: 从 ns3 推送过来的仿真数据字典

    Returns:
        解析后的 SimulationData 对象
    """
    sim_data = SimulationData.from_dict(data)
    # print(f"[Parser] sim_data: {sim_data}", flush=True)
    # print()

    _data_store.add_simulation_data(sim_data)
    return sim_data


def parse_by_type(data: dict[str, Any]) -> tuple[str, SimulationData | SceneParamsData]:
    """根据 type 字段解析不同的数据类型。

    Args:
        data: 从 ns3 推送过来的数据字典

    Returns:
        (type, parsed_data) 元组
    """
    msg_type = data.get("type", "simulation")

    if msg_type == "scene_params":
        return (msg_type, SceneParamsData.from_dict(data))
    elif msg_type == "simulation":
        # 优先识别扁平 payload（NS3 当前格式：字段全在顶层，没有 nodes 嵌套）
        if "nodes" in data and isinstance(data["nodes"], list):
            # 旧版嵌套格式
            return (msg_type, parse_simulation_data(data))
        else:
            # 新版扁平 payload：把顶层字段组装成一个 NodeInfo
            return (msg_type, _parse_flat_simulation_data(data))
    else:
        # 未知 type 走默认 simulation 解析
        return (msg_type, _parse_flat_simulation_data(data))


def _parse_flat_simulation_data(data: dict[str, Any]) -> SimulationData:
    """解析扁平 payload（NS3 当前格式）为 SimulationData。

    NS3 不再发送 ``nodes`` 嵌套数组，而是把当前节点的 m_info / nb_* / para / result
    全平铺在顶层。本函数把这些字段组装成一个 ``NodeInfo``，让下游 agent /
    experience 库都能正常处理。
    """
    weights = {
        "distance": float(data.get("w_distance", data.get("distance")) or 0.0),
        "linkTime": float(data.get("w_linkTime", data.get("linkTime")) or 0.0),
        "relVelocity": float(data.get("w_relVelocity", data.get("relVelocity")) or 0.0),
        "neighborCount": float(
            data.get("w_neighborCount", data.get("neighborCount")) or 0.0
        ),
    }
    node = NodeInfo(
        position=list(data.get("position", []) or []),
        velocity=list(data.get("velocity", []) or []),
        # NS3 发 ``energy``（值域 0-100 的百分比）；兼容 ``energy_percentage`` 字段名
        energy_percentage=float(
            data.get("energy", data.get("energy_percentage", 0.0)) or 0.0
        ),
        hello_interval=float(data.get("hello_interval", 0.0) or 0.0),
        weight_distance=weights["distance"],
        weight_link_time=weights["linkTime"],
        weight_rel_velocity=weights["relVelocity"],
        weight_neighbor_count=weights["neighborCount"],
        multi_path_count=int(data.get("multipathCount", data.get("path_num", 0)) or 0),
    )
    sim = SimulationData(status=data.get("status", ""), nodes=[node])
    # 同步把扁平字段也存到 _data_store，便于历史查询
    try:
        _data_store.add_simulation_data(sim)
    except Exception:  # noqa: BLE001
        pass
    return sim