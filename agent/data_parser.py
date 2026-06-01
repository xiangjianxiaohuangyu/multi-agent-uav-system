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
    id: str
    position: list = field(default_factory=list)
    velocity: list = field(default_factory=list)
    energy_percentage: float = 0.0
    hello_interval: float = 0.0
    simulation_time: float = 0.0
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
            id=data.get("id", ""),
            position=data.get("position", []),
            velocity=data.get("velocity", []),
            energy_percentage=data.get("energy_percentage", 0.0),
            hello_interval=data.get("hello_interval", 0.0),
            simulation_time=data.get("simulation_time", 0.0),
            weight_distance=weights.get("distance", 0.0),
            weight_link_time=weights.get("linkTime", 0.0),
            weight_rel_velocity=weights.get("relVelocity", 0.0),
            weight_neighbor_count=weights.get("neighborCount", 0.0),
            multi_path_count=data.get("multipathCount", 0),
        )


@dataclass
class SimulationData:
    """仿真数据解析结果。"""
    task_id: str
    status: str
    simulation_time: float = 0.0
    nodes: list[NodeInfo] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationData":
        """解析 JSON 字典为 SimulationData。"""
        nodes = [NodeInfo.from_dict(n) for n in data.get("nodes", [])]
        return cls(
            task_id=data.get("task_id", ""),
            status=data.get("status", ""),
            simulation_time=data.get("simulation_time", 0.0),
            nodes=nodes,
        )


@dataclass
class CommunicationPair:
    """通信对。"""
    source: int
    destination: int


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
        self.node_history: dict[str, list[NodeInfo]] = defaultdict(list)
        self.task_history: list[SimulationData] = []
        self.scene_params_history: list[SceneParamsData] = []

    def add_simulation_data(self, sim_data: SimulationData):
        """添加仿真数据到历史记录。"""
        self.task_history.append(sim_data)
        if len(self.task_history) > self.max_history:
            self.task_history.pop(0)

        for node in sim_data.nodes:
            self.node_history[node.id].append(node)
            if len(self.node_history[node.id]) > self.max_history:
                self.node_history[node.id].pop(0)

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
    else:
        # 默认按 simulation 处理
        return (msg_type, parse_simulation_data(data))