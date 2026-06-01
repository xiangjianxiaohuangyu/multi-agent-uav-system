"""通信工具。

提供通信智能体使用的工具函数，包含三个核心工具：
1. adjust_weights - 设置单个节点的权重参数
2. adjust_multipath_count - 设置多路径数量
3. adjust_hello_interval - 设置hello时隙

设计原则：
- 大模型只下达"期望目标值"，而非"当前值+调整量"
- 工具内部处理边界限幅和安全校验
- 所有调整都针对单个指定节点
"""

from typing import Any


def adjust_weights(
    target_weights: dict[str, float],
    node_id: int,
) -> dict[str, Any]:
    """设置指定无人机节点的路由协议选择权重（绝对值）。

    Args:
        target_weights: 目标权重字典，必须包含 weightDistance, weightLinkTime,
                        weightRelVelocity, weightNeighborCount。各权重应在 0~1 之间。
        node_id: 目标无人机节点 ID

    Returns:
        包含状态和消息的字典
    """
    required_keys = {"weightDistance", "weightLinkTime", "weightRelVelocity", "weightNeighborCount"}

    # 校验：检查键是否完整
    if not required_keys.issubset(target_weights.keys()):
        return {
            "status": "error",
            "message": f"缺少必要的权重键值，必须包含: {required_keys}"
        }

    # 校验：权重范围检查（允许为0）
    for key in required_keys:
        if not (0 <= target_weights[key] <= 1.0):
            return {
                "status": "error",
                "message": f"权重 {key} 的值必须在 0~1 之间，当前值: {target_weights[key]}"
            }

    # 检查并等比例归一化权重
    total = sum(target_weights[k] for k in required_keys)
    if total != 1.0:
        # 不等于1.0时，等比例缩放至总和为1.0
        normalized_weights = {k: target_weights[k] / total for k in required_keys}
        result_weights = normalized_weights
    else:
        result_weights = target_weights

    return {
        "status": "success",
        "message": f"成功将节点 {node_id} 的权重更新为: {result_weights}",
        "data": {
            "node_id": node_id,
            "weights": result_weights,
        }
    }


def adjust_multipath_count(
    target_count: int,
    node_id: int,
    min_count: int = 1,
    max_count: int = 4,
) -> dict[str, Any]:
    """设置指定无人机节点的多路径路由最大路径数量。

    Args:
        target_count: 期望设定的多路径数量
        node_id: 目标无人机节点 ID
        min_count: 允许的最小值，默认1
        max_count: 允许的最大值，默认4

    Returns:
        包含状态和消息的字典
    """
    # 边界限幅保护
    final_count = max(min_count, min(max_count, target_count))

    return {
        "status": "success",
        "message": f"节点 {node_id} 的多路径数量已设置为: {final_count} (输入目标为: {target_count})",
        "data": {
            "node_id": node_id,
            "target_count": target_count,
            "actual_count": final_count,
        }
    }


def adjust_hello_interval(
    target_interval: float,
    node_id: int,
    min_interval: float = 0.5,
    max_interval: float = 5.0,
) -> dict[str, Any]:
    """设置指定无人机节点的 Hello 报文时间间隔（绝对值）。

    Args:
        target_interval: 期望的 Hello 间隔时间（秒）
        node_id: 目标无人机节点 ID
        min_interval: 最小允许间隔，默认 0.5 秒
        max_interval: 最大允许间隔，默认 5.0 秒

    Returns:
        包含状态和消息的字典
    """
    # 边界限幅保护
    final_interval = max(min_interval, min(max_interval, target_interval))

    # 校验：检查输入是否在合理范围外
    if target_interval < min_interval or target_interval > max_interval:
        input_warning = f"（输入值 {target_interval} 已被限幅）"
    else:
        input_warning = ""

    return {
        "status": "success",
        "message": f"节点 {node_id} 的 Hello 时隙已设置为: {final_interval} 秒 {input_warning}",
        "data": {
            "node_id": node_id,
            "target_interval": target_interval,
            "actual_interval": final_interval,
        }
    }


def get_available_tools() -> list[dict[str, Any]]:
    """获取可用的工具列表（用于 LLM 工具调用）。

    Returns:
        工具定义列表，符合 OpenAI function calling 格式
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "adjust_weights",
                "description": "设置指定节点的链路选择四项权重（绝对值模式）。各个权重值在 0~1 之间，且四者之和必须为 1.0。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_weights": {
                            "type": "object",
                            "description": "目标权重字典",
                            "properties": {
                                "weightDistance": {"type": "number", "description": "距离权重，范围 0~1"},
                                "weightLinkTime": {"type": "number", "description": "链路生存时间权重，范围 0~1"},
                                "weightRelVelocity": {"type": "number", "description": "相对速度权重，范围 0~1"},
                                "weightNeighborCount": {"type": "number", "description": "邻居节点数量权重，范围 0~1"}
                            },
                            "required": ["weightDistance", "weightLinkTime", "weightRelVelocity", "weightNeighborCount"]
                        },
                        "node_id": {"type": "integer", "description": "目标无人机节点 ID"}
                    },
                    "required": ["target_weights", "node_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "adjust_multipath_count",
                "description": "设置指定节点的多路径路由最大路径数量（范围 1~4）。数值越大路由越稳定但开销越高。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_count": {
                            "type": "integer",
                            "description": "期望的目标路径数，例如 3"
                        },
                        "node_id": {"type": "integer", "description": "目标无人机节点 ID"},
                    },
                    "required": ["target_count", "node_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "adjust_hello_interval",
                "description": "设置指定节点的 Hello 报文发送时间间隔。高速移动的节点建议缩短间隔（如 0.5s），静止或低动态节点建议拉长间隔。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_interval": {"type": "number", "description": "期望的间隔秒数，范围 0.5 ~ 5.0"},
                        "node_id": {"type": "integer", "description": "目标无人机节点 ID"},
                    },
                    "required": ["target_interval", "node_id"],
                },
            },
        },
    ]