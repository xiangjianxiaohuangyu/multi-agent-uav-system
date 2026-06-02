"""通信工具。

提供通信智能体使用的工具函数：
1. adjust_weights - 设置单个节点的权重参数

设计原则：
- 大模型只下达"期望目标值"，而非"当前值+调整量"
- 工具内部处理边界限幅和安全校验
- 工具只返回结果，不直接修改 nodes
"""

from typing import Any


def adjust_weights(
    adjustment: dict[str, Any],
) -> dict[str, Any]:
    """设置单个无人机节点的路由协议选择权重（绝对值）。

    Args:
        adjustment: 调整参数，包含:
                    - node_id: 目标无人机节点 ID
                    - target_weights: 目标权重字典，必须包含 weightDistance, weightLinkTime,
                                      weightRelVelocity, weightNeighborCount。各权重应在 0~1 之间。

    Returns:
        包含状态和消息的字典
    """
    required_keys = {"weightDistance", "weightLinkTime", "weightRelVelocity", "weightNeighborCount"}
    node_id = adjustment.get("node_id")
    target_weights = adjustment.get("target_weights", {})

    # 校验：检查键是否完整
    if not required_keys.issubset(target_weights.keys()):
        return {
            "status": "error",
            "message": f"节点 {node_id}: 缺少必要的权重键值，必须包含: {required_keys}",
        }

    # 校验：权重范围检查（允许为0）
    for key in required_keys:
        if not (0 <= target_weights[key] <= 1.0):
            return {
                "status": "error",
                "message": f"节点 {node_id}: 权重 {key} 的值必须在 0~1 之间，当前值: {target_weights[key]}",
            }

    # 检查并等比例归一化权重
    total = sum(target_weights[k] for k in required_keys)
    if total != 1.0:
        normalized_weights = {k: target_weights[k] / total for k in required_keys}
        result_weights = normalized_weights
    else:
        result_weights = target_weights

    print(f"[adjust_weights] node_id={node_id}, distance={result_weights['weightDistance']:.2f}, link_time={result_weights['weightLinkTime']:.2f}, rel_velocity={result_weights['weightRelVelocity']:.2f}, neighbor_count={result_weights['weightNeighborCount']:.2f}")

    return {
        "status": "success",
        "message": f"成功更新节点 {node_id} 的权重",
        "data": {
            "node_id": node_id,
            "weights": {
                "distance": result_weights["weightDistance"],
                "linkTime": result_weights["weightLinkTime"],
                "relVelocity": result_weights["weightRelVelocity"],
                "neighborCount": result_weights["weightNeighborCount"],
            },
        },
    }


def adjust_multipath_count(
    adjustment: dict[str, Any],
) -> dict[str, Any]:
    """设置单个无人机节点的多路径路由最大路径数量。

    Args:
        adjustment: 调整参数，包含:
                    - node_id: 目标无人机节点 ID
                    - target_count: 期望设定的多路径数量（范围 1~4）

    Returns:
        包含状态和消息的字典
    """
    node_id = adjustment.get("node_id")
    target_count = adjustment.get("target_count")

    if target_count is None:
        return {
            "status": "error",
            "message": f"节点 {node_id}: 缺少 target_count 参数",
        }

    # 边界限幅保护
    min_count = 1
    max_count = 4
    final_count = max(min_count, min(max_count, target_count))

    print(f"[adjust_multipath_count] node_id={node_id}, old_count=?, new_count={final_count}")

    return {
        "status": "success",
        "message": f"成功更新节点 {node_id} 的多路径数量为 {final_count}",
        "data": {
            "node_id": node_id,
            "old_count": None,  # 由调用方填充
            "new_count": final_count,
        },
    }


def adjust_hello_interval(
    adjustment: dict[str, Any],
) -> dict[str, Any]:
    """设置单个无人机节点的 Hello 报文时间间隔。

    Args:
        adjustment: 调整参数，包含:
                    - node_id: 目标无人机节点 ID
                    - target_interval: 期望的 Hello 间隔时间（秒，范围 0.5~5.0）

    Returns:
        包含状态和消息的字典
    """
    node_id = adjustment.get("node_id")
    target_interval = adjustment.get("target_interval")

    if target_interval is None:
        return {
            "status": "error",
            "message": f"节点 {node_id}: 缺少 target_interval 参数",
        }

    # 边界限幅保护
    min_interval = 0.5
    max_interval = 5.0
    final_interval = max(min_interval, min(max_interval, target_interval))

    print(f"[adjust_hello_interval] node_id={node_id}, old_interval=?, new_interval={final_interval}")

    return {
        "status": "success",
        "message": f"成功更新节点 {node_id} 的 Hello 间隔为 {final_interval} 秒",
        "data": {
            "node_id": node_id,
            "old_interval": None,  # 由调用方填充
            "new_interval": final_interval,
        },
    }


# 内部工具函数映射
# TOOL_FUNCTIONS = {
#     "adjust_weights": adjust_weights,
#     "adjust_multipath_count": adjust_multipath_count,
#     "adjust_hello_interval": adjust_hello_interval,
# }


def get_available_tools() -> list[dict[str, Any]]:
    """获取可用的工具列表（用于 LLM 工具调用）。

    注意：返回的 schema 不包含 nodes 参数，nodes 由 Agent 在调用时内部注入。

    Returns:
        工具定义列表，符合 OpenAI function calling 格式
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "adjust_weights",
                "description": "【决策动作】直接优化下一跳路由的选择策略。当节点高动态移动时，必须加大 weightLinkTime(链路生存时间) 和 weightRelVelocity(相对速度) 的比重；当节点密集时，加大 weightNeighborCount(邻居数) 的比重。工具内部会自动做归一化，你只需给出相对重要性评分。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "adjustment": {
                            "type": "object",
                            "description": "单个节点的权重调整参数",
                            "properties": {
                                "node_id": {"type": "integer", "description": "目标无人机节点 ID"},
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
                                }
                            },
                            "required": ["node_id", "target_weights"]
                        }
                    },
                    "required": ["adjustment"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "adjust_multipath_count",
                "description": "【决策动作】控制多路径备份链路数量。当仿真网络发生拥堵、单路径断连频繁、丢包率上升时，应增大此数值（范围 1~4）以提升数据传输的容错和成功率。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "adjustment": {
                            "type": "object",
                            "description": "单个节点的多路径数量调整参数",
                            "properties": {
                                "node_id": {"type": "integer", "description": "目标无人机节点 ID"},
                                "target_count": {"type": "integer", "description": "期望的目标路径数，范围 1~4"}
                            },
                            "required": ["node_id", "target_count"]
                        }
                    },
                    "required": ["adjustment"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "adjust_hello_interval",
                "description": "【决策动作】调节邻居发现频率以平衡『拓扑感知速度』与『信道开销』。节点速度快时必须减小间隔（如0.5秒）防止拓扑过期；节点静止或信道拥堵时必须增大间隔（如2.0秒）以优化网络开销。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "adjustment": {
                            "type": "object",
                            "description": "单个节点的 Hello 间隔调整参数",
                            "properties": {
                                "node_id": {"type": "integer", "description": "目标无人机节点 ID"},
                                "target_interval": {"type": "number", "description": "期望的间隔秒数，范围 0.5 ~ 5.0"}
                            },
                            "required": ["node_id", "target_interval"]
                        }
                    },
                    "required": ["adjustment"],
                },
            },
        },
    ]