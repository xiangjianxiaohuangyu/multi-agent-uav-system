"""通信工具。

提供通信智能体使用的工具函数，包含三个核心工具：
1. adjust_weights - 调整权重参数
2. adjust_multipath_count - 多路径数量
3. adjust_hello_interval - hello时隙
"""

from typing import Any


def adjust_weights(
    current_weights: dict[str, float],
    adjustment: dict[str, float],
) -> dict[str, float]:
    """调整权重参数。

    根据调整量修改当前权重，支持增量调整和绝对值调整两种模式。

    Args:
        current_weights: 当前权重字典，包含 weightDistance, weightLinkTime,
                        weightRelVelocity, weightNeighborCount
        adjustment: 调整量字典，支持:
                   - 增量模式: {"weightDistance": 0.1} 表示增加0.1
                   - 绝对值模式: {"weightDistance": 0.5} 直接设置新值

    Returns:
        调整后的权重字典
    """
    result = dict(current_weights)

    for key, value in adjustment.items():
        if key not in result:
            continue

        if isinstance(value, (int, float)):
            current = result[key]
            new_value = current + value
            # 判断是增量还是绝对值
            if 0 <= new_value <= 1.0:
                result[key] = new_value  # 增量模式
            else:
                result[key] = value  # 绝对值模式

    return result


def adjust_multipath_count(
    current_count: int,
    adjustment: int,
    min_count: int = 1,
    max_count: int = 4,
) -> int:
    """调整多路径数量。

    Args:
        current_count: 当前多路径数量
        adjustment: 调整量（正数增加，负数减少）
        min_count: 最小值，默认1
        max_count: 最大值，默认4

    Returns:
        调整后的多路径数量
    """
    new_count = current_count + adjustment
    return max(min_count, min(max_count, new_count))


def adjust_hello_interval(
    current_interval: float,
    adjustment: float,
    min_interval: float = 0.5,
    max_interval: float = 5.0,
) -> float:
    """调整 hello 时隙间隔。

    Args:
        current_interval: 当前 hello 间隔（秒）
        adjustment: 调整量（正数增加，负数减少）
        min_interval: 最小间隔，默认0.5秒
        max_interval: 最大间隔，默认5.0秒

    Returns:
        调整后的 hello 间隔
    """
    new_interval = current_interval + adjustment
    return max(min_interval, min(max_interval, new_interval))


def get_available_tools() -> list[dict[str, Any]]:
    """获取可用的工具列表（用于 LLM 工具调用）。

    Returns:
        工具定义列表
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "adjust_weights",
                "description": "调整权重参数，用于优化链路选择策略。可以调整 distance、linkTime、relVelocity、neighborCount 四个权重。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "current_weights": {
                            "type": "object",
                            "description": "当前权重字典",
                        },
                        "adjustment": {
                            "type": "object",
                            "description": "调整量字典，如 {\"weightDistance\": 0.1} 表示增加0.1",
                        },
                    },
                    "required": ["current_weights", "adjustment"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "adjust_multipath_count",
                "description": "调整多路径数量，用于控制路由算法的多路径备份数量。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "current_count": {
                            "type": "integer",
                            "description": "当前多路径数量",
                        },
                        "adjustment": {
                            "type": "integer",
                            "description": "调整量（正数增加，负数减少）",
                        },
                    },
                    "required": ["current_count", "adjustment"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "adjust_hello_interval",
                "description": "调整 hello 时隙间隔，用于控制邻居发现频率。间隔越小发现越快但消耗更多资源。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "current_interval": {
                            "type": "number",
                            "description": "当前 hello 间隔（秒）",
                        },
                        "adjustment": {
                            "type": "number",
                            "description": "调整量（秒），正数增加间隔，负数减少间隔",
                        },
                    },
                    "required": ["current_interval", "adjustment"],
                },
            },
        },
    ]
