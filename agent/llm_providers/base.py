"""LLM Provider 接口模块。

定义 LLM providers 必须实现的抽象接口，便于切换不同大模型。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LlmProvider(ABC):
    """LLM Provider 抽象基类。"""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """发送聊天请求到 LLM。

        Args:
            messages: 消息列表
            tools: 工具列表（可选）
            stream: 是否使用流式输出（可选）

        Returns:
            LLM 响应字典
        """
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """流式调用 LLM，边输出边收集结果。

        Args:
            messages: 消息列表
            tools: 工具列表（可选）

        Returns:
            (完整响应, 流式块列表)
        """
        ...