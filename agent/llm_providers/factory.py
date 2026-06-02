"""LLM Provider 工厂模块。

通过配置轻松切换不同的 LLM Provider。
"""

from __future__ import annotations

from typing import Any

from agent.llm_providers.base import LlmProvider
from agent.llm_providers.ollama import OllamaProvider
from agent.llm_providers.qwen import QwenProvider


class LlmProviderFactory:
    """LLM Provider 工厂类。"""

    _providers: dict[str, type[LlmProvider]] = {
        "ollama": OllamaProvider,
        "qwen": QwenProvider,
    }

    @classmethod
    def register(cls, name: str, provider_class: type[LlmProvider]) -> None:
        """注册新的 Provider。

        Args:
            name: Provider 名称
            provider_class: Provider 类
        """
        cls._providers[name] = provider_class

    @classmethod
    def create(cls, provider: str = "ollama", **kwargs: Any) -> LlmProvider:
        """创建指定 Provider 的实例。

        Args:
            provider: Provider 名称（ollama / qwen）
            **kwargs: 传递给 Provider 构造函数的参数

        Returns:
            LlmProvider 实例

        Raises:
            ValueError: 未知的 provider 名称
        """
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"Unknown provider '{provider}'. Available: {available}")

        provider_class = cls._providers[provider]
        return provider_class(**kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有可用的 Provider。"""
        return list(cls._providers.keys())


__all__ = [
    "LlmProvider",
    "OllamaProvider",
    "QwenProvider",
    "LlmProviderFactory",
]