"""LLM Provider 工厂模块。

通过配置轻松切换不同的 LLM Provider。

提供两种使用方式：
1. 动态创建：`LlmProviderFactory.create("ollama", model="llama3.1:8b")`
2. 预设方法：`LlmProviderFactory.use_ollama_qwen3_4b()` —— 推荐，可读性更好
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from agent.llm_providers.base import LlmProvider
from agent.llm_providers.ollama import OllamaProvider
from agent.llm_providers.qwen import QwenProvider


def _load_api_key() -> str | None:
    """从 secrets/api_key.txt 读取 API key，文件不存在时返回 None。"""
    secrets_path = Path(__file__).parent.parent / "secrets" / "api_key.txt"
    if secrets_path.exists():
        return secrets_path.read_text().strip()
    return None


# 预设名称字面量类型，用于 IDE 自动补全和类型检查
PresetName = Literal[
    "use_ollama_llama3_1_8b",
    "use_ollama_qwen3_4b",
    "use_ollama_llama3_1_4b",
    "use_api_qwen3_6_flash",
]


class LlmProviderFactory:
    """LLM Provider 工厂类。

    既支持按 provider 名称动态创建，也提供语义化的预设方法快速选择。
    """

    _providers: dict[str, type[LlmProvider]] = {
        "ollama": OllamaProvider,
        "qwen": QwenProvider,
    }

    @classmethod
    def register(cls, name: str, provider_class: type[LlmProvider]) -> None:
        """注册新的 Provider。"""
        cls._providers[name] = provider_class

    @classmethod
    def create(cls, provider: str = "ollama", **kwargs: Any) -> LlmProvider:
        """根据 provider 名称动态创建实例。"""
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"Unknown provider '{provider}'. Available: {available}")
        return cls._providers[provider](**kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有可用的 Provider 名称。"""
        return list(cls._providers.keys())

    # ============================================================
    # 预设配置（Presets）—— 通过语义化方法名快速选择 LLM
    # ============================================================

    @staticmethod
    def use_ollama_llama3_1_8b() -> dict[str, Any]:
        """默认：本地 Ollama + Llama 3.1 8B。"""
        return {
            "provider": "ollama",
            "model": "llama3.1:8b",
            "base_url": "http://localhost:11434",
            "api_key": None,
        }

    @staticmethod
    def use_ollama_qwen3_4b() -> dict[str, Any]:
        """本地 Ollama + Qwen3 4B。轻量、工具调用支持较好。"""
        return {
            "provider": "ollama",
            "model": "qwen3:4b",
            "base_url": "http://localhost:11434",
            "api_key": None,
        }

    @staticmethod
    def use_ollama_llama3_1_4b() -> dict[str, Any]:
        """本地 Ollama + Llama 3.1 4B。使用前需先 `ollama pull llama3.1:4b`。"""
        return {
            "provider": "ollama",
            "model": "llama3.1:4b",
            "base_url": "http://localhost:11434",
            "api_key": None,
        }

    @staticmethod
    def use_api_qwen3_6_flash() -> dict[str, Any]:
        """阿里云 DashScope API + Qwen3.6 Flash。需 secrets/api_key.txt。"""
        return {
            "provider": "qwen",
            "model": "qwen3.6-flash",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": _load_api_key(),
        }

    @classmethod
    def get_preset(cls, name: str) -> dict[str, Any]:
        """根据预设名称查找 LLM 配置。

        Args:
            name: 预设方法名，必须以 "use_" 开头（如 "use_ollama_qwen3_4b"）

        Returns:
            LLM 配置字典

        Raises:
            ValueError: 预设名称不存在或格式错误
        """
        if not name or not name.startswith("use_"):
            available = cls.list_presets()
            raise ValueError(
                f"预设名称必须以 'use_' 开头，收到 '{name}'。可用: {', '.join(available)}"
            )
        method = getattr(cls, name, None)
        if method is None or not callable(method):
            available = cls.list_presets()
            raise ValueError(
                f"未找到预设 '{name}'。可用预设: {', '.join(available)}"
            )
        return method()

    @classmethod
    def list_presets(cls) -> list[str]:
        """列出所有可用的预设名称。"""
        return sorted(
            m for m in dir(cls)
            if m.startswith("use_") and callable(getattr(cls, m))
        )


__all__ = [
    "LlmProvider",
    "OllamaProvider",
    "QwenProvider",
    "LlmProviderFactory",
    "PresetName",
]
