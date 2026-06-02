"""LLM Providers 模块。

支持多种大模型的切换：
- Ollama: 本地模型
- Qwen: 阿里云 DashScope API
"""

from agent.llm_providers.base import LlmProvider
from agent.llm_providers.factory import LlmProviderFactory
from agent.llm_providers.ollama import OllamaProvider
from agent.llm_providers.qwen import QwenProvider

__all__ = [
    "LlmProvider",
    "OllamaProvider",
    "QwenProvider",
    "LlmProviderFactory",
]