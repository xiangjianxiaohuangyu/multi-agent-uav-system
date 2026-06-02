"""Ollama LLM Provider。

使用本地 Ollama 服务调用大模型。
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any

import httpx

from agent.llm_providers.base import LlmProvider


class OllamaProvider(LlmProvider):
    """Ollama LLM Provider。"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
    ):
        self.base_url = base_url
        self.model = model
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端。"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def close(self):
        """关闭客户端。"""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _save_llm_raw_response(
        self,
        request_id: str,
        messages: list[dict[str, Any]],
        response: dict[str, Any],
        chunks: list[dict[str, Any]] | None = None,
    ) -> str:
        """保存 LLM 原始响应到 log 文件夹。"""
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "log")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_raw_{timestamp}_{request_id}.json"
        filepath = os.path.join(log_dir, filename)

        log_data = {
            "request_id": request_id,
            "timestamp": timestamp,
            "model": self.model,
            "provider": "ollama",
            "messages": messages,
            "chunks": chunks,
            "full_response": response,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        return filepath

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """调用 Ollama LLM API（非流式）。"""
        request_id = str(uuid.uuid4())[:8]
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
        }
        if tools:
            payload["tools"] = tools

        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        full_response = response.json()

        log_path = self._save_llm_raw_response(request_id, messages, full_response)
        print(f"[LLM Raw Response saved to] {log_path}", flush=True)

        return full_response

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """流式调用 Ollama LLM API。"""
        request_id = str(uuid.uuid4())[:8]
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": False,
        }
        if tools:
            payload["tools"] = tools

        chunks: list[dict[str, Any]] = []
        full_response: dict[str, Any] = {}

        async with self.client.stream("POST", url, json=payload) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                chunks.append(chunk)

                if "message" in chunk and "content" in chunk["message"]:
                    content_delta = chunk["message"]["content"]
                    if content_delta:
                        print(content_delta, end="", flush=True)

                if "message" in chunk and "thinking" in chunk["message"]:
                    thinking_delta = chunk["message"]["thinking"]
                    if thinking_delta:
                        print(f"[thinking] {thinking_delta}", end="", flush=True)

                if "message" in chunk and "tool_calls" in chunk["message"]:
                    tool_calls = chunk["message"]["tool_calls"]
                    if tool_calls:
                        print(f"\n[tool_calls] {tool_calls}", flush=True)

                if chunk.get("done"):
                    full_response = chunk

        print()

        log_path = self._save_llm_raw_response(request_id, messages, full_response, chunks)
        print(f"[LLM Raw Response saved to] {log_path}", flush=True)

        return full_response, chunks