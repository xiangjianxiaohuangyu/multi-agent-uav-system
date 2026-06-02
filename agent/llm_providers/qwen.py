"""Qwen LLM Provider。

使用阿里云 DashScope API 调用 Qwen 系列大模型。
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any

import httpx

from agent.llm_providers.base import LlmProvider


class QwenProvider(LlmProvider):
    """Qwen LLM Provider，支持 DashScope API 调用。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "qwen3.6-flash",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.model = model
        self.base_url = base_url
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
            "provider": "qwen",
            "messages": messages,
            "chunks": chunks,
            "full_response": response,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        return filepath

    def _convert_messages_for_qwen(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """转换消息格式以适配 Qwen API。

        Args:
            messages: 标准消息格式

        Returns:
            Qwen API 格式的消息列表
        """
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            # Qwen API 不接受 system 角色，放在 user 消息中
            if role == "system":
                role = "user"
            converted.append({
                "role": role,
                "content": msg.get("content", ""),
            })
        return converted

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """调用 Qwen LLM API（非流式）。"""
        request_id = str(uuid.uuid4())[:8]
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        qwen_messages = self._convert_messages_for_qwen(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": qwen_messages,
            "stream": False,
        }

        # Qwen 支持 function call，通过 tools 参数
        if tools:
            qwen_tools = []
            for tool in tools:
                func = tool.get("function", {})
                qwen_tools.append({
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    },
                })
            payload["tools"] = qwen_tools

        response = await self.client.post(url, headers=headers, json=payload)
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
        """流式调用 Qwen LLM API。"""
        request_id = str(uuid.uuid4())[:8]
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        qwen_messages = self._convert_messages_for_qwen(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": qwen_messages,
            "stream": True,
        }

        if tools:
            qwen_tools = []
            for tool in tools:
                func = tool.get("function", {})
                qwen_tools.append({
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    },
                })
            payload["tools"] = qwen_tools

        chunks: list[dict[str, Any]] = []
        full_response: dict[str, Any] = {}
        accumulated_content = ""
        accumulated_tool_calls: list[dict[str, Any]] = []

        async with self.client.stream("POST", url, headers=headers, json=payload) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.strip() or line.startswith(":"):
                    continue

                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break

                chunk = json.loads(line)
                chunks.append(chunk)

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content_delta = delta.get("content", "")
                if content_delta:
                    accumulated_content += content_delta

                # 累积工具调用
                tool_calls = delta.get("tool_calls", [])
                if tool_calls:
                    accumulated_tool_calls.extend(tool_calls)

                # 检查结束
                finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")
                if finish_reason in ("stop", "tool_calls"):
                    full_response = {
                        "message": {
                            "content": accumulated_content,
                            "tool_calls": accumulated_tool_calls,
                        },
                        "done": True,
                    }

        # 如果没有收到完整响应，构建一个
        if not full_response:
            full_response = {
                "message": {
                    "content": accumulated_content,
                    "tool_calls": accumulated_tool_calls,
                },
                "done": True,
            }

        log_path = self._save_llm_raw_response(request_id, messages, full_response, chunks)
        print(f"[LLM Raw Response saved to] {log_path}", flush=True)

        return full_response, chunks