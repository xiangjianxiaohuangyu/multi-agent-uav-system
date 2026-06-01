"""智能体处理模块。

负责处理从 ns3 推送过来的仿真数据，支持 LLM 推理和工具调用。
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

import httpx

from agent.data_parser import SimulationData
from agent.tools import get_available_tools


class OllamaConfig(TypedDict):
    """Ollama 配置。"""
    base_url: str
    model: str


class ToolCall(TypedDict):
    """工具调用。"""
    name: str
    arguments: dict[str, Any]


DEFAULT_OLLAMA_CONFIG: OllamaConfig = {
    "base_url": "http://localhost:11434",
    "model": "qwen3:4b",
}


class LlmAgent:
    """基于 LLM 的智能体，支持工具调用。"""

    def __init__(
        self,
        agent_id: str = "main_agent",
        name: str = "Main Agent",
        ollama_config: OllamaConfig | None = None,
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ):
        self.agent_id = agent_id
        self.name = name
        self.ollama_config = ollama_config or DEFAULT_OLLAMA_CONFIG
        self.tools = tools or get_available_tools()
        self.system_prompt = system_prompt or self._default_system_prompt()
        self._client: httpx.AsyncClient | None = None

    def _default_system_prompt(self) -> str:
        """默认系统提示词。"""
        return """你是一个专业的 UAV 网络通信智能体，负责分析和优化多智能体系统的链路质量。

你可以使用以下三个工具：
1. adjust_weights - 调整权重参数（distance、linkTime、relVelocity、neighborCount）
2. adjust_multipath_count - 调整多路径数量（1-4）
3. adjust_hello_interval - 调整 hello 时隙间隔（0.5-5.0秒）

每次分析后，请提供：
1. 当前网络状态分析
2. 建议的优化方案
3. 参数调整的具体数值

仅在收到 simulation 数据时调用工具进行分析。"""

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

    async def _call_llm(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """调用 Ollama LLM API。

        Args:
            messages: 消息列表
            tools: 工具列表

        Returns:
            LLM 响应
        """
        url = f"{self.ollama_config['base_url']}/api/chat"
        payload = {
            "model": self.ollama_config["model"],
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        # 发送请求到 Ollama API
        response = await self.client.post(url, json=payload)
        # 检查 HTTP 状态码，非 2xx 会抛出异常
        response.raise_for_status()
        return response.json()

    def _format_simulation_data(self, sim_data: SimulationData) -> str:
        """格式化仿真数据为文本描述。"""
        node_info_list = []
        for n in sim_data.nodes:
            weights = {
                "distance": n.weight_distance,
                "linkTime": n.weight_link_time,
                "relVelocity": n.weight_rel_velocity,
                "neighborCount": n.weight_neighbor_count,
            }
            node_info_list.append({
                "id": n.id,
                "position": n.position,
                "velocity": n.velocity,
                "energy_percentage": n.energy_percentage,
                "weights": weights,
                "multipathCount": n.multi_path_count,
            })

        return json.dumps({
            "task_id": sim_data.task_id,
            "simulation_time": sim_data.simulation_time,
            "node_count": len(sim_data.nodes),
            "nodes": node_info_list,
        }, indent=2)

    async def think(self, sim_data: SimulationData) -> str:
        """让智能体思考仿真数据并给出分析。

        Args:
            sim_data: 仿真数据

        Returns:
            智能体的分析结果
        """
        data_description = self._format_simulation_data(sim_data)
        print(f"[LLM format data Input]\n{data_description}", flush=True)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"请分析以下仿真数据并提供优化建议：\n\n{data_description}"},
        ]

        response = await self._call_llm(messages, tools=self.tools)
        return response.get("message", {}).get("content", "")

    async def process(self, sim_data: SimulationData) -> dict[str, Any]:
        """处理仿真数据，使用 LLM 分析并返回结果。

        Args:
            sim_data: 仿真数据

        Returns:
            处理结果字典
        """
        # 先获取基本处理结果
        result = {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "processed": True,
            "task_id": sim_data.task_id,
            "simulation_time": sim_data.simulation_time,
            "node_count": len(sim_data.nodes),
            "nodes": [
                {
                    "id": n.id,
                    "position": n.position,
                    "velocity": n.velocity,
                    "energy_percentage": n.energy_percentage,
                    "hello_interval": n.hello_interval,
                    "simulation_time": n.simulation_time,
                    "weight_distance": n.weight_distance,
                    "weight_link_time": n.weight_link_time,
                    "weight_rel_velocity": n.weight_rel_velocity,
                    "weight_neighbor_count": n.weight_neighbor_count,
                    "multi_path_count": n.multi_path_count,
                }
                for n in sim_data.nodes
            ],
        }

        # 使用 LLM 进行分析
        try:
            analysis = await self.think(sim_data)
            result["llm_analysis"] = analysis
        except Exception as e:
            result["llm_analysis"] = f"LLM 分析失败: {str(e)}"

        return result