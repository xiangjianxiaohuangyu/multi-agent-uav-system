"""Agent 运行器。

提供统一的入口来调用 agent 处理仿真数据。
"""

from __future__ import annotations

from typing import Any

from agent.agent import LlmAgent
from agent.data_parser import parse_by_type
from agent.data_parser import SimulationData
from agent.data_parser import parse_scene_params_data


async def process_with_agent(
    data: dict[str, Any],
    ollama_config: dict[str, str] | None = None,
) -> dict[str, Any]:
    """处理仿真数据的主入口函数。

    Args:
        data: 从 ns3 推送过来的仿真数据字典
        ollama_config: Ollama 配置，包含 base_url 和 model

    Returns:
        包含处理结果的字典
    """
    msg_type, parsed_data = parse_by_type(data)
    print(f"[Parser] type: {msg_type}", flush=True)
    print()

    if msg_type == "scene_params":
        parse_scene_params_data(data)
        return {"code": 0, "msg": "ok"}
    else:
        sim_data: SimulationData = parsed_data
        llm_config = ollama_config or {"base_url": "http://localhost:11434", "model": "qwen3:4b"}
        agent: LlmAgent = LlmAgent(
            agent_id="llm_agent",
            name="LLM Agent",
            ollama_config=llm_config,
        )
        try:
            result = await agent.process(sim_data)
        finally:
            await agent.close()
        return result


# 导出 process_with_agent 作为别名，方便外部调用
run = process_with_agent

__all__ = ["run", "process_with_agent", "LlmAgent"]