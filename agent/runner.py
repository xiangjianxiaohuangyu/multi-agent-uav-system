"""Agent 运行器。

提供统一的入口来调用 agent 处理仿真数据。
"""

from __future__ import annotations

from typing import Any

from agent.agent import Agent, LlmAgent
from agent.data_parser import parse_by_type
from agent.data_parser import SimulationData
from agent.data_parser import SceneParamsData


async def process_with_agent(
    data: dict[str, Any],
    use_llm: bool = False,
    ollama_config: dict[str, str] | None = None,
) -> dict[str, Any]:
    """处理仿真数据的主入口函数。

    Args:
        data: 从 ns3 推送过来的仿真数据字典
        use_llm: 是否使用 LLM 智能体进行分析
        ollama_config: Ollama 配置，包含 base_url 和 model

    Returns:
        包含处理结果的字典
    """
    msg_type, parsed_data = parse_by_type(data)
    print(f"[Agent] type: {msg_type}, data: {parsed_data}", flush=True)
    print()

    if msg_type == "scene_params":
        scene_params: SceneParamsData = parsed_data
        return {
            "agent_id": "main_agent",
            "processed": True,
            "type": msg_type,
            "task_id": scene_params.task_id,
            "node_count": scene_params.node_count,
            "communication_pairs": [
                {"source": p.source, "destination": p.destination}
                for p in scene_params.communication_pairs
            ],
        }
    else:
        sim_data: SimulationData = parsed_data

        if use_llm:
            llm_config = ollama_config or {"base_url": "http://localhost:11434", "model": "qwen3:4b"}
            agent: Agent | LlmAgent = LlmAgent(
                agent_id="llm_agent",
                name="LLM Agent",
                ollama_config=llm_config,
            )
            try:
                result = await agent.process(sim_data)
            finally:
                await agent.close()
        else:
            agent = Agent()
            result = await agent.process(sim_data)

        result["type"] = msg_type
        return result


# 导出 process_with_agent 作为别名，方便外部调用
run = process_with_agent

__all__ = ["run", "process_with_agent", "Agent", "LlmAgent"]