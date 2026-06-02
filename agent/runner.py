"""Agent 运行器。

提供统一的入口来调用 agent 处理仿真数据。
"""

from __future__ import annotations

from typing import Any

from agent.agent import LlmAgent, LlmConfig
from agent.data_parser import parse_by_type
from agent.data_parser import SimulationData
from agent.data_parser import parse_scene_params_data


async def process_with_agent(
    data: dict[str, Any],
    llm_config: LlmConfig | None = None,
) -> dict[str, Any]:
    """处理仿真数据的主入口函数。

    Args:
        data: 从 ns3 推送过来的仿真数据字典
        llm_config: LLM 配置，包含 provider、model、base_url 等

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
        # 默认使用 qwen provider
        config = llm_config or {
            "provider": "qwen",
            "model": "qwen3.6-flash",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "sk-048bfae0242c43e283b6353b5291d2f0",
        }
        agent: LlmAgent = LlmAgent(
            agent_id="llm_agent",
            name="LLM Agent",
            llm_config=config,
        )
        try:
            result = await agent.process(sim_data)
        finally:
            await agent.close()
        return result


# 导出 process_with_agent 作为别名，方便外部调用
run = process_with_agent

__all__ = ["run", "process_with_agent", "LlmAgent"]