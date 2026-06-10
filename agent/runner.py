"""Agent 运行器。

提供统一的入口来调用 agent 处理仿真数据。
"""

from __future__ import annotations

from typing import Any

from agent.agent import LlmAgent
from agent.data_parser import parse_by_type
from agent.data_parser import SimulationData
from agent.data_parser import parse_scene_params_data
from agent.llm_providers import LlmProviderFactory, PresetName


async def process_with_agent(
    data: dict[str, Any],
    preset: PresetName | None = None,
    prior_experiences: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """处理仿真数据的主入口函数。

    Args:
        data: 从 ns3 推送过来的仿真数据字典
        preset: 预设的 LLM 名称（如 "use_ollama_qwen3_4b"），None 时使用默认
        prior_experiences: FAISS 经验库检索到的历史经验，会被注入到 LLM system prompt

    Returns:
        包含处理结果的字典

    Raises:
        ValueError: preset 名称不存在（不会回退到默认）
    """
    msg_type, parsed_data = parse_by_type(data)
    print(f"[Parser] type: {msg_type}", flush=True)
    print()

    if msg_type == "scene_params":
        parse_scene_params_data(data)
        return {"code": 0, "msg": "ok"}
    else:
        sim_data: SimulationData = parsed_data

        if preset:
            # 显式指定了 preset → 查找，找不到就报错（不回退到默认）
            config = LlmProviderFactory.get_preset(preset)
        else:
            # 未指定 → 使用默认预设
            config = LlmProviderFactory.use_ollama_llama3_1_8b()
            print("[Runner] 未指定 preset，使用默认: use_ollama_llama3_1_8b", flush=True)

        agent: LlmAgent = LlmAgent(
            agent_id="llm_agent",
            name="LLM Agent",
            llm_config=config,
        )
        try:
            result = await agent.process(sim_data, prior_experiences=prior_experiences)
        finally:
            await agent.close()
        return result


# 导出 process_with_agent 作为别名，方便外部调用
run = process_with_agent

__all__ = ["run", "process_with_agent", "LlmAgent"]
