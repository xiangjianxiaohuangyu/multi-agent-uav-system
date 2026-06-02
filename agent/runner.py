"""Agent 运行器。

提供统一的入口来调用 agent 处理仿真数据。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.agent import LlmAgent, LlmConfig
from agent.data_parser import parse_by_type
from agent.data_parser import SimulationData
from agent.data_parser import parse_scene_params_data


def _load_api_key() -> str:
    """从 secrets/api_key.txt 读取 API key."""
    secrets_path = Path(__file__).parent.parent / "secrets" / "api_key.txt"
    if secrets_path.exists():
        return secrets_path.read_text().strip()
    raise FileNotFoundError(f"API key file not found: {secrets_path}")


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
            "api_key": _load_api_key(),
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