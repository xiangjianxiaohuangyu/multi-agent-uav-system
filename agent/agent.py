"""智能体处理模块。

负责处理从 ns3 推送过来的仿真数据，支持 LLM 推理和参数决策。
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from agent.data_parser import SimulationData
from agent.llm_providers import LlmProvider, LlmProviderFactory
from agent.tools import communication_tool


# ---------------------------------------------------------------------- #
# LLM 决策输出 schema（用 Pydantic 强校验）
# ---------------------------------------------------------------------- #


class RoutingDecision(BaseModel):
    """LLM 必须按此结构输出 JSON，作为本次调参的最终决策。

    - 4 个权重取值范围 [0, 1]，和不必为 1.0（系统会等比归一化）
    - multi_path_count 范围 [1, 4]
    - hello_interval 范围 [0.5, 5.0] 秒
    - analysis 字段为可读的决策理由
    """

    model_config = ConfigDict(extra="forbid")

    weight_distance: float = Field(ge=0.0, le=1.0)
    weight_link_time: float = Field(ge=0.0, le=1.0)
    weight_rel_velocity: float = Field(ge=0.0, le=1.0)
    weight_neighbor_count: float = Field(ge=0.0, le=1.0)
    multi_path_count: int = Field(ge=1, le=4)
    hello_interval: float = Field(ge=0.5, le=5.0)
    analysis: str = ""

    def normalized_weights(self) -> dict[str, float]:
        """把 4 个权重等比归一化到和=1.0；全 0 时退回等分。"""
        w = [
            self.weight_distance,
            self.weight_link_time,
            self.weight_rel_velocity,
            self.weight_neighbor_count,
        ]
        total = sum(w)
        if total <= 0:
            return {
                "weight_distance": 0.25,
                "weight_link_time": 0.25,
                "weight_rel_velocity": 0.25,
                "weight_neighbor_count": 0.25,
            }
        return {
            "weight_distance": w[0] / total,
            "weight_link_time": w[1] / total,
            "weight_rel_velocity": w[2] / total,
            "weight_neighbor_count": w[3] / total,
        }


class LlmConfig(TypedDict):
    """LLM 配置。"""
    provider: str
    model: str
    base_url: str | None
    api_key: str | None


DEFAULT_LLM_CONFIG: LlmConfig = {
    "provider": "ollama",
    "model": "llama3.1:8b",
    "base_url": "http://localhost:11434",
    "api_key": None,
}


class ToolCall(TypedDict):
    """工具调用。"""
    name: str
    arguments: dict[str, Any]


def _extract_assistant_content(response: dict[str, Any]) -> str:
    """从 LLM provider 返回的原始响应里抽取 assistant 文本 content。

    兼容两种格式：
    - OpenAI 风格：``{"choices": [{"message": {"content": "..."}}]}``
    - Ollama 风格：``{"message": {"role": "assistant", "content": "..."}}``（无 choices 包裹）

    抽取不到时返回空串，便于上层 fallback。
    """
    if not isinstance(response, dict):
        return ""
    # 1) OpenAI / Qwen 等
    choices = response.get("choices") or []
    if choices:
        first = choices[0] or {}
        msg = first.get("message") or {}
        content = msg.get("content")
        if content:
            return content
    # 2) Ollama 原生
    msg = response.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if content:
            return content
    return ""


class LlmAgent:
    """基于 LLM 的智能体，支持工具调用。"""

    def __init__(
        self,
        agent_id: str = "main_agent",
        name: str = "Main Agent",
        llm_config: LlmConfig | None = None,
        system_prompt: str | None = None,
    ):
        self.agent_id = agent_id
        self.name = name
        self.llm_config = llm_config or DEFAULT_LLM_CONFIG

        # 创建 LLM Provider
        provider = self.llm_config.get("provider", "ollama")
        model = self.llm_config.get("model", "")
        base_url = self.llm_config.get("base_url")
        api_key = self.llm_config.get("api_key")

        init_kwargs: dict[str, Any] = {"model": model}
        if base_url:
            init_kwargs["base_url"] = base_url
        if api_key:
            init_kwargs["api_key"] = api_key

        self.provider: LlmProvider = LlmProviderFactory.create(provider, **init_kwargs)

        # 暂时不使用工具
        self.tools = None
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _get_available_tools(self) -> list[dict[str, Any]]:
        """获取可用的工具列表。"""
        from agent.tools import get_available_tools
        return get_available_tools()

    def _default_system_prompt(self) -> str:
        """默认系统提示词。"""
        return """# Role & Objective
你是一个无人机动态自组织网络（FANET）的【自动化参数调优状态机】。

# 任务说明
你将收到源节点的完整仿真数据，包含【源节点自身指标、邻居聚合统计、当前参数、性能结果】四类信息，需要据此输出新的路由参数（4 个权重 + 多路径数 + Hello 间隔）。

# 输入数据结构
- source_node: 源节点自身指标
    - speed: 标量速度（单位 m/s，与 velocity 矢量不同）
    - energy_percentage: 剩余能量百分比 (0-100)
    - queue_length: 当前队列长度
    - neighbor_count: 直接邻居数量
    - distance_to_destination: 源到目的的直线距离
- neighbor_stats: 邻居聚合统计（mean / std / min / max）
    - forward_candidate_ratio: 邻居中能作为下一跳候选的比例
    - distance_to_me: 邻居到源的距离
    - distance_to_destination: 邻居到目的的距离
    - relative_speed: 邻居与源的相对速度
    - link_lifetime: 链路预期生存时间（秒）
    - neighbor_degree: 邻居自身的二度邻居数
    - queue_length: 邻居的拥塞程度
    - energy: 邻居的能量水平
- current_parameters: 当前生效的可调参数
- performance: 上一轮性能（avg_pdr 0-1、avg_delay 毫秒）

# 决策建议
- 节点速度快（source_node.speed > 8 或 neighbor_stats.relative_speed.mean > 8）→ 增大 weightLinkTime 和 weightRelVelocity，同时可减小 hello_interval
- 节点密集（source_node.neighbor_count >= 5 或 neighbor_stats.forward_candidate_ratio > 0.5）→ 增大 weightNeighborCount
- 丢包严重（performance.avg_pdr < 0.5）→ 增大 multi_path_count、weightNeighborCount
- 时延高（performance.avg_delay > 100 ms）→ 减小 hello_interval 加快拓扑更新
- 邻居稀疏（source_node.neighbor_count < 3）→ 增大 hello_interval 维持拓扑感知，weightNeighborCount 适当提升

# Output Format（严格遵守）
你必须且只能输出一个 JSON 对象（不要包含 ```、不要解释、不要多余文字）。结构如下：

{
  "weight_distance": float,        // [0, 1]
  "weight_link_time": float,       // [0, 1]
  "weight_rel_velocity": float,    // [0, 1]
  "weight_neighbor_count": float,  // [0, 1]
  "multi_path_count": int,         // 1~4
  "hello_interval": float,         // 0.5~5.0 秒
  "analysis": string               // 一句话简述决策理由，<= 200 字
}

4 个权重取值不需要相加为 1.0；系统会自动等比归一化。
任何字段超出规定范围都会被系统拒绝，请严格遵守。"""

    async def close(self):
        """关闭 provider 客户端。"""
        await self.provider.close()

    def _format_prior_experiences(
        self,
        prior_experiences: list[dict[str, Any]] | None,
    ) -> str:
        """把 FAISS 检索到的历史经验格式化为可读的 system prompt 段落。

        输入格式参考 ``ExperienceOut``：每条经验含 score / scene / parameter / result。

        输出形态（示例）：
            # Historical Experience (Top-K by score, scene-similarity)
            对于与当前场景相似的历史仿真，以下参数在历史上获得了最高的 score。
            请参考这些历史经验选择与当前场景最匹配的参数组合（score 越高越值得参考）。

              #1  score=0.872  pdr=0.950  delay=42.1ms  energy=12.34
                  parameter = {"hello_interval": 1.5, "multi_path_count": 2, ...}
              #2  score=0.731  pdr=0.910  delay=51.0ms  energy=15.20
                  parameter = {"hello_interval": 2.0, "multi_path_count": 3, ...}
        """
        # 1) 防御：检索为空时直接返回空串，让 _build_system_prompt 退化为仅基础 prompt
        if not prior_experiences:
            return ""

        # 2) 段落头：标题 + 引导 LLM 如何使用这些经验的指令
        lines: list[str] = [
            "",
            "# Historical Experience (Top-K by scene-similarity)",
            "对于与当前场景相似的历史仿真，以下参数被使用过。",
            "请参考这些历史经验选择与当前场景最匹配的参数组合（distance 越小越相似）。",
            "",
        ]

        # 3) 遍历每条经验（已由 FAISS 按 L2 距离 ASC 排序好），生成两行：
        #    - 摘要行：distance + 两个核心性能指标（pdr / delay）
        #    - 参数行：完整 parameter 字典（按 key 排序 + 保留中文以保可读性）
        for i, exp in enumerate(prior_experiences, start=1):
            # distance：FAISS L2 距离，越小越相似
            distance = exp.get("distance")
            # parameter：当时仿真的可调参数组合（权重 / 多路径数 / hello 间隔等）
            param = exp.get("parameter", {})
            # result：当时仿真跑出来的端到端性能指标
            # 注意：result 字典的键名严格对齐 NS3 扁平 payload —— ``avg_pdr`` / ``avg_delay``，
            # 与 experience/scoring.py:RESULT_FIELDS / schemas.py:Result 保持一致。
            # （历史版本曾误用 ``e2e_pdr`` / ``e2e_delay``，导致所有历史经验的性能都显示为 0）
            res = exp.get("result", {})

            # 摘要行：把关键指标压缩到一行，方便 LLM 横向对比
            distance_str = (
                f"{float(distance):.3f}" if distance is not None else "n/a"
            )
            lines.append(
                f"  #{i}  distance={distance_str}  "
                f"pdr={float(res.get('avg_pdr', 0)):.3f}  "                # 端到端投递率
                f"delay={float(res.get('avg_delay', 0)):.1f}ms  "          # 端到端时延（ms）
            )
            # 参数行：完整参数快照，便于 LLM 直接复现该组合
            lines.append(
                f"      parameter = {json.dumps(param, ensure_ascii=False, sort_keys=True)}"
            )

        # 4) 段落尾补一个空行，让拼接时与下游 system_prompt 段落之间有清晰分隔
        lines.append("")
        return "\n".join(lines)

    def _build_system_prompt(
        self,
        prior_experiences: list[dict[str, Any]] | None = None,
    ) -> str:
        """构造完整的 system prompt（基础 + 历史经验段落）。"""
        return self.system_prompt + self._format_prior_experiences(prior_experiences)

    def _execute_tool_call(self, tool_name: str, arguments: dict[str, Any], nodes: list[dict]) -> dict[str, Any]:
        """执行工具调用。

        Args:
            tool_name: 工具名称
            arguments: 工具参数
            nodes: 节点列表，工具会直接修改对应节点的值

        Returns:
            工具执行结果
        """
        tool_map = {
            "adjust_weights": communication_tool.adjust_weights,
            "adjust_multipath_count": communication_tool.adjust_multipath_count,
            "adjust_hello_interval": communication_tool.adjust_hello_interval,
        }

        if tool_name not in tool_map:
            return {
                "status": "error",
                "message": f"未知工具: {tool_name}"
            }

        try:
            tool_func = tool_map[tool_name]
            result = tool_func(**arguments)
            # 根据返回结果更新 nodes 中对应节点的值
            if result.get("status") == "success" and "data" in result:
                data = result["data"]
                node_id = data["node_id"]
                for node in nodes:
                    if node["id"] == node_id:
                        if tool_name == "adjust_weights":
                            weights = data["weights"]
                            node["weight_distance"] = weights["distance"]
                            node["weight_link_time"] = weights["linkTime"]
                            node["weight_rel_velocity"] = weights["relVelocity"]
                            node["weight_neighbor_count"] = weights["neighborCount"]
                        elif tool_name == "adjust_multipath_count":
                            node["multi_path_count"] = data["new_count"]
                        elif tool_name == "adjust_hello_interval":
                            node["hello_interval"] = data["new_interval"]
                        break
            return result
        except Exception as e:
            return {
                "status": "error",
                "message": f"工具执行失败: {str(e)}"
            }

    def _format_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: dict[str, Any],
    ) -> dict[str, str]:
        """格式化工具执行结果为 LLM 消息。

        Args:
            tool_call_id: 工具调用 ID
            tool_name: 工具名称
            result: 工具执行结果

        Returns:
            格式化的消息字典
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps({
                "tool": tool_name,
                "result": result,
            }, ensure_ascii=False),
        }

    def _format_simulation_data(
        self,
        sim_data: SimulationData,
        source_node_ids: list[int] | None = None,
        raw_data: dict[str, Any] | None = None,
    ) -> str:
        """格式化仿真数据为文本描述。

        优先使用 ``raw_data``（NS3 扁平 payload，含完整 23 字段指标），
        这样 LLM 能看到 ``speed``（标量）、``neighbor_stats``、``performance``
        等关键指标。仅当 ``raw_data`` 缺失时，才回退到 ``NodeInfo`` 的 4 字段
        旧格式（用于单元测试 / 离线调用场景）。

        Args:
            sim_data: 仿真数据（解析后的结构化对象）
            source_node_ids: 保留参数以保持 API 兼容；NS3 不再传 node_id 后不再用于过滤
            raw_data: NS3 推送的原始扁平 payload（顶层字段，type=simulation）

        Returns:
            格式化的 JSON 字符串
        """
        if raw_data and isinstance(raw_data, dict) and raw_data.get("type") != "scene_params":
            return self._format_from_raw(raw_data)

        # NS3 不再传 node_id 后无法按 ID 过滤，直接处理全部节点
        filtered_nodes = sim_data.nodes

        node_info_list = []
        for idx, n in enumerate(filtered_nodes):
            weights = {
                "distance": n.weight_distance,
                "linkTime": n.weight_link_time,
                "relVelocity": n.weight_rel_velocity,
                "neighborCount": n.weight_neighbor_count,
            }
            node_info_list.append({
                "index": idx,
                "position": n.position,
                "velocity": n.velocity,
                "energy_percentage": n.energy_percentage,
                "weights": weights,
            })

        return json.dumps({
            "source_node_count": len(filtered_nodes),
            "source_nodes": node_info_list,
        }, indent=2)

    def _format_from_raw(self, data: dict[str, Any]) -> str:
        """从 NS3 扁平 payload 直接格式化所有指标。

        NS3 实际推送的字段（已在 NS3 侧对齐，本函数不再做猜测）：

        源节点自身 (source_node):
            speed, energy, queue_length, neighbor_count, distance_to_destination

        邻居统计 (neighbor_stats):
            forward_candidate_ratio,
            distance_to_me_mean/std,
            distance_to_destination_mean/std/min,
            relative_speed_mean/std,
            link_lifetime_mean/std,
            neighbor_degree_mean/std,
            queue_length_mean/std/max,
            energy_mean/std/min

        当前参数 (current_parameters):
            hello_interval, path_num,
            w_distance, w_linkTime, w_relVelocity, w_neighborCount

        性能结果 (performance):
            avg_pdr (0-1), avg_delay (ms)
        """
        def fnum(key: str) -> float:
            v = data.get(key)
            return float(v) if v is not None else 0.0

        def inum(key: str) -> int:
            v = data.get(key)
            return int(v) if v is not None else 0

        payload = {
            "source_node": {
                "speed": fnum("speed"),
                "energy_percentage": fnum("energy"),
                "queue_length": inum("queue_length"),
                "neighbor_count": inum("neighbor_count"),
                "distance_to_destination": fnum("distance_to_destination"),
            },
            "neighbor_stats": {
                "forward_candidate_ratio": fnum("forward_candidate_ratio"),
                "distance_to_me": {
                    "mean": fnum("distance_to_me_mean"),
                    "std": fnum("distance_to_me_std"),
                },
                "distance_to_destination": {
                    "mean": fnum("distance_to_destination_mean"),
                    "std": fnum("distance_to_destination_std"),
                    "min": fnum("distance_to_destination_min"),
                },
                "relative_speed": {
                    "mean": fnum("relative_speed_mean"),
                    "std": fnum("relative_speed_std"),
                },
                "link_lifetime": {
                    "mean": fnum("link_lifetime_mean"),
                    "std": fnum("link_lifetime_std"),
                },
                "neighbor_degree": {
                    "mean": fnum("neighbor_degree_mean"),
                    "std": fnum("neighbor_degree_std"),
                },
                "queue_length": {
                    "mean": fnum("queue_length_mean"),
                    "std": fnum("queue_length_std"),
                    "max": inum("queue_length_max"),
                },
                "energy": {
                    "mean": fnum("energy_mean"),
                    "std": fnum("energy_std"),
                    "min": fnum("energy_min"),
                },
            },
            "current_parameters": {
                "hello_interval": fnum("hello_interval"),
                "path_num": inum("path_num"),
                "w_distance": fnum("w_distance"),
                "w_linkTime": fnum("w_linkTime"),
                "w_relVelocity": fnum("w_relVelocity"),
                "w_neighborCount": fnum("w_neighborCount"),
            },
            "performance": {
                "avg_pdr": fnum("avg_pdr"),
                "avg_delay_ms": fnum("avg_delay"),
            },
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    async def think_with_loop(
        self,
        sim_data: SimulationData,
        source_node_ids: list[int] | None = None,
        prior_experiences: list[dict[str, Any]] | None = None,
        raw_data: dict[str, Any] | None = None,
    ) -> str:
        """让智能体单轮思考仿真数据。

        Args:
            sim_data: 仿真数据
            source_node_ids: 保留参数以保持 API 兼容；NS3 不再传 node_id 后不再用于过滤
            prior_experiences: FAISS 经验库检索到的历史经验（注入到 system prompt）
            raw_data: NS3 原始扁平 payload，含完整 23 字段指标（推荐传入）

        Returns:
            智能体的分析结果
        """
        data_description = self._format_simulation_data(sim_data, source_node_ids, raw_data=raw_data)
        print(f"[LLM format data Input]\n{data_description}", flush=True)

        system_prompt = self._build_system_prompt(prior_experiences)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请分析以下源节点仿真数据并优化路由参数：\n\n{data_description}"},
        ]

        # 单次非流式调用（不传 tools）
        response = await self.provider.chat(messages)

        # 提取 assistant 消息（Qwen API 响应在 choices[0].message 中）
        choices = response.get("choices", [])
        if not choices:
            print(f"[LLM] No choices in response.", flush=True)
            return ""

        message_content = choices[0].get("message", {})
        return message_content.get("content", "")

    async def process(
        self,
        sim_data: SimulationData,
        prior_experiences: list[dict[str, Any]] | None = None,
        raw_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """处理仿真数据，使用 LLM 分析并返回结果。

        Args:
            sim_data: 仿真数据
            prior_experiences: FAISS 经验库检索到的历史经验，会被注入到 LLM system prompt
            raw_data: NS3 原始扁平 payload，含完整 23 字段指标（推荐传入以让 LLM 看到
                speed / neighbor_stats / performance 等关键信息）

        Returns:
            处理结果字典
        """
        # 防御：parser 把 sim_data 解析成空 nodes 时，直接返回带 code 的错误结果，
        # 避免 ``sim_data.nodes[0]`` 抛 IndexError 把整次 callback 拖崩。
        if not sim_data.nodes:
            return {
                "code": -1,
                "status": "error",
                "msg": "sim_data.nodes is empty; agent skipped",
                "weight_distance": 0.0,
                "weight_link_time": 0.0,
                "weight_rel_velocity": 0.0,
                "weight_neighbor_count": 0.0,
                "multi_path_count": 0,
                "hello_interval": 0.0,
                "llm_analysis": "skipped: empty nodes",
            }

        # 先获取基本处理结果
        n = sim_data.nodes[0]
        result = {
            "weight_distance": n.weight_distance,
            "weight_link_time": n.weight_link_time,
            "weight_rel_velocity": n.weight_rel_velocity,
            "weight_neighbor_count": n.weight_neighbor_count,
            "multi_path_count": n.multi_path_count,
            "hello_interval": n.hello_interval,
        }

        # 直接进行 RAG 分析 + JSON 决策解析
        try:
            print(f"[Process] 处理源节点 0", flush=True)
            data_description = self._format_simulation_data(sim_data, raw_data=raw_data)
            print(f"[LLM format data Input]\n{data_description}", flush=True)

            system_prompt = self._build_system_prompt(prior_experiences)
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"请分析以下源节点（node_id=0）仿真数据并按规定的 JSON 格式输出新的路由参数：\n\n{data_description}"
                    ),
                },
            ]

            # 单次非流式调用：不传 tools，让 LLM 直接输出 JSON 文本
            response = await self.provider.chat(messages)
            raw_text = _extract_assistant_content(response) or ""

            # 解析 LLM 文本 → Pydantic RoutingDecision；失败时安全回退
            decision = self._parse_routing_decision(raw_text)
            if decision is not None:
                w = decision.normalized_weights()
                result["weight_distance"] = w["weight_distance"]
                result["weight_link_time"] = w["weight_link_time"]
                result["weight_rel_velocity"] = w["weight_rel_velocity"]
                result["weight_neighbor_count"] = w["weight_neighbor_count"]
                result["multi_path_count"] = decision.multi_path_count
                result["hello_interval"] = decision.hello_interval
                result["llm_analysis"] = decision.analysis or raw_text
                result["code"] = 0
            else:
                # 解析失败：保留仿真原始值，把 LLM 原始输出塞到 analysis 里便于排查
                result["llm_analysis"] = (
                    f"LLM 输出无法解析为 RoutingDecision，保留原值。原始输出: {raw_text[:500]}"
                )
                result["code"] = 1
        except Exception as e:
            result["llm_analysis"] = f"LLM 分析失败: {str(e)}"
            result["code"] = -1

        return result

    def _parse_routing_decision(self, text: str) -> RoutingDecision | None:
        """从 LLM 文本中抽取并校验 RoutingDecision JSON。

        兼容 LLM 常见的"啰嗦"输出：
        - 纯 JSON
        - ```json ... ``` 包裹
        - JSON 前后带自然语言（用正则抓取第一个 {...} 块）
        - 末尾可能带 ``}``` 这类尾巴
        """
        if not text:
            return None
        candidate = self._extract_json_block(text)
        if candidate is None:
            return None
        try:
            return RoutingDecision.model_validate_json(candidate)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _extract_json_block(text: str) -> str | None:
        """从 LLM 输出里抽出第一个 JSON 对象文本。

        优先级：
        1) 显式 ```json ... ``` 代码块
        2) 第一个 { 到最后一个 } 的区间
        """
        if not text:
            return None
        # 1) markdown code fence
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            return fence.group(1)
        # 2) 抓取第一个 { 到最后一个 } 的内容
        start = text.find("{")
        if start == -1:
            return None
        end = text.rfind("}")
        if end == -1 or end <= start:
            return None
        return text[start : end + 1]