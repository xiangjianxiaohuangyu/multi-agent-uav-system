"""智能体处理模块。

负责处理从 ns3 推送过来的仿真数据，支持 LLM 推理和工具调用。
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from agent.data_parser import SimulationData, get_data_store
from agent.llm_providers import LlmProvider, LlmProviderFactory
from agent.tools import communication_tool


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


class LlmAgent:
    """基于 LLM 的智能体，支持工具调用。"""

    def __init__(
        self,
        agent_id: str = "main_agent",
        name: str = "Main Agent",
        llm_config: LlmConfig | None = None,
        tools: list[dict[str, Any]] | None = None,
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

        self.tools = tools or self._get_available_tools()
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
你将收到源节点的仿真数据，需要根据节点的速度、位置、能量等信息，优化路由协议的选择权重。

# 可调参数说明
1. adjust_weights - 路由选择权重优化
   - weightDistance: 距离权重，影响基于欧氏距离的路由选择
   - weightLinkTime: 链路生存时间权重，影响链路的稳定性和持续时间预测
   - weightRelVelocity: 相对速度权重，影响高速移动场景下的路由适应性
   - weightNeighborCount: 邻居节点数量权重，影响基于网络拓扑密度的路由选择

2. adjust_multipath_count - 多路径数量控制
   - 控制多路径备份链路数量（范围 1~4）
   - 网络拥堵、单路径断连频繁、丢包率上升时增大此数值

3. adjust_hello_interval - Hello 报文间隔调节
   - 调节邻居发现频率（范围 0.5~5.0 秒）
   - 节点速度快时减小间隔防止拓扑过期；静止或拥堵时增大间隔优化开销

# 决策建议
- 节点速度高（velocity > 8）→ 增大 weightLinkTime 和 weightRelVelocity，同时可减小 hello_interval
- 节点密集/信道拥堵 → 增大 weightNeighborCount，可增大 hello_interval 和 multipath_count
- 权重参数总和必须为 1.0

# Output Format
调用适当的工具返回优化结果。
注意：针对同一组输入仿真数据，只需调用一次工具即可。调用成功后，请直接总结你的优化思路并结束对话，切勿重复调用相同参数。"""

    async def close(self):
        """关闭 provider 客户端。"""
        await self.provider.close()

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
    ) -> str:
        """格式化仿真数据为文本描述。

        Args:
            sim_data: 仿真数据
            source_node_ids: 源节点 ID 列表，如果为 None 则格式化所有节点

        Returns:
            格式化的 JSON 字符串
        """
        # 过滤源节点（如果指定了 source_node_ids）
        if source_node_ids is not None:
            filtered_nodes = [n for n in sim_data.nodes if n.id in source_node_ids]
        else:
            filtered_nodes = sim_data.nodes

        node_info_list = []
        for n in filtered_nodes:
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
            })

        return json.dumps({
            "task_id": sim_data.task_id,
            "simulation_time": sim_data.simulation_time,
            "source_node_count": len(filtered_nodes),
            "source_nodes": node_info_list,
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

        response = await self.provider.chat(messages, tools=self.tools)
        return response.get("message", {}).get("content", "")

    async def think_with_loop(
        self,
        sim_data: SimulationData,
        nodes: list[dict],
        source_node_ids: list[int] | None = None,
        max_iterations: int = 5,
    ) -> str:
        """让智能体循环思考仿真数据，支持工具调用反馈。

        实现 ReAct 模式：LLM 生成响应 -> 执行工具 -> 将结果反馈给 LLM -> 再次决策

        Args:
            sim_data: 仿真数据
            nodes: 节点列表，工具调用会直接修改对应节点的值
            source_node_ids: 源节点 ID 列表
            max_iterations: 最大迭代次数

        Returns:
            智能体的最终分析结果
        """
        data_description = self._format_simulation_data(sim_data, source_node_ids)
        print(f"[LLM format data Input]\n{data_description}", flush=True)

        # 提供所有工具
        available_tools = self.tools

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"请分析以下源节点仿真数据并优化路由参数：\n\n{data_description}"},
        ]

        iteration = 0
        final_response = ""

        while iteration < max_iterations:
            iteration += 1
            print(f"[LLM Loop] Iteration {iteration}/{max_iterations}", flush=True)

            # 使用非流式调用
            response = await self.provider.chat(messages, tools=available_tools)

            # 提取 assistant 消息（Qwen API 响应在 choices[0].message 中）
            choices = response.get("choices", [])
            if not choices:
                print(f"[LLM Loop] No choices in response, stopping.", flush=True)
                break
            message_content = choices[0].get("message", {})
            final_response = message_content.get("content", "")

            # 检查是否有工具调用
            tool_calls = message_content.get("tool_calls")
            if not tool_calls:
                print(f"[LLM Loop] No more tool calls, stopping.", flush=True)
                break

            # 将 Assistant 消息（包含 tool_calls）添加到历史记录
            messages.append(message_content)

            print(f"[LLM Loop] Found {len(tool_calls)} tool call(s)", flush=True)

            # 执行每个工具调用并将结果反馈给 LLM
            for tool_call in tool_calls:
                tool_id = tool_call.get("id", "")
                tool_name = tool_call.get("function", {}).get("name", "")
                tool_args = tool_call.get("function", {}).get("arguments", {})

                # 安全解析参数字符串为字典
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError as e:
                        print(f"[LLM Loop] Error decoding tool arguments: {e}", flush=True)
                        result = {"status": "error", "message": f"Invalid JSON arguments: {str(e)}"}
                        tool_message = self._format_tool_result(tool_id, tool_name, result)
                        messages.append(tool_message)
                        continue

                print(f"[LLM Loop] Executing tool: {tool_name}", flush=True)

                # 执行工具（传入 nodes 以便直接修改）
                result = self._execute_tool_call(tool_name, tool_args, nodes)

                # 格式化工具结果并添加到消息历史
                tool_message = self._format_tool_result(tool_id, tool_name, result)
                messages.append(tool_message)

                print(f"[LLM Loop] Tool result: {result}", flush=True)

        return final_response

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

        # 从 DataStore 获取最新的场景参数（包含通信对信息）
        data_store = get_data_store()
        latest_scene_params = None
        if data_store.scene_params_history:
            latest_scene_params = data_store.scene_params_history[-1]

        # 提取源节点 ID 列表
        source_node_ids = []
        if latest_scene_params:
            source_node_ids = [pair.source for pair in latest_scene_params.communication_pairs]

        # 遍历每个源节点，单独调用 LLM 进行分析
        analysis_results = {}
        try:
            for source_id in source_node_ids:
                # 每次只传递单个源节点的 ID
                single_source_ids = [source_id]
                print(f"[Process] 处理源节点: {source_id}", flush=True)
                analysis = await self.think_with_loop(
                    sim_data, result["nodes"], source_node_ids=single_source_ids
                )
                analysis_results[str(source_id)] = analysis
            result["llm_analysis"] = analysis_results
        except Exception as e:
            result["llm_analysis"] = f"LLM 分析失败: {str(e)}"

        return result