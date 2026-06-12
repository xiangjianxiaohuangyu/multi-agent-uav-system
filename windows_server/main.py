"""Windows 端纯推送服务。

职责：
1. 暴露 `/api/simulation/callback`，接收 ns3 主动推送的仿真状态、节点数据和结果。
2. 在回调中：
   - 保存原始 JSON；
   - 写入经验库（MySQL + FAISS，含 StandardScaler 归一化）；
   - 用相似场景检索 Top-K 历史经验；
   - 调用 LLM Agent（带 prior_experiences）生成参数决策。

注意：
- ns3 回调 Windows 时不能使用 `localhost`，必须使用 ns3 能访问到的 Windows IP。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

# 加载项目根 .env（在所有 os.environ.get 之前调用，且 override=False 不覆盖已有 shell 变量）
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from agent.runner import process_with_agent
from agent.test_local import save_raw_data
from db.config import is_mysql_disabled
from db.repository import write_simulation_record
from experience.engine import get_config as get_experience_config
from experience.router import router as experience_router
from experience.repository import ExperienceRepository
from experience.scoring import payload_to_experience

# 关闭 uvicorn 默认的 access log，避免在 callback 中打印大量噪声
log = logging.getLogger("windows_server")
log.setLevel(os.environ.get("WINDOWS_SERVER_LOG_LEVEL", "INFO"))

# 运行模式开关：
#   - "store"      : 只存储数据（保存原始 JSON + 写入经验库 + 写入 simulation_records），不调用 Agent
#   - "agent"      : 存储数据 + 调用 LLM Agent 进行决策
#   - "agent_only" : 不存储数据，只调用 LLM Agent（适合"有数据但只想推理"的场景）
#   - "disable"    : 跳过存储与 Agent，仅做日志
# 通过环境变量 WINDOWS_SERVER_RUN_MODE 切换，默认 "agent"。
RUN_MODE = os.environ.get("WINDOWS_SERVER_RUN_MODE", "agent_only").lower()
_VALID_RUN_MODES = {"store", "agent", "agent_only", "disable"}
if RUN_MODE not in _VALID_RUN_MODES:
    log.warning("invalid_run_mode_fallback got=%s fallback=agent", RUN_MODE)
    RUN_MODE = "agent_only"


# FastAPI 应用实例。运行命令：
# uvicorn windows_server.main:app --host 0.0.0.0 --port 8000 --reload
app = FastAPI(title="Windows Push Server", version="1.1.0")

# 挂载经验库 REST API（5 个原端点 + 1 个 admin/rebuild）
app.include_router(experience_router, prefix="/experience")

# 进程级经验库仓储单例
_experience_repo = ExperienceRepository()


def _log(level: str, message: str, data: dict[str, Any] | None = None) -> None:
    """输出结构化 JSON 日志。

    使用 JSON 日志比普通 print 更方便后续接入日志采集系统。
    """
    payload: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "level": level,
        "component": "windows_server",
        "message": message,
    }
    if data is not None:
        payload["data"] = data
    print(json.dumps(payload, ensure_ascii=False), flush=True)


@app.post("/api/simulation/callback")
async def receive_simulation_data(request: Request) -> dict[str, Any]:
    """接收 ns3 主动推送的仿真数据。

    数据流：
    ns3 仿真进程 -> Windows `/api/simulation/callback`
                  -> save_raw_data
                  -> ExperienceRepository.add_from_payload (MySQL + FAISS)
                  -> ExperienceRepository.search            (Top-K 历史经验)
                  -> LlmAgent.process(data, prior_experiences=hits)

    异常隔离：
    - 经验库 / LLM 任何一步失败都只记 warning，不影响 HTTP 200。
    - LLM 失败时返回的 agent_result 标记为 skipped，但 callback 仍然成功。
    """
    try:
        raw_payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="request body is not valid JSON") from exc

    if not isinstance(raw_payload, dict):
        raise HTTPException(status_code=400, detail="callback body must be a JSON object")

    task_id = raw_payload.get("task_id", "unknown")
    _log("info", "callback received", {"task_id": task_id, "run_mode": RUN_MODE})

    # 透出 NS3 推送的完整 raw_payload（debug 级别；_log 始终 print，方便排查字段名）
    _log(
        "debug",
        "ns3_raw_payload",
        {
            "task_id": task_id,
            "payload_type": raw_payload.get("type"),
            "payload_keys": list(raw_payload.keys()),
            "payload": raw_payload,
        },
    )

    # 根据 RUN_MODE 决定：是否进入"存储数据"分支 / "调用 agent" 分支
    should_store = RUN_MODE in {"store", "agent"}
    should_call_agent = RUN_MODE in {"agent", "agent_only"}
    # 仅"调用 agent"时才需要 RAG 检索（包括 agent_only：不写库但要读历史经验）
    should_read_experience = should_call_agent

    sim_record_ok: bool = False  # 在 if 之外先占位，避免 disable 模式下被读取时未定义

    # 1) 保存原始数据用于本地测试 / 回放
    if should_store:
        try:
            saved_path = save_raw_data(raw_payload)
            _log("debug", "raw_data_saved", {"task_id": task_id, "path": saved_path})
        except Exception as exc:  # noqa: BLE001
            _log("warning", "save_raw_data_failed", {"task_id": task_id, "error": str(exc)})

        # 1.5) 写入 simulation_records 扁平化明细（best-effort）
        try:
            sim_record_ok = write_simulation_record(raw_payload)
            if not sim_record_ok:
                # 区分"被 disabled"与"payload 不合法"，便于排查
                if is_mysql_disabled():
                    _log("warning", "simulation_record_skipped", {"task_id": task_id, "reason": "mysql_disabled"})
                else:
                    _log(
                        "warning",
                        "simulation_record_skipped",
                        {"task_id": task_id, "reason": "payload_invalid_or_empty",
                         "payload_type": raw_payload.get("type"),
                         "payload_keys": list(raw_payload.keys())},
                    )
        except Exception as exc:  # noqa: BLE001
            _log("warning", "simulation_record_failed", {"task_id": task_id, "error": str(exc)})

    # 2) 经验库：写入 + 检索 拆成两个独立分支（best-effort，永不抛）
    # ------------------------------------------------------------------
    # 数据流：
    #   raw_payload
    #     ├─► [2a 写入，仅 store/agent]  add_from_payload
    #     │       → MySQL 写经验行 + FAISS 写入向量
    #     │       → 返回新自增 experience_id
    #     └─► [2b 检索，仅 agent/agent_only]  payload_to_experience
    #             → 解析出 scene_dict（与写入共用映射）
    #             → search(scene, k=topk_default)
    #             → 23 维向量 → FAISS L2 TopK → MySQL 批量取行
    #             → 按 L2 距离 ASC 重排 → prior_experiences
    # 注意：add_from_payload 内部已经做一次 payload_to_experience；这里
    # 检索路径再解析一次，是为了拿到 scene_dict（add 路径只把 scene 存进去，
    # 不返回结构化对象）。
    # ------------------------------------------------------------------
    experience_id: int | None = None
    prior_experiences: list[dict[str, Any]] = []

    # 2a) 经验库写入：仅 store / agent 进入；agent_only / disable 跳过
    if should_store:
        try:
            cfg = get_experience_config()
            if cfg.is_disabled():
                _log("warning", "experience_disabled", {"task_id": task_id})
            else:
                experience_id = _experience_repo.add_from_payload(raw_payload)
                if experience_id is None:
                    # payload_to_experience 内部判定为不可入库（scene_params / payload 为空）
                    _log(
                        "warning",
                        "experience_add_skipped",
                        {
                            "task_id": task_id,
                            "payload_type": raw_payload.get("type"),
                            "payload_keys": list(raw_payload.keys()),
                        },
                    )
        except Exception as exc:  # noqa: BLE001
            _log("warning", "experience_add_failed", {"task_id": task_id, "error": str(exc)})

    # 2b) 经验库检索：仅 agent / agent_only 进入；store / disable 跳过
    #     agent_only 模式：不写库，但仍读取 Top-K 历史经验注入 RAG 上下文
    if should_read_experience:
        try:
            cfg = get_experience_config()
            if cfg.is_disabled():
                _log("warning", "experience_disabled", {"task_id": task_id})
            else:
                # 返回 (scene_dict, param_dict, result_dict)；任一块缺失则 None
                mapped = payload_to_experience(raw_payload)
                if mapped is not None:
                    # 检索只需要 scene，丢弃 param / result
                    scene_dict, _, _ = mapped
                    # TopK 相似场景经验；返回结果已按 L2 距离 ASC 排序
                    prior_experiences = _experience_repo.search(
                        scene_dict, k=cfg.topk_default,
                    )
                    # 输出 top-k 明细（debug 级别；降级日志可看）
                    _log(
                        "debug",
                        "rag_topk",
                        {
                            "task_id": task_id,
                            "k": cfg.topk_default,
                            "topk": [
                                {
                                    "rank": i + 1,
                                    "distance": exp.get("distance"),
                                    "experience_id": exp.get("experience_id"),
                                    "avg_pdr": exp.get("result", {}).get("avg_pdr"),
                                    "avg_delay_ms": exp.get("result", {}).get("avg_delay"),
                                    "parameter": exp.get("parameter", {}),
                                }
                                for i, exp in enumerate(prior_experiences)
                            ],
                        },
                    )
                else:
                    _log(
                        "warning",
                        "experience_search_skipped",
                        {
                            "task_id": task_id,
                            "reason": "payload_to_experience returned None",
                        },
                    )
        except Exception as exc:  # noqa: BLE001
            _log("warning", "experience_search_failed", {"task_id": task_id, "error": str(exc)})

    _log(
        "info",
        "experience_lookup",
        {
            "task_id": task_id,
            "added_eid": experience_id,
            "prior_count": len(prior_experiences),
            "sim_record_written": sim_record_ok,
        },
    )

    # 3) LLM Agent 决策（带 prior_experiences）
    agent_result: dict[str, Any] = {"processed": False, "status": "skipped"}
    if should_call_agent:
        try:
            agent_result = await process_with_agent(
                raw_payload, prior_experiences=prior_experiences
            )
            agent_result.setdefault("processed", True)
            _log(
                "info",
                "agent_processed",
                {
                    "task_id": task_id,
                    "prior_count": len(prior_experiences),
                    "code": agent_result.get("code"),
                },
            )
        except Exception as exc:  # noqa: BLE001
            _log("warning", "agent_processing_failed", {"task_id": task_id, "error": str(exc)})

        # 透出 LLM 原始输出（agent_result 完整内容；含 llm_analysis / weights / 工具调用等）
        _log(
            "debug",
            "llm_raw_result",
            {
                "task_id": task_id,
                "result": agent_result,
            },
        )

    # 4) 模拟 Windows 端算法处理耗时（1s 物理时间）
    await asyncio.sleep(1)

    # 5) 把 LLM 决策后的"新参数"返回给 NS3
    #    字段名严格对齐 NS3 扁平 payload（w_distance / w_linkTime / w_relVelocity / w_neighborCount
    #    / path_num / hello_interval），NS3 端可直接落盘 / 应用
    #    - agent_result 里的 key 是内部 Python 命名（weight_* / multi_path_count）
    #    - 转换为 NS3 wire 格式时再映射一次
    new_parameters: dict[str, Any] | None = None
    if should_call_agent and agent_result.get("processed"):
        try:
            new_parameters = {
                "w_distance": float(agent_result.get("weight_distance", 0.0)),
                "w_linkTime": float(agent_result.get("weight_link_time", 0.0)),
                "w_relVelocity": float(agent_result.get("weight_rel_velocity", 0.0)),
                "w_neighborCount": float(agent_result.get("weight_neighbor_count", 0.0)),
                "path_num": int(agent_result.get("multi_path_count", 0)),
                "hello_interval": float(agent_result.get("hello_interval", 0.0)),
            }
        except (TypeError, ValueError) as exc:  # noqa: BLE001
            _log("warning", "extract_new_parameters_failed", {"task_id": task_id, "error": str(exc)})
            new_parameters = None

    return {
        "status": "success",
        "message": "Data processed by Windows",
        "task_id": task_id,
        "code": agent_result.get("code", 0) if should_call_agent else 0,
        "experience_id": experience_id,
        "prior_experiences_used": len(prior_experiences),
        "simulation_record_written": sim_record_ok,
        "agent_processed": bool(agent_result.get("processed", False)),
        # ★ 关键：把 LLM 决策的新参数返回给 NS3
        "new_parameters": new_parameters,
        "llm_analysis": agent_result.get("llm_analysis", ""),
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    # 直接运行 python windows_server/main.py 时使用。
    # 正式调试更推荐 uvicorn 命令，便于开启 reload。
    print("Windows backend starting on port 8000...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
