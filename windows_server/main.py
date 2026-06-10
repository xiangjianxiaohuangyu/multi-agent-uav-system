"""Windows 端纯推送服务。

职责：
1. 暴露 `/api/simulation/callback`，接收 ns3 主动推送的仿真状态、节点数据和结果。

注意：
- ns3 回调 Windows 时不能使用 `localhost`，必须使用 ns3 能访问到的 Windows IP。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from agent.runner import process_with_agent
from agent.test_local import save_raw_data


# FastAPI 应用实例。运行命令：
# uvicorn windows_server.main:app --host 0.0.0.0 --port 8000 --reload
app = FastAPI(title="Windows Push Server", version="1.0.0")


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

    这里把回调模型设计得相对宽松，核心原因是：
    - `task_id` 和 `status` 是必须字段；
    - `nodes/results/error` 的结构可能随着 ns-3 输出变化；
    - 宽松接收可以减少 422，把语义错误转为更容易理解的 400 或日志。
    """

    try:
        raw_payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="request body is not valid JSON") from exc

    if not isinstance(raw_payload, dict):
        raise HTTPException(status_code=400, detail="callback body must be a JSON object")

    _log("info", "callback received", {"task_id": raw_payload.get("task_id", "unknown")})

    # 保存原始数据用于本地测试
    save_raw_data(raw_payload)

    # # 调用智能体处理数据
    # agent_result = await process_with_agent(raw_payload)
    # _log("info", "agent_processed", agent_result)

    # 模拟 Windows 端算法处理耗时（5秒物理时间）
    await asyncio.sleep(1)

    # 直接将决策结果或确认信息作为当前 HTTP 请求的 Response 返回
    return {
        "status": "success",
        "message": "Data processed by Windows",
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn

    # 直接运行 python windows_server/main.py 时使用。
    # 正式调试更推荐 uvicorn 命令，便于开启 reload。
    print("Windows backend starting on port 8000...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
