"""本地测试模块。

用于：
1. 存储 ns3 推送的原始数据
2. 本地回放数据进行测试

使用方法：
1. 正常运行时，数据会自动保存到 data/ 目录
2. 本地测试时：python -m agent.test_local
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# 测试配置：控制使用多少个 JSON 文件，1 表示只使用第一个
TEST_JSON_COUNT = 1

# 数据存储目录
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def save_raw_data(data: dict[str, Any]) -> str:
    """保存原始数据到文件。

    文件路径统一为 ``data/<task_id>.json``（无子目录、无时间戳），
    同一 task_id 多次写入会覆盖前一次结果。

    Args:
        data: 原始 JSON 数据

    Returns:
        保存的文件路径
    """
    task_id = str(data.get("task_id") or "default")
    filepath = DATA_DIR / f"{task_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(filepath)


def load_raw_data(filepath: str) -> dict[str, Any]:
    """加载原始数据文件。

    Args:
        filepath: 文件路径（相对或绝对）
    """
    path = Path(filepath)
    if not path.is_absolute():
        path = DATA_DIR / filepath
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_latest_data() -> dict[str, Any] | None:
    """加载最新的原始数据文件。"""
    json_files = sorted(DATA_DIR.glob("*.json"), reverse=True)
    if json_files:
        return load_raw_data(str(json_files[0]))
    return None


def list_saved_data() -> list[str]:
    """列出所有已保存的数据文件（相对于 DATA_DIR 的路径）。"""
    return [str(f.relative_to(DATA_DIR)) for f in sorted(DATA_DIR.glob("*.json"), reverse=True)]


def list_task_ids() -> list[str]:
    """列出所有任务ID（data 目录下 .json 文件的 stem）。"""
    return [f.stem for f in DATA_DIR.glob("*.json") if f.is_file()]


async def run_local_test():
    """本地测试入口，根据 TEST_JSON_COUNT 加载数据并执行处理流程。"""
    from agent.runner import process_with_agent

    # 收集 data/ 目录下所有 .json 文件（路径即 task_id.json）
    json_files = sorted(DATA_DIR.glob("*.json"))

    if not json_files:
        print("[Test] No saved data found in data/", flush=True)
        print()
        print("[Test] Please run the server first to collect data", flush=True)
        print()
        return

    # 读出来后按 type 字段分流：scene_params 优先处理一次
    scene_payload: dict[str, Any] | None = None
    ns3_payloads: list[dict[str, Any]] = []
    for json_file in json_files:
        payload = load_raw_data(str(json_file))
        if payload.get("type") == "scene_params":
            scene_payload = payload
        else:
            ns3_payloads.append(payload)

    if scene_payload is not None:
        print(f"[Test] Loading scene_params: {scene_payload.get('task_id')}.json", flush=True)
        await process_with_agent(scene_payload)
        print()

    # 根据 TEST_JSON_COUNT 确定要使用的 ns3 数据条数
    ns3_count = min(len(ns3_payloads), TEST_JSON_COUNT)
    payloads_to_use = ns3_payloads[:ns3_count]

    print(f"[Test] Using {len(payloads_to_use)} payload(s) (TEST_JSON_COUNT={TEST_JSON_COUNT})", flush=True)

    for payload in payloads_to_use:
        print(f"[Test] Loading: {payload.get('task_id')}.json", flush=True)
        result = await process_with_agent(payload)
        print(f"[Test] Result: {result}", flush=True)


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_local_test())