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

# 数据存储目录
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def save_raw_data(data: dict[str, Any], filename: str | None = None) -> str:
    """保存原始数据到文件。

    Args:
        data: 原始 JSON 数据
        filename: 指定文件名，默认根据 type 生成

    Returns:
        保存的文件路径
    """
    task_id = data.get("task_id", "default")
    task_dir = DATA_DIR / task_id
    task_dir.mkdir(exist_ok=True)

    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        msg_type = data.get("type", "simulation")
        if msg_type == "scene_params":
            filename = f"scene_params_{timestamp}.json"
        else:
            filename = f"ns3_data_{timestamp}.json"

    filepath = task_dir / filename
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
    json_files = sorted(DATA_DIR.glob("**/ns3_data_*.json"), reverse=True)
    if json_files:
        return load_raw_data(str(json_files[0]))
    return None


def list_saved_data() -> list[str]:
    """列出所有已保存的数据文件（相对于 DATA_DIR 的路径）。"""
    return [str(f.relative_to(DATA_DIR)) for f in sorted(DATA_DIR.glob("**/ns3_data_*.json"), reverse=True)]


def list_task_ids() -> list[str]:
    """列出所有任务ID（子文件夹）。"""
    return [d.name for d in DATA_DIR.iterdir() if d.is_dir()]


async def run_local_test():
    """本地测试入口，按顺序加载所有数据并执行处理流程。"""
    from agent.runner import process_with_agent

    # 收集所有 JSON 文件
    json_files = sorted(DATA_DIR.glob("**/ns3_data_*.json"))
    json_files.extend(sorted(DATA_DIR.glob("**/scene_params_*.json")))

    if not json_files:
        print("[Test] No saved data found in data/", flush=True)
        print()
        print("[Test] Please run the server first to collect data", flush=True)
        print()
        return

    for json_file in json_files:
        data = load_raw_data(str(json_file))
        result = await process_with_agent(data)


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_local_test())