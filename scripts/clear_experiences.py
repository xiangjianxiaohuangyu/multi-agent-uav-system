"""清空经验库：MySQL + FAISS。

用法：
    python -m scripts.clear_experiences                          # 交互式：仅清空 experiences
    python -m scripts.clear_experiences --yes                    # 跳过确认：仅清空 experiences
    python -m scripts.clear_experiences --include-simulation-records       # 同时清空 simulation_records
    python -m scripts.clear_experiences --include-simulation-records --yes # 跳过确认
    python -m scripts.clear_experiences --dry-run                # 只打印计划，不实际删除

注意：
- 默认只清空 ``experiences`` 表和 FAISS 索引文件。
- 加 ``--include-simulation-records`` 会**同时**清空 ``simulation_records``（破坏性更大）。
- 删除 FAISS 索引文件后必须**重启服务**，否则内存中的 FAISS 仍持有旧数据。
- 脚本会自动加载项目根目录下的 ``.env``（不覆盖已有环境变量）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from sqlalchemy import text

from db.config import DBConfig
from db.engine import get_engine
from db.models import SimulationRecord
from experience.config import ExperienceConfig
from experience.models import Experience


# 脚本入口处自动加载 .env（PowerShell 不会自动 source .env）
# ``find_dotenv`` 从 CWD 向上查找，失败时回退到项目根目录
_dotenv_path = find_dotenv(usecwd=True) or str(
    Path(__file__).resolve().parent.parent / ".env"
)
if _dotenv_path and Path(_dotenv_path).is_file():
    load_dotenv(_dotenv_path, override=False)
    print(f"[env] 已加载 {_dotenv_path}")
else:
    print("[env] 未发现 .env，沿用当前环境变量。")


# 强制读取真实表名（避免硬编码与模型脱节）
EXPERIENCES_TABLE = Experience.__tablename__
SIMULATION_TABLE = SimulationRecord.__tablename__


def _confirm(skip: bool, include_sim: bool) -> None:
    if skip:
        return
    print("即将执行以下操作：")
    print(f"  [mysql] TRUNCATE TABLE {EXPERIENCES_TABLE}")
    if include_sim:
        print(f"  [mysql] TRUNCATE TABLE {SIMULATION_TABLE}  (--include-simulation-records)")
    print("  [faiss] 删除 index.faiss 和 id_map.json")
    print("  → 操作后请重启服务（python -m windows_server.main）。")
    resp = input("\n确认？(yes/no): ").strip().lower()
    if resp != "yes":
        print("已取消。")
        sys.exit(0)


def _truncate_table(conn, table: str, dry_run: bool) -> int:
    """对单张表执行存在性检查 + 计数 + TRUNCATE，返回被删除行数。"""
    exists = conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t"
        ),
        {"t": table},
    ).scalar()
    if not exists:
        print(f"[mysql] {table} 表不存在，跳过。")
        return 0

    before = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    print(f"[mysql] {table} 行数（删前）: {before}")

    if before == 0:
        print(f"[mysql] {table} 已为空，跳过。")
        return 0

    if dry_run:
        print(f"[mysql] [dry-run] 将删除 {table} 中 {before} 行（未实际执行）")
        return before  # 报告"将删除"的行数

    # TRUNCATE 会重置 AUTO_INCREMENT，并释放磁盘空间
    conn.execute(text(f"TRUNCATE TABLE {table}"))
    conn.commit()

    after = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    print(f"[mysql] {table} 行数（删后）: {after}")
    return before


def _clear_mysql(include_sim: bool, dry_run: bool) -> dict[str, int]:
    engine = get_engine()
    url_safe = engine.url.render_as_string(hide_password=True)
    print(f"[mysql] engine: {url_safe}")

    tables = [EXPERIENCES_TABLE]
    if include_sim:
        tables.append(SIMULATION_TABLE)

    deleted: dict[str, int] = {}
    with engine.connect() as conn:
        for t in tables:
            deleted[t] = _truncate_table(conn, t, dry_run)
    return deleted


def _clear_faiss(dry_run: bool) -> int:
    cfg = ExperienceConfig.from_env_and_yaml()
    print(f"[faiss] index path: {cfg.faiss_index_path}")
    print(f"[faiss] id_map path: {cfg.faiss_id_map_path}")

    removed = 0
    for p in (cfg.faiss_index_path, cfg.faiss_id_map_path):
        if p.is_file():
            if dry_run:
                print(f"[faiss] [dry-run] 将删除: {p}")
            else:
                p.unlink()
                print(f"[faiss] 已删除: {p}")
            removed += 1
        elif p.exists():
            print(f"[faiss] 跳过（非文件）: {p}")
        else:
            print(f"[faiss] 不存在: {p}")
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="清空经验库（MySQL + FAISS）")
    parser.add_argument("--yes", action="store_true", help="跳过交互确认")
    parser.add_argument(
        "--include-simulation-records",
        action="store_true",
        help="同时清空 simulation_records 表（破坏性更大，慎用）",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印计划，不实际删除"
    )
    args = parser.parse_args()

    if args.dry_run:
        print("[dry-run] 仅打印计划，不会实际删除。\n")
    elif not args.yes:
        _confirm(skip=False, include_sim=args.include_simulation_records)

    deleted_rows = _clear_mysql(
        include_sim=args.include_simulation_records, dry_run=args.dry_run
    )
    print()
    deleted_files = _clear_faiss(dry_run=args.dry_run)

    print()
    parts = ", ".join(f"{k}={v} 行" for k, v in deleted_rows.items())
    if args.dry_run:
        print(f"[dry-run 完成] 计划删除：{parts}；FAISS 文件 {deleted_files} 个。")
    else:
        print(f"[完成] 删除：{parts}；FAISS 文件 {deleted_files} 个。")
        if any(v > 0 for v in deleted_rows.values()) or deleted_files > 0:
            print(
                "→ 请重启服务（python -m windows_server.main）以重建内存中的 FAISS 索引。"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
