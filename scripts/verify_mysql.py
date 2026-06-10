"""读检脚本：连 MySQL 跑几条简单 SELECT 验证表结构和数据。

用法：
    python -m scripts.verify_mysql
"""

from __future__ import annotations

import sys

from sqlalchemy import text

from db.engine import get_engine, get_session
from db.models import SimulationRecord


def main() -> int:
    engine = get_engine()
    print(f"[verify] engine: {engine.url.render_as_string(hide_password=True)}")

    with engine.connect() as conn:
        # 1. 表是否存在
        rows = conn.execute(
            text(
                "SELECT TABLE_NAME FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t"
            ),
            {"t": SimulationRecord.__tablename__},
        ).fetchall()
        if not rows:
            print(f"[verify] FAIL: table `{SimulationRecord.__tablename__}` not found")
            return 1
        print(f"[verify] OK: table `{SimulationRecord.__tablename__}` exists")

        # 2. 行数
        total = conn.execute(
            text(f"SELECT COUNT(*) FROM {SimulationRecord.__tablename__}")
        ).scalar()
        print(f"[verify] row count: {total}")

        if total and total > 0:
            # 3. 最近 5 行关键列
            conn.execute(text("SET SESSION TRANSACTION READ ONLY"))
            with get_session() as s:
                last_five = (
                    s.query(SimulationRecord)
                    .order_by(SimulationRecord.id.desc())
                    .limit(5)
                    .all()
                )
            print("[verify] latest 5 rows (key columns):")
            for r in last_five:
                print(
                    f"  id={r.id} task_id={r.task_id!r} device_id={r.device_id!r} "
                    f"sim_t={r.simulation_time} m_speed={r.m_speed} "
                    f"nb_rs_mean={r.nb_relative_speed_mean} res_pdr={r.res_avg_pdr}"
                )

            # 4. 索引是否齐全
            indexes = conn.execute(
                text(
                    "SELECT INDEX_NAME, GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) "
                    "FROM information_schema.STATISTICS "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t "
                    "GROUP BY INDEX_NAME"
                ),
                {"t": SimulationRecord.__tablename__},
            ).fetchall()
            print("[verify] indexes:")
            for name, cols in indexes:
                print(f"  - {name} ({cols})")

    print("[verify] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
