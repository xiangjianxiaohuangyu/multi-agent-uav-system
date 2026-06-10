"""经验库仓储层。

核心不变量（与 ``db.repository.write_simulation_record`` 对齐）：
- 所有公开方法**永不抛出异常**；失败仅 ``log.warning``/``log.exception``，
  返回 ``None`` / ``False`` / ``[]``。
- 双写顺序：MySQL 先提交，FAISS 失败时 MySQL 行补偿删除。
- 读路径：FAISS 返 id → MySQL ``SELECT ... WHERE id IN (...)`` → 按 score DESC 排序。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError

from db.engine import get_session
from experience.config import ExperienceConfig
from experience.engine import get_config, get_faiss_manager
from experience.models import Experience
from experience.scoring import (
    compute_score,
    payload_to_experience,
    scene_to_vector,
)

log = logging.getLogger(__name__)


class ExperienceRepository:
    """经验库仓储。模块级单例足够。"""

    def __init__(self, config: ExperienceConfig | None = None) -> None:
        # config 在第一次使用 manager 时被引擎初始化；这里仅缓存
        self._config: ExperienceConfig | None = config

    # ------------------------------------------------------------------ #
    # 内部辅助
    # ------------------------------------------------------------------ #

    def _cfg(self) -> ExperienceConfig:
        if self._config is None:
            self._config = get_config()
        return self._config

    def _row_to_dict(self, row: Experience) -> dict[str, Any]:
        return {
            "experience_id": int(row.experience_id),
            "scene": dict(row.scene_json) if row.scene_json else {},
            "scene_vector": list(row.scene_vector) if row.scene_vector else [],
            "parameter": dict(row.parameter_json) if row.parameter_json else {},
            "result": dict(row.result_json) if row.result_json else {},
            "score": float(row.score) if row.score is not None else 0.0,
            "created_time": row.created_time.isoformat() if row.created_time else None,
        }

    # ------------------------------------------------------------------ #
    # 写入路径
    # ------------------------------------------------------------------ #

    def add(
        self,
        scene: dict[str, Any],
        parameter: dict[str, Any],
        result: dict[str, Any],
    ) -> int | None:
        """新增一条经验。返回新 ``experience_id``，失败返回 ``None``。"""
        cfg = self._cfg()
        if cfg.is_disabled():
            log.debug("experience library disabled; skip add")
            return None

        vector = scene_to_vector(scene)
        try:
            score = compute_score(
                pdr=float(result.get("e2e_pdr", 0.0)),
                delay_ms=float(result.get("e2e_delay", 0.0)),
                energy=float(result.get("energy_consumption", 0.0)),
                cfg=cfg,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("compute_score failed: %s; falling back to 0.0", exc)
            score = 0.0

        # 1) MySQL 写入
        row = Experience(
            scene_json=scene,
            scene_vector=vector,
            parameter_json=parameter,
            result_json=result,
            score=float(score),
        )
        eid: int | None = None
        try:
            with get_session() as s:
                s.add(row)
                s.commit()
                s.refresh(row)
                eid = int(row.experience_id)
        except SQLAlchemyError as exc:
            log.warning("mysql add experience failed: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            log.exception("unexpected mysql add experience error: %s", exc)
            return None

        # 2) FAISS 写入
        try:
            mgr = get_faiss_manager()
            mgr.add_vector(eid, np.asarray(vector, dtype=np.float32))
        except ValueError as exc:
            # id 重复 → 极少见（autoincrement 不应重复）
            log.exception("faiss add_vector id collision: %s", exc)
            self._compensating_delete(eid)
            return None
        except Exception as exc:  # noqa: BLE001
            log.exception("faiss add_vector failed; rolling back mysql row: %s", exc)
            self._compensating_delete(eid)
            return None

        return eid

    def _compensating_delete(self, experience_id: int) -> None:
        """FAISS 失败时删除 MySQL 行（双写补偿）。"""
        try:
            with get_session() as s:
                stmt = delete(Experience).where(Experience.experience_id == experience_id)
                s.execute(stmt)
                s.commit()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "compensating delete failed for experience_id=%s: %s; orphan row will remain",
                experience_id, exc,
            )

    def delete_by_id(self, experience_id: int) -> bool:
        """按 id 删除（FAISS + MySQL）。"""
        cfg = self._cfg()
        if cfg.is_disabled():
            return False
        try:
            mgr = get_faiss_manager()
            faiss_ok = mgr.delete_vector(int(experience_id))
        except Exception as exc:  # noqa: BLE001
            log.warning("faiss delete_vector(%s) failed: %s", experience_id, exc)
            faiss_ok = False

        try:
            with get_session() as s:
                stmt = delete(Experience).where(Experience.experience_id == int(experience_id))
                result = s.execute(stmt)
                s.commit()
                return (result.rowcount or 0) > 0
        except Exception as exc:  # noqa: BLE001
            log.exception("mysql delete experience %s failed: %s", experience_id, exc)
            return False

    def update(
        self,
        experience_id: int,
        scene: dict[str, Any],
        parameter: dict[str, Any],
        result: dict[str, Any],
    ) -> bool:
        """更新一条经验。先更新 MySQL 行，再更新 FAISS 向量。"""
        cfg = self._cfg()
        if cfg.is_disabled():
            return False

        vector = scene_to_vector(scene)
        try:
            score = compute_score(
                pdr=float(result.get("e2e_pdr", 0.0)),
                delay_ms=float(result.get("e2e_delay", 0.0)),
                energy=float(result.get("energy_consumption", 0.0)),
                cfg=cfg,
            )
        except Exception:
            score = 0.0

        try:
            with get_session() as s:
                row = s.get(Experience, int(experience_id))
                if row is None:
                    return False
                row.scene_json = scene
                row.scene_vector = vector
                row.parameter_json = parameter
                row.result_json = result
                row.score = float(score)
                s.commit()
        except Exception as exc:  # noqa: BLE001
            log.exception("mysql update experience %s failed: %s", experience_id, exc)
            return False

        try:
            mgr = get_faiss_manager()
            return mgr.update_vector(int(experience_id), np.asarray(vector, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            log.exception("faiss update_vector(%s) failed: %s", experience_id, exc)
            return False

    # ------------------------------------------------------------------ #
    # 读取路径
    # ------------------------------------------------------------------ #

    def get_by_id(self, experience_id: int) -> dict[str, Any] | None:
        try:
            with get_session() as s:
                row = s.get(Experience, int(experience_id))
                if row is None:
                    return None
                return self._row_to_dict(row)
        except Exception as exc:  # noqa: BLE001
            log.exception("get_by_id(%s) failed: %s", experience_id, exc)
            return None

    def search(
        self,
        scene: dict[str, Any],
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """根据场景相似度检索 Top-K 经验，按 score DESC 排序。

        流程：Scene → Vector → FAISS TopK → MySQL 批量取完整记录 → 按 score 排序。
        """
        cfg = self._cfg()
        if cfg.is_disabled():
            return []
        kk = cfg.topk_capped(k)
        vector = scene_to_vector(scene)

        # 1) FAISS 搜索
        try:
            mgr = get_faiss_manager()
            distances, ids = mgr.search_topk(
                np.asarray(vector, dtype=np.float32), kk
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("faiss search failed: %s", exc)
            return []
        if ids.size == 0:
            return []

        # 2) MySQL 批量取
        id_list = [int(i) for i in ids.tolist() if int(i) >= 0]
        if not id_list:
            return []
        try:
            with get_session() as s:
                stmt = select(Experience).where(Experience.experience_id.in_(id_list))
                rows = s.execute(stmt).scalars().all()
        except Exception as exc:  # noqa: BLE001
            log.exception("mysql bulk read for search failed: %s", exc)
            return []

        # 3) 按 score DESC 排序
        hits = [self._row_to_dict(r) for r in rows]
        hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
        return hits

    def list_paginated(
        self,
        offset: int = 0,
        limit: int = 100,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """分页列出经验。"""
        try:
            with get_session() as s:
                stmt = select(Experience)
                if min_score is not None:
                    stmt = stmt.where(Experience.score >= float(min_score))
                stmt = stmt.order_by(Experience.experience_id.desc())
                stmt = stmt.offset(max(0, int(offset))).limit(max(1, int(limit)))
                rows = s.execute(stmt).scalars().all()
                return [self._row_to_dict(r) for r in rows]
        except Exception as exc:  # noqa: BLE001
            log.exception("list_paginated failed: %s", exc)
            return []

    def count(self) -> int:
        try:
            with get_session() as s:
                from sqlalchemy import func as sqlfunc
                stmt = select(sqlfunc.count(Experience.experience_id))
                return int(s.execute(stmt).scalar() or 0)
        except Exception as exc:  # noqa: BLE001
            log.exception("count() failed: %s", exc)
            return 0

    # ------------------------------------------------------------------ #
    # 启动钩子（被 router.on_event 调用）
    # ------------------------------------------------------------------ #

    def ensure_table(self) -> bool:
        """启动时建表（幂等）。"""
        try:
            from db.engine import get_engine
            from db.models import Base
            Base.metadata.create_all(bind=get_engine())
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("ensure_table failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Payload 便捷入口（被 main.py 调用）
    # ------------------------------------------------------------------ #

    def add_from_payload(self, payload: dict[str, Any]) -> int | None:
        """从 callback payload 一步写入。``scene_params`` / 字段全缺失返回 None。"""
        mapped = payload_to_experience(payload)
        if mapped is None:
            return None
        scene, parameter, result = mapped
        return self.add(scene, parameter, result)
