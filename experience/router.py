"""经验库 FastAPI 路由（5 端点）。

由 ``windows_server.main`` 通过 ``app.include_router(router, prefix=\"/experience\")`` 挂载。
所有端点：

- POST   /experience/add
- GET    /experience/{experience_id}
- DELETE /experience/{experience_id}
- POST   /experience/search
- GET    /experience/statistics
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response

from experience.engine import get_config, init_faiss
from experience.repository import ExperienceRepository
from experience.schemas import (
    AddExperienceRequest,
    AddExperienceResponse,
    ExperienceOut,
    SearchExperienceRequest,
    SearchExperienceResponse,
    StatisticsResponse,
)
from experience.statistics import compute_statistics, to_csv

log = logging.getLogger(__name__)

router = APIRouter(prefix="/experience", tags=["experience"])
_repo = ExperienceRepository()


def _log(level: str, message: str, data: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "level": level,
        "component": "experience",
        "message": message,
    }
    if data is not None:
        payload["data"] = data
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _disabled_guard() -> None:
    """``EXPERIENCE_DISABLED=1`` 时所有端点返 503。"""
    if get_config().is_disabled():
        raise HTTPException(status_code=503, detail="experience library disabled")


# ---------------------------------------------------------------------- #
# 启动钩子
# ---------------------------------------------------------------------- #


@router.on_event("startup")
def _startup() -> None:
    """FastAPI 启动时确保表存在 + FAISS 索引加载。"""
    cfg = get_config()
    if cfg.auto_create_table:
        ok = _repo.ensure_table()
        _log("info", "experiences table ensured", {"ok": ok})
    if not cfg.is_disabled():
        try:
            init_faiss()
            _log("info", "faiss manager ready")
        except Exception as exc:  # noqa: BLE001
            _log("warning", "faiss init failed", {"error": str(exc)})


# ---------------------------------------------------------------------- #
# POST /experience/add
# ---------------------------------------------------------------------- #


@router.post("/add", response_model=AddExperienceResponse)
def add_experience(req: AddExperienceRequest) -> AddExperienceResponse:
    _disabled_guard()
    try:
        scene = req.scene.model_dump()
        parameter = req.parameter.model_dump()
        result = req.result.model_dump()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"validation error: {exc}") from exc

    eid = _repo.add(scene, parameter, result)
    if eid is None:
        raise HTTPException(status_code=500, detail="failed to persist experience")

    _log("info", "experience added", {"experience_id": eid})
    return AddExperienceResponse(experience_id=eid)


# ---------------------------------------------------------------------- #
# GET /experience/{id}
# ---------------------------------------------------------------------- #


@router.get("/{experience_id}", response_model=ExperienceOut)
def get_experience(experience_id: int) -> ExperienceOut:
    _disabled_guard()
    row = _repo.get_by_id(int(experience_id))
    if row is None:
        raise HTTPException(status_code=404, detail=f"experience {experience_id} not found")
    return ExperienceOut(**row)


# ---------------------------------------------------------------------- #
# DELETE /experience/{id}
# ---------------------------------------------------------------------- #


@router.delete("/{experience_id}", status_code=204)
def delete_experience(experience_id: int) -> Response:
    _disabled_guard()
    ok = _repo.delete_by_id(int(experience_id))
    if not ok:
        raise HTTPException(status_code=404, detail=f"experience {experience_id} not found")
    _log("info", "experience deleted", {"experience_id": experience_id})
    return Response(status_code=204)


# ---------------------------------------------------------------------- #
# POST /experience/search
# ---------------------------------------------------------------------- #


@router.post("/search", response_model=SearchExperienceResponse)
def search_experiences(req: SearchExperienceRequest) -> SearchExperienceResponse:
    _disabled_guard()
    scene = req.scene.model_dump()
    cfg = get_config()
    kk = cfg.topk_capped(req.top_k)
    hits_raw = _repo.search(scene, k=kk)
    return SearchExperienceResponse(hits=[ExperienceOut(**h) for h in hits_raw])


# ---------------------------------------------------------------------- #
# GET /experience/statistics
# ---------------------------------------------------------------------- #


@router.get("", response_model=StatisticsResponse)
def get_statistics(
    export: bool = Query(False, description="若为 true，返回 CSV；否则返回 JSON"),
) -> Any:
    _disabled_guard()
    stats = compute_statistics(_repo)
    if export:
        rows = _repo.list_paginated(offset=0, limit=10_000)
        csv_text = to_csv(stats, rows)
        return Response(
            content=csv_text,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=experience_statistics.csv"},
        )
    return StatisticsResponse(
        count=int(stats.get("count", 0)),
        parameter_distribution=stats.get("parameter_distribution", {}),
        scene_distribution=stats.get("scene_distribution", {}),
    )


# ---------------------------------------------------------------------- #
# POST /experience/admin/rebuild
# ---------------------------------------------------------------------- #


@router.post("/admin/rebuild")
def admin_rebuild() -> dict[str, Any]:
    """从 MySQL 重建 FAISS 索引（用于迁移 / 修复）。"""
    _disabled_guard()
    restored = _repo.rebuild_from_mysql()
    _log("info", "faiss rebuilt from mysql", {"restored": restored})
    return {"restored": int(restored)}
