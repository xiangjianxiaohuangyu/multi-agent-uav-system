"""经验库集成测试：SQLite + 临时 FAISS 目录。

运行：
    pytest tests/test_experience_integration.py -v

特性：
- 通过环境变量把 MySQL URL 切到 ``sqlite://``，并把 FAISS 切到 ``tmp_path``。
- 强制 ``reset_faiss()`` 以确保每次测试都重新初始化。
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest


# ---------------------------------------------------------------------- #
# 强制：在 import experience.* 之前设置环境变量，让 MySQL → SQLite
# ---------------------------------------------------------------------- #


@pytest.fixture(scope="session", autouse=True)
def _configure_test_env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """session 级 fixture：切到 SQLite + 临时 FAISS 目录。"""
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    faiss_dir = tmp_path_factory.mktemp("faiss")

    # 清空并设置环境
    for k in ("MYSQL_HOST", "MYSQL_PORT", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"):
        os.environ.pop(k, None)
    # 用 sqlite url；DBConfig.from_env 在 url() 里硬编码 mysql+pymysql；
    # 所以我们直接 monkey-patch get_engine
    os.environ["EXPERIENCE_FAISS_DIR"] = str(faiss_dir)

    # Patch db.engine 的 URL 生成（最简方式：直接给一个 _engine）
    from db import engine as db_engine
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    test_engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    db_engine._engine = test_engine  # type: ignore[attr-defined]
    db_engine._SessionLocal = sessionmaker(  # type: ignore[attr-defined]
        bind=test_engine, autocommit=False, autoflush=False
    )

    # Patch get_session 以使用新 session
    from contextlib import contextmanager

    @contextmanager
    def _get_session():
        s = db_engine._SessionLocal()  # type: ignore[attr-defined]
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    db_engine.get_session = _get_session  # type: ignore[assignment]

    # 关键：experience.repository 在 import 时已绑定 get_session/get_engine 引用，
    # 也得同步 patch（因为用的是 ``from db.engine import X``）
    from experience import repository as exp_repository
    exp_repository.get_session = _get_session  # type: ignore[assignment]
    exp_repository.get_engine = lambda: test_engine  # type: ignore[assignment]

    # 兜底建表
    from db.models import Base
    Base.metadata.create_all(bind=test_engine)

    yield

    # 清理
    try:
        shutil.rmtree(faiss_dir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _reset_faiss_each_test() -> Iterator[None]:
    """每个测试都重置 FAISS 单例 + 临时目录。"""
    from experience import engine as exp_engine

    exp_engine.reset_faiss()
    yield
    exp_engine.reset_faiss()


# ---------------------------------------------------------------------- #
# 实际测试
# ---------------------------------------------------------------------- #


def _make_scene(values: list[float]) -> dict:
    from experience.scoring import SCENE_FIELD_ORDER

    return {k: float(v) for k, v in zip(SCENE_FIELD_ORDER, values)}


def _make_parameter(idx: int) -> dict:
    return {
        "hello_interval": 1.0,
        "candidate_num": 2,
        "w_distance": 0.4,
        "w_linktime": 0.3,
        "w_energy": 0.2 - idx * 0.01,
        "w_queue": 0.1 + idx * 0.01,
    }


def _make_result(pdr: float, delay: float, energy: float) -> dict:  # noqa: ARG001
    return {
        "e2e_pdr": pdr,
        "e2e_delay": delay,
        "routing_overhead": 10.0,
    }


def test_add_then_search_roundtrip() -> None:
    from experience.repository import ExperienceRepository

    repo = ExperienceRepository()
    # 添加 5 条不同 scene
    eids: list[int] = []
    base = [5.0, 80.0, 2, 4, 120.0, 0.6, 50.0, 5.0, 100.0, 12.0, 60.0,
            1.2, 0.3, 30.0, 4.0, 4.5, 1.0, 2.0, 0.5, 4, 75.0, 5.0, 60.0]
    for i in range(5):
        vec = list(base)
        vec[0] = 5.0 + i  # 改变 speed 让 scene 略不同
        eid = repo.add(
            scene=_make_scene(vec),
            parameter=_make_parameter(i),
            result=_make_result(0.9, 150.0, 8.0),
        )
        assert eid is not None and eid > 0
        eids.append(eid)

    # 检索：用一个相似 scene（与第 0 条最接近）
    query_vec = list(base)
    query_vec[0] = 5.1
    hits = repo.search(_make_scene(query_vec), k=3)
    assert len(hits) > 0
    # 按 L2 距离 ASC 排序
    distances = [h["distance"] for h in hits if h["distance"] is not None]
    assert distances == sorted(distances)
    # 至少返回 1 条
    assert len(hits) <= 3


def test_get_by_id_roundtrip() -> None:
    from experience.repository import ExperienceRepository

    repo = ExperienceRepository()
    eid = repo.add(
        scene=_make_scene([1.0] * 23),
        parameter=_make_parameter(0),
        result=_make_result(0.95, 100.0, 5.0),
    )
    assert eid is not None
    row = repo.get_by_id(eid)
    assert row is not None
    assert row["experience_id"] == eid
    assert "hello_interval" in row["parameter"]


def test_get_by_id_missing_returns_none() -> None:
    from experience.repository import ExperienceRepository

    repo = ExperienceRepository()
    assert repo.get_by_id(999_999_999) is None


def test_delete_by_id() -> None:
    from experience.repository import ExperienceRepository

    repo = ExperienceRepository()
    eid = repo.add(
        scene=_make_scene([1.0] * 23),
        parameter=_make_parameter(0),
        result=_make_result(0.9, 100.0, 5.0),
    )
    assert eid is not None
    assert repo.get_by_id(eid) is not None

    ok = repo.delete_by_id(eid)
    assert ok is True
    assert repo.get_by_id(eid) is None

    # 再次删除返回 False
    assert repo.delete_by_id(eid) is False


def test_count_and_list_paginated() -> None:
    from experience.repository import ExperienceRepository

    repo = ExperienceRepository()
    for i in range(3):
        repo.add(
            scene=_make_scene([1.0 + i] * 23),
            parameter=_make_parameter(i),
            result=_make_result(0.8, 200.0, 10.0),
        )

    assert repo.count() == 3
    rows = repo.list_paginated(offset=0, limit=10)
    assert len(rows) == 3
    # 按 id DESC 排序
    ids = [r["experience_id"] for r in rows]
    assert ids == sorted(ids, reverse=True)


def test_topk_clamped_to_ntotal() -> None:
    """k > ntotal 时不应抛，应返回所有。"""
    from experience.repository import ExperienceRepository

    repo = ExperienceRepository()
    for i in range(2):
        repo.add(
            scene=_make_scene([float(i)] * 23),
            parameter=_make_parameter(i),
            result=_make_result(0.9, 100.0, 5.0),
        )
    hits = repo.search(_make_scene([1.0] * 23), k=100)
    assert len(hits) == 2


def test_search_empty_index_returns_empty() -> None:
    from experience.repository import ExperienceRepository

    repo = ExperienceRepository()
    hits = repo.search(_make_scene([1.0] * 23), k=5)
    assert hits == []


def test_dual_write_rollback_on_faiss_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """FAISS 写失败时 MySQL 行应被补偿删除。"""
    from experience.faiss_manager import FaissIndexManager
    from experience.repository import ExperienceRepository

    # 先 add 一次以触发 manager 初始化
    repo = ExperienceRepository()
    repo.add(
        scene=_make_scene([1.0] * 23),
        parameter=_make_parameter(0),
        result=_make_result(0.9, 100.0, 5.0),
    )
    before = repo.count()

    # 强制 FAISS 抛异常
    def _boom(self: FaissIndexManager, vec_id: int, vector: np.ndarray) -> None:  # noqa: ARG001
        raise RuntimeError("simulated faiss failure")

    monkeypatch.setattr(FaissIndexManager, "add_vector", _boom)

    eid = repo.add(
        scene=_make_scene([2.0] * 23),
        parameter=_make_parameter(1),
        result=_make_result(0.9, 100.0, 5.0),
    )
    assert eid is None  # 应当返回 None
    assert repo.count() == before  # 行已被补偿删除


def test_disabled_config_blocks_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """``EXPERIENCE_DISABLED=1`` 时 ``add`` 应当返 None。"""
    from experience import engine as exp_engine
    from experience.config import ExperienceConfig
    from experience.repository import ExperienceRepository

    cfg = ExperienceConfig.from_env_and_yaml()
    cfg_disabled = ExperienceConfig(
        dimension=cfg.dimension,
        faiss_index_path=cfg.faiss_index_path,
        faiss_id_map_path=cfg.faiss_id_map_path,
        topk_default=cfg.topk_default,
        topk_max=cfg.topk_max,
        disabled=True,
        auto_create_table=cfg.auto_create_table,
        log_level=cfg.log_level,
    )
    repo = ExperienceRepository(config=cfg_disabled)
    eid = repo.add(
        scene=_make_scene([1.0] * 23),
        parameter=_make_parameter(0),
        result=_make_result(0.9, 100.0, 5.0),
    )
    assert eid is None
    # 切回非 disabled
    exp_engine.reset_faiss()


def test_payload_to_experience_e2e() -> None:
    """整个 pipeline：从 payload 到 add 到 search。"""
    from experience.repository import ExperienceRepository
    from experience.scoring import payload_to_experience

    payload = {
        "type": "simulation",
        "speed": 3.0, "energy": 70.0, "queue_length": 1,
        "neighbor_count": 3, "distance_to_destination": 100.0,
        "forward_candidate_ratio": 0.5,
        "distance_to_me_mean": 40.0,
        "distance_to_me_std": 4.0,
        "distance_to_destination_mean": 90.0,
        "distance_to_destination_std": 10.0,
        "distance_to_destination_min": 50.0,
        "relative_speed_mean": 1.0,
        "relative_speed_std": 0.2,
        "link_lifetime_mean": 20.0,
        "link_lifetime_std": 3.0,
        "neighbor_degree_mean": 3.5,
        "neighbor_degree_std": 0.8,
        "queue_length_mean": 1.0,
        "queue_length_std": 0.2,
        "queue_length_max": 2,
        "energy_mean": 65.0,
        "energy_std": 4.0,
        "energy_min": 50.0,
        "hello_interval": 1.0, "path_num": 1,
        "w_distance": 0.4, "w_linkTime": 0.3,
        "w_relVelocity": 0.2, "w_neighborCount": 0.1,
        "avg_pdr": 0.85, "avg_delay": 120.0,
    }
    mapped = payload_to_experience(payload)
    assert mapped is not None

    repo = ExperienceRepository()
    eid = repo.add_from_payload(payload)
    assert eid is not None
    assert eid > 0
