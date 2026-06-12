"""经验库 FastAPI API 测试。

运行：
    pytest tests/test_experience_api.py -v
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def _configure_test_env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """切到 SQLite + 临时 FAISS 目录（与 integration 测试相同）。"""
    db_path = tmp_path_factory.mktemp("api_db") / "test.db"
    faiss_dir = tmp_path_factory.mktemp("api_faiss")

    for k in ("MYSQL_HOST", "MYSQL_PORT", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"):
        os.environ.pop(k, None)
    os.environ["EXPERIENCE_FAISS_DIR"] = str(faiss_dir)

    from db import engine as db_engine
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from contextlib import contextmanager

    test_engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    db_engine._engine = test_engine  # type: ignore[attr-defined]
    db_engine._SessionLocal = sessionmaker(  # type: ignore[attr-defined]
        bind=test_engine, autocommit=False, autoflush=False
    )

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

    # 同步 patch experience.repository 已绑定的引用
    from experience import repository as exp_repository
    exp_repository.get_session = _get_session  # type: ignore[assignment]
    exp_repository.get_engine = lambda: test_engine  # type: ignore[assignment]

    from db.models import Base
    Base.metadata.create_all(bind=test_engine)

    yield

    shutil.rmtree(faiss_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def _reset_state() -> Iterator[None]:
    from experience import engine as exp_engine
    from experience.repository import ExperienceRepository
    from db import engine as db_engine
    from db.models import Experience

    exp_engine.reset_faiss()
    # 清表（保证测试间隔离）
    with db_engine.get_session() as s:  # type: ignore[attr-defined]
        from sqlalchemy import delete
        s.execute(delete(Experience))
        s.commit()
    yield
    exp_engine.reset_faiss()


@pytest.fixture
def client() -> TestClient:
    from windows_server.main import app
    return TestClient(app)


# ---------------------------------------------------------------------- #
# 构造合法 payload
# ---------------------------------------------------------------------- #


def _full_payload() -> dict[str, Any]:
    return {
        "scene": {
            "speed": 5.0,
            "energy": 80.0,
            "queue_length": 2,
            "neighbor_count": 4,
            "distance_to_destination": 120.0,
            "forward_candidate_ratio": 0.6,
            "distance_to_me_mean": 50.0,
            "distance_to_me_std": 5.0,
            "distance_to_destination_mean": 100.0,
            "distance_to_destination_std": 12.0,
            "distance_to_destination_min": 60.0,
            "relative_speed_mean": 1.2,
            "relative_speed_std": 0.3,
            "link_lifetime_mean": 30.0,
            "link_lifetime_std": 4.0,
            "neighbor_degree_mean": 4.5,
            "neighbor_degree_std": 1.0,
            "queue_length_mean": 2.0,
            "queue_length_std": 0.5,
            "queue_length_max": 4,
            "energy_mean": 75.0,
            "energy_std": 5.0,
            "energy_min": 60.0,
        },
        "parameter": {
            "hello_interval": 1.0,
            "candidate_num": 2,
            "w_distance": 0.4,
            "w_linktime": 0.3,
            "w_energy": 0.2,
            "w_queue": 0.1,
        },
        "result": {
            "e2e_pdr": 0.92,
            "e2e_delay": 150.0,
            "routing_overhead": 12.0,
        },
    }


# ---------------------------------------------------------------------- #
# POST /experience/add
# ---------------------------------------------------------------------- #


class TestAdd:
    def test_add_returns_id(self, client: TestClient) -> None:
        r = client.post("/experience/add", json=_full_payload())
        assert r.status_code == 200, r.text
        body = r.json()
        assert "experience_id" in body
        assert body["experience_id"] > 0

    def test_add_rejects_missing_scene(self, client: TestClient) -> None:
        payload = _full_payload()
        del payload["scene"]
        r = client.post("/experience/add", json=payload)
        assert r.status_code == 422

    def test_add_rejects_extra_field(self, client: TestClient) -> None:
        payload = _full_payload()
        payload["scene"]["extra_field"] = 1.0  # type: ignore[index]
        r = client.post("/experience/add", json=payload)
        assert r.status_code == 422

    def test_add_rejects_invalid_weight_range(self, client: TestClient) -> None:
        payload = _full_payload()
        payload["parameter"]["w_distance"] = 1.5  # > 1
        r = client.post("/experience/add", json=payload)
        assert r.status_code == 422

    def test_add_rejects_negative_pdr(self, client: TestClient) -> None:
        payload = _full_payload()
        payload["result"]["e2e_pdr"] = -0.1
        r = client.post("/experience/add", json=payload)
        assert r.status_code == 422


# ---------------------------------------------------------------------- #
# GET /experience/{id}
# ---------------------------------------------------------------------- #


class TestGet:
    def test_get_returns_full_record(self, client: TestClient) -> None:
        r = client.post("/experience/add", json=_full_payload())
        eid = r.json()["experience_id"]
        r2 = client.get(f"/experience/{eid}")
        assert r2.status_code == 200
        body = r2.json()
        assert body["experience_id"] == eid
        assert len(body["scene_vector"]) == 23
        assert "hello_interval" in body["parameter"]
        assert "e2e_pdr" in body["result"]

    def test_get_404_for_missing(self, client: TestClient) -> None:
        r = client.get("/experience/9999999")
        assert r.status_code == 404


# ---------------------------------------------------------------------- #
# DELETE /experience/{id}
# ---------------------------------------------------------------------- #


class TestDelete:
    def test_delete_returns_204(self, client: TestClient) -> None:
        r = client.post("/experience/add", json=_full_payload())
        eid = r.json()["experience_id"]
        rd = client.delete(f"/experience/{eid}")
        assert rd.status_code == 204

    def test_delete_then_get_404(self, client: TestClient) -> None:
        r = client.post("/experience/add", json=_full_payload())
        eid = r.json()["experience_id"]
        client.delete(f"/experience/{eid}")
        r2 = client.get(f"/experience/{eid}")
        assert r2.status_code == 404

    def test_delete_missing_returns_404(self, client: TestClient) -> None:
        r = client.delete("/experience/9999999")
        assert r.status_code == 404


# ---------------------------------------------------------------------- #
# POST /experience/search
# ---------------------------------------------------------------------- #


class TestSearch:
    def test_search_returns_sorted_by_distance(self, client: TestClient) -> None:
        for i in range(3):
            p = _full_payload()
            p["scene"]["speed"] = 3.0 + i
            p["result"]["e2e_pdr"] = 0.7 + i * 0.1
            client.post("/experience/add", json=p)

        r = client.post(
            "/experience/search",
            json={"scene": _full_payload()["scene"], "top_k": 2},
        )
        assert r.status_code == 200
        body = r.json()
        assert "hits" in body
        assert len(body["hits"]) == 2
        distances = [h["distance"] for h in body["hits"]]
        assert distances == sorted(distances)

    def test_search_empty_index_returns_empty(self, client: TestClient) -> None:
        r = client.post(
            "/experience/search",
            json={"scene": _full_payload()["scene"], "top_k": 5},
        )
        assert r.status_code == 200
        assert r.json()["hits"] == []


# ---------------------------------------------------------------------- #
# GET /experience/statistics
# ---------------------------------------------------------------------- #


class TestStatistics:
    def test_statistics_returns_json(self, client: TestClient) -> None:
        for _ in range(2):
            client.post("/experience/add", json=_full_payload())

        r = client.get("/experience")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 2
        assert body["avg_pdr"] is not None
        assert "w_distance" in body["parameter_distribution"]

    def test_statistics_export_returns_csv(self, client: TestClient) -> None:
        client.post("/experience/add", json=_full_payload())
        r = client.get("/experience?export=true")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")
        text = r.text
        assert "# SECTION" in text
        assert "SUMMARY" in text
        assert "EXPERIENCES" in text


# ---------------------------------------------------------------------- #
# 禁用开关
# ---------------------------------------------------------------------- #


class TestDisabled:
    def test_disabled_returns_503(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from experience import engine as exp_engine
        from experience.config import ExperienceConfig

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

        # patch repository 内使用的 get_config
        from experience import repository as exp_repo
        original_get_config = exp_repo.get_config
        exp_repo.get_config = lambda: cfg_disabled  # type: ignore[assignment]
        try:
            r = client.post("/experience/add", json=_full_payload())
            assert r.status_code == 503
        finally:
            exp_repo.get_config = original_get_config  # type: ignore[assignment]
            exp_engine.reset_faiss()
