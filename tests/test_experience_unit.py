"""经验库单元测试：纯函数无 IO。

运行：
    pytest tests/test_experience_unit.py -v
"""

from __future__ import annotations

import math

import pytest

from experience.config import ExperienceConfig, ScoreWeights
from experience.scoring import (
    PARAM_FIELDS,
    RESULT_FIELDS,
    SCENE_DIMENSION,
    SCENE_FIELD_ORDER,
    compute_score,
    payload_to_experience,
    scene_to_vector,
    validate_parameter,
    validate_result,
    validate_scene,
)


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #


@pytest.fixture
def cfg() -> ExperienceConfig:
    return ExperienceConfig(
        dimension=11,
        max_delay_ms=1000.0,
        max_energy=100.0,
        score_weights=ScoreWeights(pdr=0.5, delay=0.3, energy=0.2),
    )


def _full_scene() -> dict:
    return {k: float(i + 1) for i, k in enumerate(SCENE_FIELD_ORDER)}


# ---------------------------------------------------------------------- #
# compute_score
# ---------------------------------------------------------------------- #


class TestComputeScore:
    def test_pure_pdr(self, cfg: ExperienceConfig) -> None:
        # pdr=1.0, delay=0, energy=0 → score = 0.5*1 + 0.3*1 + 0.2*1 = 1.0
        assert math.isclose(compute_score(1.0, 0.0, 0.0, cfg), 1.0, rel_tol=1e-9)

    def test_zero_pdr_max_delay_max_energy(self, cfg: ExperienceConfig) -> None:
        # pdr=0, delay=max → clamp(1,0,1) = 1 → 1 - 1 = 0
        # energy=max → 同上
        assert math.isclose(compute_score(0.0, 1000.0, 100.0, cfg), 0.0, abs_tol=1e-9)

    def test_clamp_pdr_above_one(self, cfg: ExperienceConfig) -> None:
        # pdr>1 → clamp 到 1
        s = compute_score(2.0, 0.0, 0.0, cfg)
        assert math.isclose(s, 1.0, rel_tol=1e-9)

    def test_clamp_negative_pdr(self, cfg: ExperienceConfig) -> None:
        s = compute_score(-0.5, 0.0, 0.0, cfg)
        # pdr_n=0, delay_n=1, energy_n=1 → 0 + 0.3 + 0.2 = 0.5
        assert math.isclose(s, 0.5, abs_tol=1e-9)

    def test_clamp_delay_above_max(self, cfg: ExperienceConfig) -> None:
        # delay=2000 → 1 - clamp(2.0, 0, 1) = 0
        s = compute_score(1.0, 2000.0, 0.0, cfg)
        assert math.isclose(s, 0.5 + 0.2, abs_tol=1e-9)  # 0.7

    def test_clamp_energy_above_max(self, cfg: ExperienceConfig) -> None:
        s = compute_score(1.0, 0.0, 500.0, cfg)
        assert math.isclose(s, 0.5 + 0.3, abs_tol=1e-9)  # 0.8

    def test_clamp_negative_delay(self, cfg: ExperienceConfig) -> None:
        # 负 delay → 1 - clamp(-0.5/1000, 0, 1) = 1
        s = compute_score(0.0, -1.0, 0.0, cfg)
        assert math.isclose(s, 0.3 + 0.2, abs_tol=1e-9)  # 0.5

    def test_score_in_unit_interval(self, cfg: ExperienceConfig) -> None:
        for pdr in (-1, 0, 0.5, 1, 2):
            for delay in (-100, 0, 500, 1000, 1500):
                for energy in (-10, 0, 50, 100, 200):
                    s = compute_score(pdr, delay, energy, cfg)
                    assert 0.0 <= s <= 1.0, f"score out of range: pdr={pdr}, delay={delay}, energy={energy}, score={s}"

    def test_custom_weights(self) -> None:
        cfg = ExperienceConfig(
            dimension=11,
            max_delay_ms=1000.0,
            max_energy=100.0,
            score_weights=ScoreWeights(pdr=0.6, delay=0.3, energy=0.1),
        )
        s = compute_score(1.0, 0.0, 0.0, cfg)
        assert math.isclose(s, 1.0, rel_tol=1e-9)

    def test_non_numeric_inputs_fallback(self, cfg: ExperienceConfig) -> None:
        # "abc" 不会抛；走 _to_float fallback 到 0
        s = compute_score("abc", "xyz", None, cfg)  # type: ignore[arg-type]
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------- #
# scene_to_vector
# ---------------------------------------------------------------------- #


class TestSceneToVector:
    def test_dimension_is_11(self) -> None:
        assert SCENE_DIMENSION == 11
        assert len(SCENE_FIELD_ORDER) == 11

    def test_field_order_exact(self) -> None:
        assert SCENE_FIELD_ORDER == (
            "speed",
            "energy",
            "queue_length",
            "neighbor_count",
            "distance_to_destination",
            "forward_candidate_ratio",
            "avg_neighbor_distance",
            "relative_speed_mean",
            "link_stability",
            "link_lifetime_mean",
            "traffic_load",
        )

    def test_full_scene_preserves_order(self) -> None:
        scene = _full_scene()
        v = scene_to_vector(scene)
        assert len(v) == 11
        for i, k in enumerate(SCENE_FIELD_ORDER):
            assert v[i] == scene[k]

    def test_missing_key_yields_zero(self) -> None:
        scene = {k: 1.0 for k in SCENE_FIELD_ORDER}
        del scene["traffic_load"]
        v = scene_to_vector(scene)
        assert v[-1] == 0.0
        assert v[0] == 1.0  # speed 仍然在第一位

    def test_empty_scene_yields_all_zeros(self) -> None:
        v = scene_to_vector({})
        assert v == [0.0] * 11

    def test_non_dict_returns_zeros(self) -> None:
        v = scene_to_vector("not a dict")  # type: ignore[arg-type]
        assert v == [0.0] * 11

    def test_non_numeric_yields_zero(self) -> None:
        scene = {k: "abc" for k in SCENE_FIELD_ORDER}
        v = scene_to_vector(scene)
        assert v == [0.0] * 11

    def test_nan_yields_zero(self) -> None:
        scene = {**{k: 0.0 for k in SCENE_FIELD_ORDER}, "speed": float("nan")}
        v = scene_to_vector(scene)
        assert v[0] == 0.0


# ---------------------------------------------------------------------- #
# 校验
# ---------------------------------------------------------------------- #


class TestValidation:
    def test_validate_scene_ok(self) -> None:
        validate_scene(_full_scene())  # 不抛

    def test_validate_scene_not_dict(self) -> None:
        with pytest.raises(ValueError):
            validate_scene("not dict")  # type: ignore[arg-type]

    def test_validate_scene_non_numeric(self) -> None:
        scene = _full_scene()
        scene["speed"] = "abc"
        with pytest.raises(ValueError):
            validate_scene(scene)

    def test_validate_parameter_ok(self) -> None:
        validate_parameter(
            {
                "hello_interval": 1.0,
                "candidate_num": 2,
                "w_distance": 0.4,
                "w_linktime": 0.3,
                "w_energy": 0.2,
                "w_queue": 0.1,
            }
        )

    def test_validate_parameter_missing_field(self) -> None:
        with pytest.raises(ValueError):
            validate_parameter({"hello_interval": 1.0})

    def test_validate_parameter_weight_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            validate_parameter(
                {
                    "hello_interval": 1.0,
                    "candidate_num": 2,
                    "w_distance": 1.5,  # > 1
                    "w_linktime": 0.3,
                    "w_energy": 0.2,
                    "w_queue": 0.1,
                }
            )

    def test_validate_result_ok(self) -> None:
        validate_result(
            {
                "e2e_pdr": 0.9,
                "e2e_delay": 100.0,
                "routing_overhead": 10.0,
                "energy_consumption": 5.0,
            }
        )

    def test_validate_result_missing(self) -> None:
        with pytest.raises(ValueError):
            validate_result({"e2e_pdr": 0.9})


# ---------------------------------------------------------------------- #
# payload_to_experience
# ---------------------------------------------------------------------- #


def _make_payload() -> dict:
    return {
        "type": "simulation",
        "task_id": "abc",
        "simulation_time": 1.0,
        "nodes": [
            {
                "id": 0,
                "scene_info": {
                    "m_info": {
                        "speed": 5.0,
                        "energy": 80.0,
                        "queue_length": 2,
                        "neighbor_count": 4,
                        "distance_to_destination": 120.0,
                    },
                    "neighbor_info": {
                        "forward_candidate_ratio": 0.6,
                        "distance_to_me_mean": 50.0,
                        "relative_speed_mean": 1.2,
                        "link_lifetime_mean": 30.0,
                        "queue_length_mean": 2.0,
                    },
                },
                "para_info": {
                    "hello_interval": 1.0,
                    "path_num": 2,
                    "weights": {
                        "w_distance": 0.4,
                        "w_linkTime": 0.3,
                        "w_energy": 0.2,
                        "w_queue": 0.1,
                    },
                },
                "result_info": {
                    "e2e_pdr": 0.92,
                    "e2e_delay": 150.0,
                    "routing_overhead": 12.0,
                    "energy_consumption": 8.5,
                },
            }
        ],
    }


class TestPayloadMapping:
    def test_full_payload_maps(self) -> None:
        mapped = payload_to_experience(_make_payload())
        assert mapped is not None
        scene, param, result = mapped
        # 11 维都在
        for k in SCENE_FIELD_ORDER:
            assert k in scene
        # 6 字段都在
        for k in PARAM_FIELDS:
            assert k in param
        # 4 字段都在
        for k in RESULT_FIELDS:
            assert k in result

    def test_scene_speed_mapping(self) -> None:
        mapped = payload_to_experience(_make_payload())
        assert mapped is not None
        scene, _, _ = mapped
        assert scene["speed"] == 5.0
        assert scene["energy"] == 80.0

    def test_link_stability_normalized(self) -> None:
        # link_lifetime_mean=30s → link_stability = clamp(30/60) = 0.5
        mapped = payload_to_experience(_make_payload())
        assert mapped is not None
        scene, _, _ = mapped
        assert 0.0 <= scene["link_stability"] <= 1.0
        assert scene["link_stability"] == pytest.approx(0.5, abs=1e-6)

    def test_traffic_load_normalized(self) -> None:
        # queue_length_mean=2.0 → traffic_load = clamp(2/10) = 0.2
        mapped = payload_to_experience(_make_payload())
        assert mapped is not None
        scene, _, _ = mapped
        assert scene["traffic_load"] == pytest.approx(0.2, abs=1e-6)

    def test_parameter_candidate_num_uses_path_num(self) -> None:
        mapped = payload_to_experience(_make_payload())
        assert mapped is not None
        _, param, _ = mapped
        assert param["candidate_num"] == 2
        assert param["hello_interval"] == 1.0

    def test_parameter_w_linktime_alias(self) -> None:
        mapped = payload_to_experience(_make_payload())
        assert mapped is not None
        _, param, _ = mapped
        # 原始用 w_linkTime，映射到 w_linktime（小写）
        assert param["w_linktime"] == 0.3

    def test_result_mapping(self) -> None:
        mapped = payload_to_experience(_make_payload())
        assert mapped is not None
        _, _, result = mapped
        assert result["e2e_pdr"] == 0.92
        assert result["e2e_delay"] == 150.0
        assert result["routing_overhead"] == 12.0
        assert result["energy_consumption"] == 8.5

    def test_scene_params_returns_none(self) -> None:
        assert payload_to_experience({"type": "scene_params"}) is None

    def test_empty_nodes_returns_none(self) -> None:
        assert payload_to_experience({"type": "simulation", "nodes": []}) is None

    def test_no_scene_info_returns_none(self) -> None:
        p = _make_payload()
        del p["nodes"][0]["scene_info"]
        assert payload_to_experience(p) is None

    def test_missing_weight_fields_default_to_zero(self) -> None:
        p = _make_payload()
        del p["nodes"][0]["para_info"]["weights"]["w_energy"]
        del p["nodes"][0]["para_info"]["weights"]["w_queue"]
        mapped = payload_to_experience(p)
        assert mapped is not None
        _, param, _ = mapped
        assert param["w_energy"] == 0.0
        assert param["w_queue"] == 0.0

    def test_fallback_to_legacy_field_names(self) -> None:
        """兼容没有 w_* 命名的旧 payload（用 distance/linkTime 等）。"""
        p = _make_payload()
        p["nodes"][0]["para_info"]["weights"] = {
            "distance": 0.4,
            "linkTime": 0.3,
            "relVelocity": 0.2,
            "neighborCount": 0.1,
        }
        mapped = payload_to_experience(p)
        assert mapped is not None
        _, param, _ = mapped
        assert param["w_distance"] == 0.4
        assert param["w_linktime"] == 0.3
        assert param["w_energy"] == 0.0
        assert param["w_queue"] == 0.0

    def test_fallback_result_aliases(self) -> None:
        """旧 result 字段（avg_pdr/avg_delay/control_packets）。"""
        p = _make_payload()
        p["nodes"][0]["result_info"] = {
            "avg_pdr": 0.8,
            "avg_delay": 200.0,
            "control_packets": 15.0,
            "energy_consumption": 7.0,
        }
        mapped = payload_to_experience(p)
        assert mapped is not None
        _, _, result = mapped
        assert result["e2e_pdr"] == 0.8
        assert result["e2e_delay"] == 200.0
        assert result["routing_overhead"] == 15.0
        assert result["energy_consumption"] == 7.0

    def test_non_dict_payload_returns_none(self) -> None:
        assert payload_to_experience("not a dict") is None  # type: ignore[arg-type]
        assert payload_to_experience(None) is None  # type: ignore[arg-type]
