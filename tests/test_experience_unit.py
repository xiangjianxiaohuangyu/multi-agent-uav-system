"""经验库单元测试：纯函数无 IO。

运行：
    pytest tests/test_experience_unit.py -v
"""

from __future__ import annotations

import pytest

from experience.config import ExperienceConfig
from experience.scoring import (
    PARAM_FIELDS,
    RESULT_FIELDS,
    SCENE_DIMENSION,
    SCENE_FIELD_ORDER,
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
    return ExperienceConfig(dimension=23)


def _full_scene() -> dict:
    return {k: float(i + 1) for i, k in enumerate(SCENE_FIELD_ORDER)}


# ---------------------------------------------------------------------- #
# scene_to_vector
# ---------------------------------------------------------------------- #


class TestSceneToVector:
    def test_dimension_is_23(self) -> None:
        assert SCENE_DIMENSION == 23
        assert len(SCENE_FIELD_ORDER) == 23

    def test_field_order_exact(self) -> None:
        assert SCENE_FIELD_ORDER == (
            "speed",
            "energy",
            "queue_length",
            "neighbor_count",
            "distance_to_destination",
            "forward_candidate_ratio",
            "distance_to_me_mean",
            "distance_to_me_std",
            "distance_to_destination_mean",
            "distance_to_destination_std",
            "distance_to_destination_min",
            "relative_speed_mean",
            "relative_speed_std",
            "link_lifetime_mean",
            "link_lifetime_std",
            "neighbor_degree_mean",
            "neighbor_degree_std",
            "queue_length_mean",
            "queue_length_std",
            "queue_length_max",
            "energy_mean",
            "energy_std",
            "energy_min",
        )

    def test_full_scene_preserves_order(self) -> None:
        scene = _full_scene()
        v = scene_to_vector(scene)
        assert len(v) == 23
        for i, k in enumerate(SCENE_FIELD_ORDER):
            assert v[i] == scene[k]

    def test_missing_key_yields_zero(self) -> None:
        scene = {k: 1.0 for k in SCENE_FIELD_ORDER}
        del scene["energy_min"]
        v = scene_to_vector(scene)
        assert v[-1] == 0.0
        assert v[0] == 1.0  # speed 仍然在第一位

    def test_empty_scene_yields_all_zeros(self) -> None:
        v = scene_to_vector({})
        assert v == [0.0] * 23

    def test_non_dict_returns_zeros(self) -> None:
        v = scene_to_vector("not a dict")  # type: ignore[arg-type]
        assert v == [0.0] * 23

    def test_non_numeric_yields_zero(self) -> None:
        scene = {k: "abc" for k in SCENE_FIELD_ORDER}
        v = scene_to_vector(scene)
        assert v == [0.0] * 23

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
                "path_num": 2,
                "w_distance": 0.4,
                "w_linkTime": 0.3,
                "w_relVelocity": 0.2,
                "w_neighborCount": 0.1,
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
                    "path_num": 2,
                    "w_distance": 1.5,  # > 1
                    "w_linkTime": 0.3,
                    "w_relVelocity": 0.2,
                    "w_neighborCount": 0.1,
                }
            )

    def test_validate_result_ok(self) -> None:
        validate_result(
            {
                "avg_pdr": 0.9,
                "avg_delay": 100.0,
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
        "hello_interval": 1.0,
        "path_num": 2,
        "w_distance": 0.4,
        "w_linkTime": 0.3,
        "w_relVelocity": 0.2,
        "w_neighborCount": 0.1,
        "avg_pdr": 0.92,
        "avg_delay": 150.0,
    }


class TestPayloadMapping:
    def test_full_payload_maps(self) -> None:
        mapped = payload_to_experience(_make_payload())
        assert mapped is not None
        scene, param, result = mapped
        # 23 维都在
        assert len(scene) == 23
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

    def test_link_lifetime_mean_raw(self) -> None:
        # link_lifetime_mean 现在是原始值，不再归一化
        mapped = payload_to_experience(_make_payload())
        assert mapped is not None
        scene, _, _ = mapped
        assert scene["link_lifetime_mean"] == pytest.approx(30.0, abs=1e-6)
        assert scene["link_lifetime_std"] == pytest.approx(4.0, abs=1e-6)

    def test_queue_length_mean_raw(self) -> None:
        # queue_length_mean 现在是原始值，不再归一化
        mapped = payload_to_experience(_make_payload())
        assert mapped is not None
        scene, _, _ = mapped
        assert scene["queue_length_mean"] == pytest.approx(2.0, abs=1e-6)
        assert scene["queue_length_max"] == 4

    def test_parameter_path_num_mapping(self) -> None:
        mapped = payload_to_experience(_make_payload())
        assert mapped is not None
        _, param, _ = mapped
        assert param["path_num"] == 2
        assert param["hello_interval"] == 1.0

    def test_parameter_w_linktime_case_sensitive(self) -> None:
        """``w_linkTime`` 大写 T 是 NS3 扁平 payload 的规范键名。"""
        mapped = payload_to_experience(_make_payload())
        assert mapped is not None
        _, param, _ = mapped
        assert param["w_linkTime"] == 0.3
        assert param["w_distance"] == 0.4
        assert param["w_relVelocity"] == 0.2
        assert param["w_neighborCount"] == 0.1

    def test_result_mapping(self) -> None:
        mapped = payload_to_experience(_make_payload())
        assert mapped is not None
        _, _, result = mapped
        assert result["avg_pdr"] == 0.92
        assert result["avg_delay"] == 150.0

    def test_scene_params_returns_none(self) -> None:
        assert payload_to_experience({"type": "scene_params"}) is None

    def test_empty_payload_returns_none(self) -> None:
        """空 payload 应当返回 None（让经验库跳过空事件）。"""
        assert payload_to_experience({}) is None

    def test_payload_without_scene_fields_returns_tuple_with_zeros(self) -> None:
        """所有 23 维 scene 字段缺失时，仍返回 scene 全 0 的三元组，
        让经验库至少能记录这一次"参数 + 结果"事件。
        """
        p = {
            "type": "simulation",
            "hello_interval": 1.0,
            "path_num": 1,
            "w_distance": 0.4,
            "w_linkTime": 0.3,
            "w_relVelocity": 0.2,
            "w_neighborCount": 0.1,
            "avg_pdr": 0.92,
            "avg_delay": 150.0,
        }
        mapped = payload_to_experience(p)
        assert mapped is not None
        scene, param, result = mapped
        # scene 字段全部为 0（23 维）
        assert all(v == 0 for v in scene.values())
        # param/result 仍然能被正常提取
        assert param["hello_interval"] == 1.0
        assert result["avg_pdr"] == 0.92

    def test_missing_weight_fields_default_to_zero(self) -> None:
        """缺个别 weight 字段时，对应位置应回退为 0。"""
        p = _make_payload()
        del p["w_relVelocity"]
        del p["w_neighborCount"]
        mapped = payload_to_experience(p)
        assert mapped is not None
        _, param, _ = mapped
        assert param["w_relVelocity"] == 0.0
        assert param["w_neighborCount"] == 0.0
        # 其它字段不受影响
        assert param["w_distance"] == 0.4
        assert param["w_linkTime"] == 0.3

    def test_missing_result_fields_default_to_zero(self) -> None:
        """缺 result 字段时，对应位置应回退为 0。"""
        p = _make_payload()
        del p["avg_pdr"]
        mapped = payload_to_experience(p)
        assert mapped is not None
        _, _, result = mapped
        assert result["avg_pdr"] == 0.0
        assert result["avg_delay"] == pytest.approx(150.0, abs=1e-6)

    def test_missing_path_num_defaults_to_zero(self) -> None:
        """``path_num`` 缺失时回退为 0。"""
        p = _make_payload()
        del p["path_num"]
        mapped = payload_to_experience(p)
        assert mapped is not None
        _, param, _ = mapped
        assert param["path_num"] == 0
        assert param["hello_interval"] == 1.0

    def test_non_dict_payload_returns_none(self) -> None:
        assert payload_to_experience("not a dict") is None  # type: ignore[arg-type]
        assert payload_to_experience(None) is None  # type: ignore[arg-type]
