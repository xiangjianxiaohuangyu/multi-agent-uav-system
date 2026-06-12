"""经验库 REST API 的 Pydantic v2 模型。

请求体一律 ``extra="forbid"`` 强校验；响应体用 ``dict`` 透传。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------- #
# 23 维场景特征
# ---------------------------------------------------------------------- #


class SceneFeature(BaseModel):
    """23 维定长场景特征向量。

    字段顺序与 ``SCENE_FIELD_ORDER`` 严格一致；
    任何额外字段都会被 Pydantic v2 的 ``extra="forbid"`` 拒绝。
    """

    model_config = ConfigDict(extra="forbid")

    speed: float
    energy: float
    queue_length: int | float = 0
    neighbor_count: int | float = 0
    distance_to_destination: float
    forward_candidate_ratio: float
    distance_to_me_mean: float
    distance_to_me_std: float
    distance_to_destination_mean: float
    distance_to_destination_std: float
    distance_to_destination_min: float
    relative_speed_mean: float
    relative_speed_std: float
    link_lifetime_mean: float
    link_lifetime_std: float
    neighbor_degree_mean: float
    neighbor_degree_std: float
    queue_length_mean: float
    queue_length_std: float
    queue_length_max: int | float = 0
    energy_mean: float
    energy_std: float
    energy_min: float


# ---------------------------------------------------------------------- #
# 6 字段参数（键名严格对齐 NS3 扁平 payload）
# ---------------------------------------------------------------------- #


class Parameter(BaseModel):
    """路由参数（6 字段）。"""

    model_config = ConfigDict(extra="forbid")

    hello_interval: float = Field(ge=0)
    path_num: int = Field(ge=0)
    w_distance: float = Field(ge=0, le=1)
    w_linkTime: float = Field(ge=0, le=1)
    w_relVelocity: float = Field(ge=0, le=1)
    w_neighborCount: float = Field(ge=0, le=1)


# ---------------------------------------------------------------------- #
# 2 字段结果（键名严格对齐 NS3 扁平 payload）
# ---------------------------------------------------------------------- #


class Result(BaseModel):
    """性能结果（2 字段）。"""

    model_config = ConfigDict(extra="forbid")

    avg_pdr: float = Field(ge=0, le=1)
    avg_delay: float = Field(ge=0)


# ---------------------------------------------------------------------- #
# 请求 / 响应
# ---------------------------------------------------------------------- #


class AddExperienceRequest(BaseModel):
    """POST /experience/add 请求体。"""

    model_config = ConfigDict(extra="forbid")

    scene: SceneFeature
    parameter: Parameter
    result: Result


class AddExperienceResponse(BaseModel):
    experience_id: int


class SearchExperienceRequest(BaseModel):
    """POST /experience/search 请求体。"""

    model_config = ConfigDict(extra="forbid")

    scene: SceneFeature
    top_k: int = 5


class ExperienceOut(BaseModel):
    """单条经验的完整输出。"""

    experience_id: int
    scene: dict[str, Any]
    scene_vector: list[float]
    parameter: dict[str, Any]
    result: dict[str, Any]
    distance: float | None = None
    created_time: datetime | None = None


class SearchExperienceResponse(BaseModel):
    hits: list[ExperienceOut]


# ---------------------------------------------------------------------- #
# 统计
# ---------------------------------------------------------------------- #


class ParameterBucket(BaseModel):
    """单个字段的分布（桶名 → 计数）。"""

    field: str
    buckets: dict[str, int]


class StatisticsResponse(BaseModel):
    count: int
    parameter_distribution: dict[str, dict[str, int]]
    scene_distribution: dict[str, dict[str, int]]
