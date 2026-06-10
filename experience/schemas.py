"""经验库 REST API 的 Pydantic v2 模型。

请求体一律 ``extra="forbid"`` 强校验；响应体用 ``dict`` 透传。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------- #
# 11 维场景特征
# ---------------------------------------------------------------------- #


class SceneFeature(BaseModel):
    """11 维定长场景特征向量。"""

    model_config = ConfigDict(extra="forbid")

    speed: float
    energy: float
    queue_length: int | float = 0
    neighbor_count: int | float = 0
    distance_to_destination: float
    forward_candidate_ratio: float
    avg_neighbor_distance: float
    relative_speed_mean: float
    link_stability: float
    link_lifetime_mean: float
    traffic_load: float


# ---------------------------------------------------------------------- #
# 6 字段参数
# ---------------------------------------------------------------------- #


class Parameter(BaseModel):
    """路由参数（6 字段）。"""

    model_config = ConfigDict(extra="forbid")

    hello_interval: float = Field(ge=0)
    candidate_num: int = Field(ge=0)
    w_distance: float = Field(ge=0, le=1)
    w_linktime: float = Field(ge=0, le=1)
    w_energy: float = Field(ge=0, le=1)
    w_queue: float = Field(ge=0, le=1)


# ---------------------------------------------------------------------- #
# 4 字段结果
# ---------------------------------------------------------------------- #


class Result(BaseModel):
    """性能结果（4 字段）。"""

    model_config = ConfigDict(extra="forbid")

    e2e_pdr: float = Field(ge=0, le=1)
    e2e_delay: float = Field(ge=0)
    routing_overhead: float = Field(ge=0)
    energy_consumption: float = Field(ge=0)


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
    score: float


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
    score: float
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
    avg_pdr: float | None
    avg_delay: float | None
    avg_score: float | None
    parameter_distribution: dict[str, dict[str, int]]
    scene_distribution: dict[str, dict[str, int]]
