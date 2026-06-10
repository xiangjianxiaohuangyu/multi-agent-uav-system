"""experience 包：FAISS 经验库。

公共 re-export（与 ``db/__init__.py`` 对齐）。
"""

from experience.config import ExperienceConfig, ScoreWeights, is_experience_disabled
from experience.engine import (
    get_config,
    get_faiss_manager,
    init_faiss,
    is_initialized,
    reset_faiss,
)
from experience.faiss_manager import FaissIndexManager
from experience.models import Experience
from experience.repository import ExperienceRepository
from experience.scoring import (
    PARAM_FIELDS,
    RESULT_FIELDS,
    SCENE_DIMENSION,
    SCENE_FIELD_ORDER,
    compute_score,
    payload_to_experience,
    scene_to_vector,
)

__all__ = [
    # 配置
    "ExperienceConfig",
    "ScoreWeights",
    "is_experience_disabled",
    # ORM
    "Experience",
    # 仓储
    "ExperienceRepository",
    # FAISS
    "FaissIndexManager",
    # 引擎
    "init_faiss",
    "get_faiss_manager",
    "get_config",
    "reset_faiss",
    "is_initialized",
    # Scoring 常量与纯函数
    "SCENE_FIELD_ORDER",
    "SCENE_DIMENSION",
    "PARAM_FIELDS",
    "RESULT_FIELDS",
    "compute_score",
    "scene_to_vector",
    "payload_to_experience",
]
