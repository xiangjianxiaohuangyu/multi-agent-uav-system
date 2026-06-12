"""经验库配置。

设计原则（与 ``db.config`` 对齐）：
- 不可变 dataclass，避免运行时被改。
- YAML 提供人类可审阅的默认值；环境变量在运行时覆盖。
- 配置加载失败回退到代码内默认值，永不抛出。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


# 经验库被禁用时所有端点返回 503，所有写入路径直接跳过。
def is_experience_disabled() -> bool:
    return os.environ.get("EXPERIENCE_DISABLED", "0") == "1"


@dataclass(frozen=True)
class ExperienceConfig:
    """经验库全局配置。"""

    dimension: int = 23
    faiss_index_path: Path = field(default_factory=lambda: Path("./data/faiss/index.faiss"))
    faiss_id_map_path: Path = field(default_factory=lambda: Path("./data/faiss/id_map.json"))
    topk_default: int = 5
    topk_max: int = 50
    disabled: bool = False
    auto_create_table: bool = True
    log_level: str = "INFO"

    # ------------------------------------------------------------------ #
    # YAML + env 加载
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env_and_yaml(
        cls,
        yaml_path: str | os.PathLike[str] | None = None,
    ) -> "ExperienceConfig":
        """从 YAML 文件加载默认值，再用环境变量覆盖。

        Args:
            yaml_path: YAML 文件路径；``None`` 时使用包内 ``default_config.yaml``。
        """
        cfg = cls()
        # 1. 加载 YAML（如果存在）
        path = Path(yaml_path) if yaml_path else _default_yaml_path()
        if path and path.is_file():
            try:
                with path.open("r", encoding="utf-8") as f:
                    raw: dict[str, Any] = yaml.safe_load(f) or {}
                cfg = _merge_from_dict(cfg, raw)
                log.debug("experience config loaded from %s", path)
            except Exception as exc:  # noqa: BLE001
                log.warning("failed to load experience yaml %s: %s; using defaults", path, exc)
        else:
            log.debug("experience yaml not found at %s; using built-in defaults", path)

        # 2. 环境变量覆盖
        cfg = _apply_env_overrides(cfg)

        # 3. 校验 / 警告
        cfg = _validate(cfg)
        return cfg

    def is_disabled(self) -> bool:
        return self.disabled

    def topk_capped(self, k: int) -> int:
        """把用户传的 ``k`` 截断到 ``[1, topk_max]`` 之间。"""
        try:
            k_int = int(k)
        except (TypeError, ValueError):
            k_int = self.topk_default
        return max(1, min(k_int, self.topk_max))


# ---------------------------------------------------------------------- #
# 内部辅助
# ---------------------------------------------------------------------- #


def _default_yaml_path() -> Path:
    """``experience/default_config.yaml`` 的绝对路径。"""
    return Path(__file__).resolve().parent / "default_config.yaml"


def _merge_from_dict(cfg: ExperienceConfig, raw: dict[str, Any]) -> ExperienceConfig:
    """把 YAML dict 合并进现有 config。缺失键保持不变。"""
    updates: dict[str, Any] = {}

    if "dimension" in raw:
        try:
            updates["dimension"] = int(raw["dimension"])
        except (TypeError, ValueError):
            pass

    if "faiss_index_path" in raw:
        updates["faiss_index_path"] = Path(str(raw["faiss_index_path"]))
    if "faiss_id_map_path" in raw:
        updates["faiss_id_map_path"] = Path(str(raw["faiss_id_map_path"]))

    if "topk_default" in raw:
        updates["topk_default"] = int(raw["topk_default"])
    if "topk_max" in raw:
        updates["topk_max"] = int(raw["topk_max"])
    if "auto_create_table" in raw:
        updates["auto_create_table"] = bool(raw["auto_create_table"])
    if "log_level" in raw:
        updates["log_level"] = str(raw["log_level"])

    return replace(cfg, **updates) if updates else cfg


def _apply_env_overrides(cfg: ExperienceConfig) -> ExperienceConfig:
    """把环境变量叠加到 config。"""
    updates: dict[str, Any] = {}

    if "EXPERIENCE_DISABLED" in os.environ:
        updates["disabled"] = os.environ["EXPERIENCE_DISABLED"] == "1"

    if "EXPERIENCE_FAISS_DIR" in os.environ:
        d = Path(os.environ["EXPERIENCE_FAISS_DIR"])
        updates["faiss_index_path"] = d / "index.faiss"
        updates["faiss_id_map_path"] = d / "id_map.json"

    if "EXPERIENCE_TOPK_DEFAULT" in os.environ:
        try:
            updates["topk_default"] = int(os.environ["EXPERIENCE_TOPK_DEFAULT"])
        except ValueError:
            pass
    if "EXPERIENCE_TOPK_MAX" in os.environ:
        try:
            updates["topk_max"] = int(os.environ["EXPERIENCE_TOPK_MAX"])
        except ValueError:
            pass

    if "EXPERIENCE_LOG_LEVEL" in os.environ:
        updates["log_level"] = str(os.environ["EXPERIENCE_LOG_LEVEL"])

    return replace(cfg, **updates) if updates else cfg


def _validate(cfg: ExperienceConfig) -> ExperienceConfig:
    """配置合理性检查；不合法时打 WARNING 并修正。"""
    if cfg.dimension <= 0:
        log.warning("invalid dimension %s, fallback to 23", cfg.dimension)
        return replace(cfg, dimension=23)
    if cfg.topk_default < 1:
        log.warning("invalid topk_default %s, fallback to 5", cfg.topk_default)
        return replace(cfg, topk_default=5)
    if cfg.topk_max < cfg.topk_default:
        log.warning("topk_max < topk_default, adjusting")
        return replace(cfg, topk_max=cfg.topk_default)
    return cfg
