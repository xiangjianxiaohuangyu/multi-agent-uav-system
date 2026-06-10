"""FAISS 经验库延迟初始化。

镜像 ``db.engine`` 的模式：
- 模块导入时不连接任何东西。
- 第一次 ``get_faiss_manager()`` 时才创建（自动从 ``ExperienceConfig.from_env_and_yaml()``）。
- 重复调用返回同一实例（进程级单例）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from experience.config import ExperienceConfig
from experience.faiss_manager import FaissIndexManager

log = logging.getLogger(__name__)

_manager: FaissIndexManager | None = None
_config: ExperienceConfig | None = None


def init_faiss(config: ExperienceConfig | None = None) -> FaissIndexManager:
    """创建（或返回已存在的）FAISS 索引管理器。

    Args:
        config: 自定义配置；``None`` 时从 YAML + env 加载。
    """
    global _manager, _config
    if _manager is not None:
        return _manager

    cfg = config or ExperienceConfig.from_env_and_yaml()
    # 写盘前给日志打一条
    log.info(
        "experience: init faiss dim=%d index=%s disabled=%s",
        cfg.dimension, cfg.faiss_index_path, cfg.disabled,
    )

    if cfg.disabled:
        log.warning("experience library disabled by config; FAISS manager is a stub")
        # 即使 disabled，也创建一个 manager 但不 load_or_create（避免无谓写盘）
        _manager = FaissIndexManager(
            dimension=cfg.dimension,
            index_path=cfg.faiss_index_path,
            id_map_path=cfg.faiss_id_map_path,
        )
        _config = cfg
        return _manager

    mgr = FaissIndexManager(
        dimension=cfg.dimension,
        index_path=cfg.faiss_index_path,
        id_map_path=cfg.faiss_id_map_path,
    )
    try:
        mgr.load_or_create()
    except Exception as exc:  # noqa: BLE001
        log.exception("faiss init failed: %s; manager returned but operations may fail", exc)
    _manager = mgr
    _config = cfg
    return _manager


def get_faiss_manager() -> FaissIndexManager:
    """获取 manager，未初始化时自动初始化。"""
    if _manager is None:
        return init_faiss()
    return _manager


def get_config() -> ExperienceConfig:
    """获取当前生效的 config（触发 init）。"""
    if _config is None:
        init_faiss()
    assert _config is not None
    return _config


def reset_faiss() -> None:
    """重置 manager（用于测试或重载配置）。"""
    global _manager, _config
    _manager = None
    _config = None


def is_initialized() -> bool:
    return _manager is not None
