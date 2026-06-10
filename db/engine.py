"""SQLAlchemy engine 延迟初始化。

- 模块导入时不连接数据库。
- 第一次调用 ``init_engine()`` 或 ``get_engine()`` 时才创建连接。
- ``pool_pre_ping=True`` 在连接失效时自动重连，适合 MySQL 容器重启场景。
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.config import DBConfig

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def init_engine(config: DBConfig | None = None) -> Engine:
    """创建（或返回已存在的）SQLAlchemy engine。

    幂等：多次调用只创建一次。
    """
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine

    cfg = config or DBConfig.from_env()
    _engine = create_engine(
        cfg.url(),
        pool_size=cfg.pool_size,
        pool_recycle=cfg.pool_recycle,
        pool_pre_ping=True,
        future=True,
        echo=cfg.echo,
    )
    _SessionLocal = sessionmaker(
        bind=_engine, autoflush=False, autocommit=False, future=True
    )
    return _engine


def get_engine() -> Engine:
    """获取 engine，未初始化时自动初始化。"""
    if _engine is None:
        init_engine()
    return _engine  # type: ignore[return-value]


def get_session() -> Session:
    """创建新的 Session。调用方负责 ``close()``，建议用 ``with`` 上下文。"""
    if _SessionLocal is None:
        init_engine()
    assert _SessionLocal is not None
    return _SessionLocal()


def reset_engine() -> None:
    """重置 engine（用于测试或重新加载配置）。"""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
