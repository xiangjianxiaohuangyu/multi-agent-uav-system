"""db 包：MySQL 持久化层。

设计原则：
- 写入路径永不抛出异常（``write_simulation_record`` 始终返回 bool）。
- engine 延迟初始化，服务器启动不依赖数据库可达性。
- 失败仅记录日志，不影响 JSON 落盘或 HTTP 200 返回。
"""

from db.config import DBConfig
from db.engine import get_engine, get_session, init_engine
from db.models import Base, SimulationRecord
from db.repository import write_simulation_record

__all__ = [
    "DBConfig",
    "Base",
    "SimulationRecord",
    "init_engine",
    "get_engine",
    "get_session",
    "write_simulation_record",
]
