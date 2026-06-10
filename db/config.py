"""数据库配置：从环境变量构建 SQLAlchemy URL。

支持的环境变量：
- ``MYSQL_HOST``     默认 ``127.0.0.1``
- ``MYSQL_PORT``     默认 ``3306``
- ``MYSQL_USER``     默认 ``root``
- ``MYSQL_PASSWORD`` 默认空字符串
- ``MYSQL_DATABASE`` 默认 ``uav_simulation``
- ``MYSQL_ECHO``     默认 ``0``；设为 ``1`` 打开 SQLAlchemy SQL echo
- ``MYSQL_DISABLED`` 默认 ``0``；设为 ``1`` 时 ``write_simulation_record`` 跳过
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DBConfig:
    """MySQL 连接配置。"""

    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "uav_simulation"
    echo: bool = False
    pool_size: int = 5
    pool_recycle: int = 1800

    @classmethod
    def from_env(cls) -> "DBConfig":
        """从环境变量构造配置。"""
        return cls(
            host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
            port=int(os.environ.get("MYSQL_PORT", "3306")),
            user=os.environ.get("MYSQL_USER", "root"),
            password=os.environ.get("MYSQL_PASSWORD", ""),
            database=os.environ.get("MYSQL_DATABASE", "uav_simulation"),
            echo=os.environ.get("MYSQL_ECHO", "0") == "1",
        )

    def url(self, driver: str = "pymysql") -> str:
        """生成 SQLAlchemy 连接 URL。"""
        return (
            f"mysql+{driver}://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}?charset=utf8mb4"
        )


def is_mysql_disabled() -> bool:
    """检查是否通过环境变量禁用了 MySQL 写入。"""
    return os.environ.get("MYSQL_DISABLED", "0") == "1"
