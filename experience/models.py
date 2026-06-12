"""经验库 ORM 模型。

``Experience`` 表是 FAISS 经验库的 MySQL 持久层，与 ``simulation_records`` 并存。
继承自 ``db.models.Base``，以便 ``Base.metadata.create_all`` 一次性建出两张表。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, func
from sqlalchemy.dialects.mysql import BIGINT, JSON as MySQLJSON
from sqlalchemy.orm import Mapped, mapped_column

from db.models import Base


class Experience(Base):
    """一条决策经验。

    列含义详见 ``db/ddl/002_create_experiences.sql``。
    """

    __tablename__ = "experiences"

    experience_id: Mapped[int] = mapped_column(
        BigInteger_unsigned := BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True,
    )
    scene_json: Mapped[dict] = mapped_column(MySQLJSON, nullable=False)
    scene_vector: Mapped[list] = mapped_column(MySQLJSON, nullable=False)
    parameter_json: Mapped[dict] = mapped_column(MySQLJSON, nullable=False)
    result_json: Mapped[dict] = mapped_column(MySQLJSON, nullable=False)
    created_time: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )

    __table_args__ = (
        Index("idx_created_desc", "created_time"),
        {
            "mysql_engine": "InnoDB",
            "mysql_charset": "utf8mb4",
            "mysql_collate": "utf8mb4_unicode_ci",
        },
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Experience id={self.experience_id} "
            f"created={self.created_time!r}>"
        )
