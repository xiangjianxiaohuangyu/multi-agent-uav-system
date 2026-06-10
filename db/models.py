"""ORM 模型。

``SimulationRecord`` 是一行扁平化的 ns-3 仿真回调记录，列含义详见
``db/ddl/001_create_simulation_records.sql``。
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    DECIMAL,
    BigInteger,
    DateTime,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """ORM 基类。"""


class SimulationRecord(Base):
    """单条仿真回调的扁平化记录。"""

    __tablename__ = "simulation_records"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(BIGINT(unsigned=True), "mysql"),
        primary_key=True,
        autoincrement=True,
    )

    # 标识 + 时间
    task_id: Mapped[str] = mapped_column(String(64), nullable=False, default="default")
    device_id: Mapped[str] = mapped_column(String(64), nullable=False, default="unknown")
    simulation_time: Mapped[Decimal] = mapped_column(DECIMAL(12, 3), nullable=False, default=Decimal("0"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )

    # m_info
    m_speed: Mapped[Decimal] = mapped_column(DECIMAL(8, 2), nullable=False, default=Decimal("0"))
    m_energy: Mapped[Decimal] = mapped_column(DECIMAL(8, 2), nullable=False, default=Decimal("0"))
    m_queue_length: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    m_neighbor_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    m_distance_to_destination: Mapped[Decimal] = mapped_column(
        DECIMAL(10, 2), nullable=False, default=Decimal("0")
    )

    # neighbor_info（15 列）
    nb_forward_candidate_ratio: Mapped[Decimal] = mapped_column(
        DECIMAL(4, 3), nullable=False, default=Decimal("0")
    )
    nb_distance_to_me_mean: Mapped[Decimal] = mapped_column(
        DECIMAL(8, 2), nullable=False, default=Decimal("0")
    )
    nb_distance_to_me_std: Mapped[Decimal] = mapped_column(
        DECIMAL(8, 2), nullable=False, default=Decimal("0")
    )
    nb_distance_to_destination_mean: Mapped[Decimal] = mapped_column(
        DECIMAL(10, 2), nullable=False, default=Decimal("0")
    )
    nb_distance_to_destination_std: Mapped[Decimal] = mapped_column(
        DECIMAL(10, 2), nullable=False, default=Decimal("0")
    )
    nb_distance_to_destination_min: Mapped[Decimal] = mapped_column(
        DECIMAL(10, 2), nullable=False, default=Decimal("0")
    )
    # 实测可达 1e+19 级别，DECIMAL(38,4) 是 MySQL DECIMAL 上限
    nb_relative_speed_mean: Mapped[Decimal] = mapped_column(
        DECIMAL(38, 4), nullable=False, default=Decimal("0")
    )
    nb_relative_speed_std: Mapped[Decimal] = mapped_column(
        DECIMAL(38, 4), nullable=False, default=Decimal("0")
    )
    nb_link_lifetime_mean: Mapped[Decimal] = mapped_column(
        DECIMAL(8, 2), nullable=False, default=Decimal("0")
    )
    nb_link_lifetime_std: Mapped[Decimal] = mapped_column(
        DECIMAL(8, 2), nullable=False, default=Decimal("0")
    )
    nb_neighbor_degree_mean: Mapped[Decimal] = mapped_column(
        DECIMAL(6, 2), nullable=False, default=Decimal("0")
    )
    nb_neighbor_degree_std: Mapped[Decimal] = mapped_column(
        DECIMAL(6, 2), nullable=False, default=Decimal("0")
    )
    nb_queue_length_mean: Mapped[Decimal] = mapped_column(
        DECIMAL(8, 2), nullable=False, default=Decimal("0")
    )
    nb_queue_length_std: Mapped[Decimal] = mapped_column(
        DECIMAL(8, 2), nullable=False, default=Decimal("0")
    )
    nb_queue_length_max: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    nb_energy_mean: Mapped[Decimal] = mapped_column(
        DECIMAL(8, 2), nullable=False, default=Decimal("0")
    )
    nb_energy_std: Mapped[Decimal] = mapped_column(
        DECIMAL(8, 2), nullable=False, default=Decimal("0")
    )
    nb_energy_min: Mapped[Decimal] = mapped_column(
        DECIMAL(8, 2), nullable=False, default=Decimal("0")
    )

    # para_info
    param_hello_interval: Mapped[Decimal] = mapped_column(
        DECIMAL(5, 2), nullable=False, default=Decimal("0")
    )
    param_path_num: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    weight_distance: Mapped[Decimal] = mapped_column(
        DECIMAL(4, 3), nullable=False, default=Decimal("0")
    )
    weight_link_time: Mapped[Decimal] = mapped_column(
        DECIMAL(4, 3), nullable=False, default=Decimal("0")
    )
    weight_rel_velocity: Mapped[Decimal] = mapped_column(
        DECIMAL(4, 3), nullable=False, default=Decimal("0")
    )
    weight_neighbor_count: Mapped[Decimal] = mapped_column(
        DECIMAL(4, 3), nullable=False, default=Decimal("0")
    )

    # result_info
    res_avg_pdr: Mapped[Decimal] = mapped_column(
        DECIMAL(5, 4), nullable=False, default=Decimal("0")
    )
    res_avg_delay: Mapped[Decimal] = mapped_column(
        DECIMAL(8, 2), nullable=False, default=Decimal("0")
    )
    res_energy_consumption: Mapped[Decimal] = mapped_column(
        DECIMAL(8, 2), nullable=False, default=Decimal("0")
    )
    res_control_packets: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    res_distance_progress: Mapped[Decimal] = mapped_column(
        DECIMAL(10, 2), nullable=False, default=Decimal("0")
    )

    __table_args__ = (
        Index("idx_device_time", "device_id", "created_at"),
        Index("idx_param_combo", "param_hello_interval", "param_path_num"),
        Index("idx_task_time", "task_id", "simulation_time"),
        {
            "mysql_engine": "InnoDB",
            "mysql_charset": "utf8mb4",
            "mysql_collate": "utf8mb4_unicode_ci",
        },
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<SimulationRecord id={self.id} task_id={self.task_id!r} "
            f"device_id={self.device_id!r} sim_time={self.simulation_time}>"
        )
