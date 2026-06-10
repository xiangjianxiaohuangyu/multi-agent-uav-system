-- db/ddl/001_create_simulation_records.sql
-- 仿真/实验参数与结果固定维度表。
-- 配套 ORM 模型：db.models.SimulationRecord
-- 字段映射规则：见 db/repository.py

CREATE TABLE IF NOT EXISTS simulation_records (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,

    -- 标识 + 时间
    task_id        VARCHAR(64)  NOT NULL COMMENT 'ns-3 任务 ID',
    device_id      VARCHAR(64)  NOT NULL COMMENT '设备/节点ID (来自 node_identity，回退 node_id)',
    simulation_time DECIMAL(12,3) NOT NULL COMMENT '仿真时间（ns-3 sim seconds，非 wall-clock）',
    created_at     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '落库时间（wall-clock）',

    -- m_info 自身状态
    m_speed DECIMAL(8,2) NOT NULL,
    m_energy DECIMAL(8,2) NOT NULL,
    m_queue_length INT NOT NULL,
    m_neighbor_count INT NOT NULL,
    m_distance_to_destination DECIMAL(10,2) NOT NULL,

    -- neighbor_info 邻居统计
    nb_forward_candidate_ratio DECIMAL(4,3) NOT NULL,
    nb_distance_to_me_mean DECIMAL(8,2) NOT NULL,
    nb_distance_to_me_std DECIMAL(8,2) NOT NULL,
    nb_distance_to_destination_mean DECIMAL(10,2) NOT NULL,
    nb_distance_to_destination_std DECIMAL(10,2) NOT NULL,
    nb_distance_to_destination_min DECIMAL(10,2) NOT NULL,
    -- 修正：实测相对速度可达 1e+19 量级，DECIMAL(38,4) 是 MySQL 上限
    nb_relative_speed_mean DECIMAL(38,4) NOT NULL,
    nb_relative_speed_std  DECIMAL(38,4) NOT NULL,
    nb_link_lifetime_mean DECIMAL(8,2) NOT NULL,
    nb_link_lifetime_std DECIMAL(8,2) NOT NULL,
    nb_neighbor_degree_mean DECIMAL(6,2) NOT NULL,
    nb_neighbor_degree_std DECIMAL(6,2) NOT NULL,
    nb_queue_length_mean DECIMAL(8,2) NOT NULL,
    nb_queue_length_std DECIMAL(8,2) NOT NULL,
    nb_queue_length_max INT NOT NULL,
    nb_energy_mean DECIMAL(8,2) NOT NULL,
    nb_energy_std DECIMAL(8,2) NOT NULL,
    nb_energy_min DECIMAL(8,2) NOT NULL,

    -- para_info 配置参数
    param_hello_interval DECIMAL(5,2) NOT NULL,
    param_path_num INT NOT NULL,
    weight_distance DECIMAL(4,3) NOT NULL,
    weight_link_time DECIMAL(4,3) NOT NULL,
    weight_rel_velocity DECIMAL(4,3) NOT NULL,
    weight_neighbor_count DECIMAL(4,3) NOT NULL,

    -- result_info 实验结果
    res_avg_pdr DECIMAL(5,4) NOT NULL,
    res_avg_delay DECIMAL(8,2) NOT NULL,
    res_energy_consumption DECIMAL(8,2) NOT NULL,
    res_control_packets INT NOT NULL,
    res_distance_progress DECIMAL(10,2) NOT NULL,

    -- 索引
    KEY idx_device_time (device_id, created_at) COMMENT '设备 + 落库时间 趋势分析',
    KEY idx_param_combo (param_hello_interval, param_path_num) COMMENT '参数调优查询',
    KEY idx_task_time (task_id, simulation_time) COMMENT '任务 + 仿真时间 切片'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='仿真场景参数与结果固定维度表';
