-- 002_create_experiences.sql
-- 经验库持久层（与 simulation_records 并存）
-- 由 docker-compose.mysql.yml 的 /docker-entrypoint-initdb.d 自动执行

CREATE TABLE IF NOT EXISTS experiences (
    experience_id  BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    scene_json     JSON        NOT NULL COMMENT '完整场景字典 (23 维)',
    scene_vector   JSON        NOT NULL COMMENT '23 维特征向量 (固定顺序)',
    parameter_json JSON        NOT NULL COMMENT '6 字段路由参数 (hello_interval/path_num/w_*)',
    result_json    JSON        NOT NULL COMMENT '2 字段性能结果 (avg_pdr/avg_delay)',
    created_time   TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_created_desc (created_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='FAISS 经验库的 MySQL 持久层';
