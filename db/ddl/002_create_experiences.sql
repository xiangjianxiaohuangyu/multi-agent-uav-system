-- 002_create_experiences.sql
-- 经验库持久层（与 simulation_records 并存）
-- 由 docker-compose.mysql.yml 的 /docker-entrypoint-initdb.d 自动执行

CREATE TABLE IF NOT EXISTS experiences (
    experience_id  BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    scene_json     JSON        NOT NULL COMMENT '完整场景字典 (11 维 + 其它)',
    scene_vector   JSON        NOT NULL COMMENT '11 维特征向量 (固定顺序)',
    parameter_json JSON        NOT NULL COMMENT '6 字段路由参数',
    result_json    JSON        NOT NULL COMMENT '4 字段性能结果',
    score          FLOAT       NOT NULL DEFAULT 0.0
        COMMENT '0.5*PDR + 0.3*Delay_norm + 0.2*Energy_norm, ∈ [0,1]',
    created_time   TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_score_desc   (score),
    KEY idx_created_desc (created_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='FAISS 经验库的 MySQL 持久层';
