# Multi-Agent UAV System

Windows 端仿真数据处理服务，接收 ns3 推送的 UAV 网络仿真数据并进行全面分析。

## 项目结构

```
multi_agent_uav_system/
├── agent/                    # 智能体模块
│   ├── __init__.py
│   ├── agent.py              # Agent 类定义
│   ├── runner.py             # 主入口，process_with_agent 函数
│   ├── data_parser.py        # 数据解析（SimulationData, SceneParamsData, RawNodeSnapshot）
│   ├── llm_providers/        # LLM Provider 抽象层
│   │   ├── base.py
│   │   ├── ollama.py
│   │   ├── qwen.py
│   │   └── factory.py
│   ├── tools/                # Agent 工具
│   │   └── communication_tool.py
│   └── test_local.py         # 本地测试工具 + JSON 落盘
├── db/                       # MySQL 持久化层
│   ├── config.py             # 环境变量 → SQLAlchemy URL
│   ├── engine.py             # 延迟初始化 engine / Session
│   ├── models.py             # SimulationRecord + Experience ORM
│   ├── repository.py         # write_simulation_record（永不抛）
│   └── ddl/
│       ├── 001_create_simulation_records.sql
│       └── 002_create_experiences.sql
├── experience/               # FAISS 经验库
│   ├── __init__.py           # 公共 re-export
│   ├── config.py             # ExperienceConfig (YAML + env)
│   ├── models.py             # Experience ORM
│   ├── engine.py             # 延迟初始化
│   ├── faiss_manager.py      # FaissIndexManager (IndexIDMap2 + FlatL2)
│   ├── scoring.py            # scene_to_vector / compute_score / payload 映射
│   ├── repository.py         # 双写（MySQL + FAISS）+ 补偿
│   ├── schemas.py            # Pydantic 请求/响应
│   ├── router.py             # FastAPI 5 端点
│   ├── statistics.py         # 聚合 + CSV
│   └── default_config.yaml   # 默认配置
├── tests/                    # pytest 单元 / 集成 / API 测试
├── scripts/
│   └── verify_mysql.py       # 读检脚本
├── windows_server/           # FastAPI 服务器
│   └── main.py               # 回调接口（写 JSON + 写 MySQL + 写经验库 + 检索）
├── data/                     # JSON 数据 + FAISS 索引（gitignored）
│   ├── {task_id}/
│   │   ├── {simulation_time}/node_{node_id}.json
│   │   └── scene_params/scene_params.json
│   └── faiss/
│       ├── index.faiss       # FAISS 二进制索引
│       └── id_map.json       # 重建缓冲
├── docker-compose.mysql.yml  # 一键起 MySQL
├── requirements.txt
└── README.md
```

## API 接口

- `POST /api/simulation/start`：向 Ubuntu 端推送仿真启动请求
- `POST /api/simulation/callback`：接收 ns3 推送的仿真数据
- `GET /api/simulation/tasks/{task_id}`：查看任务状态

## 数据类型

系统支持两种 JSON 消息类型（通过 `type` 字段区分）：

### 1. simulation（仿真数据）
```json
{
  "type": "simulation",
  "task_id": "xxx",
  "status": "running",
  "simulation_time": 10.0,
  "nodes": [
    {
      "id": 0,
      "position": {"x": 100.0, "y": 200.0, "z": 0.0},
      "velocity": {"x": 1.5, "y": -0.5, "z": 0.0},
      "energy_percentage": 85.5,
      "hello_interval": 1.0,
      "weights": {
        "distance": 0.25,
        "linkTime": 0.25,
        "relVelocity": 0.25,
        "neighborCount": 0.25
      },
      "multipathCount": 1
    }
  ]
}
```

### 2. scene_params（场景参数）
```json
{
  "type": "scene_params",
  "task_id": "xxx",
  "node_count": 50,
  "communication_pairs": [
    {"source": 0, "destination": 5},
    {"source": 2, "destination": 7}
  ]
}
```

## 数据库（MySQL 持久化）

仿真回调数据**同时**落盘为 JSON（保留作原始备份/debug）和 MySQL（用于聚合查询与趋势分析）。MySQL 写入失败**不会**影响 JSON 落盘或 HTTP 200 响应。

### 1. 启动 MySQL

使用项目根目录的 `docker-compose.mysql.yml` 一键起容器，DDL 会在首次启动时自动执行：

```bash
docker compose -f docker-compose.mysql.yml up -d
docker exec -it uav-mysql mysql -uroot -prootpw uav_simulation -e "DESCRIBE simulation_records;"
```

### 2. 配置环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MYSQL_HOST` | `127.0.0.1` | MySQL 主机 |
| `MYSQL_PORT` | `3306` | MySQL 端口 |
| `MYSQL_USER` | `root` | 用户名 |
| `MYSQL_PASSWORD` | `` (空) | 密码 |
| `MYSQL_DATABASE` | `uav_simulation` | 数据库名 |
| `MYSQL_ECHO` | `0` | 设为 `1` 打开 SQLAlchemy SQL echo |
| `MYSQL_DISABLED` | `0` | 设为 `1` 跳过所有 MySQL 写入（仅 JSON） |

示例（PowerShell）：
```powershell
$env:MYSQL_HOST = "127.0.0.1"
$env:MYSQL_PORT = "3306"
$env:MYSQL_USER = "root"
$env:MYSQL_PASSWORD = "rootpw"
$env:MYSQL_DATABASE = "uav_simulation"
```

### 3. 验证

```bash
# 跑读检脚本
python -m scripts.verify_mysql

# 直接 SQL 查询
docker exec -it uav-mysql mysql -uroot -prootpw uav_simulation -e "
  SELECT id, task_id, device_id, simulation_time, m_speed, m_energy,
         nb_relative_speed_mean, res_avg_pdr
  FROM simulation_records ORDER BY id DESC LIMIT 5;"
```

### 4. 表设计要点

- **扁平化**：嵌套的 `m_info` / `neighbor_info` / `para_info` / `result_info` 拆为独立列，用前缀（`m_` / `nb_` / `param_` / `weight_` / `res_`）消歧
- **DECIMAL 保精度**：所有浮点列用 `DECIMAL` 而非 `FLOAT/DOUBLE`
- **相对速度容差**：`nb_relative_speed_*` 用 `DECIMAL(38, 4)`（MySQL DECIMAL 上限），实测可达 1e+19
- **复合索引**：
  - `idx_device_time(device_id, created_at)` — 设备趋势
  - `idx_param_combo(param_hello_interval, param_path_num)` — 参数调优
  - `idx_task_time(task_id, simulation_time)` — 任务切片

完整 DDL 见 [db/ddl/001_create_simulation_records.sql](db/ddl/001_create_simulation_records.sql)。



## 经验库（FAISS）

LLM Agent 在决策前会先在经验库中按"场景相似度"检索历史上得分最高的若干条参数配置，作为 in-context 参考。

### 数据结构

**场景特征（11 维定长向量，固定顺序）：**

| 序号 | 字段 | 来源 |
|---|---|---|
| 0 | `speed` | `m_info.speed` |
| 1 | `energy` | `m_info.energy` |
| 2 | `queue_length` | `m_info.queue_length` |
| 3 | `neighbor_count` | `m_info.neighbor_count` |
| 4 | `distance_to_destination` | `m_info.distance_to_destination` |
| 5 | `forward_candidate_ratio` | `neighbor_info.forward_candidate_ratio` |
| 6 | `avg_neighbor_distance` | `neighbor_info.distance_to_me_mean` |
| 7 | `relative_speed_mean` | `neighbor_info.relative_speed_mean` |
| 8 | `link_stability` | `clamp(nb_link_lifetime_mean / 60, 0, 1)` |
| 9 | `link_lifetime_mean` | `neighbor_info.link_lifetime_mean` |
| 10 | `traffic_load` | `clamp(nb_queue_length_mean / 10, 0, 1)` |

**参数（6 字段）：** `hello_interval`、`candidate_num`、`w_distance`、`w_linktime`、`w_energy`、`w_queue`

**结果（4 字段）：** `e2e_pdr`、`e2e_delay`、`routing_overhead`、`energy_consumption`

**Score 公式：**

```
pdr_n    = clamp(pdr, 0, 1)
delay_n  = 1 - clamp(delay_ms / MAX_DELAY_MS, 0, 1)
energy_n = 1 - clamp(energy   / MAX_ENERGY,   0, 1)
score    = 0.5 * pdr_n + 0.3 * delay_n + 0.2 * energy_n   ∈ [0, 1]
```

### MySQL 表

```sql
CREATE TABLE experiences (
    experience_id  BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    scene_json     JSON        NOT NULL,
    scene_vector   JSON        NOT NULL,
    parameter_json JSON        NOT NULL,
    result_json    JSON        NOT NULL,
    score          FLOAT       NOT NULL DEFAULT 0.0,
    created_time   TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_score_desc   (score),
    KEY idx_created_desc (created_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

DDL 自动由 `docker-compose.mysql.yml` 的 `initdb` 挂载执行（`db/ddl/002_create_experiences.sql`）。启动时还会调用 `Base.metadata.create_all` 兜底建表（`checkfirst=True` 幂等）。

### FAISS 索引

- 类型：`IndexIDMap2(IndexFlatL2)`，11 维
- 持久化：`data/faiss/index.faiss`（二进制）+ `data/faiss/id_map.json`（重建缓冲）
- 启动恢复：若 `.faiss` 缺失但 `id_map.json` 存在，自动重建索引
- 线程安全：所有写操作在 `threading.Lock` 内串行

### REST API

| 端点 | 方法 | 描述 |
|---|---|---|
| `/experience/add` | POST | 新增一条经验（自动算 score + 双写） |
| `/experience/{id}` | GET | 获取单条经验完整记录 |
| `/experience/{id}` | DELETE | 删除（FAISS + MySQL 同步） |
| `/experience/search` | POST | 按 scene 检索 Top-K 经验（按 score DESC） |
| `/experience/statistics` | GET | 聚合统计；`?export=true` 返 CSV |

`EXPERIENCE_DISABLED=1` 时所有端点返回 503。

### 写入流程

1. `ExperienceRepository.add(scene, parameter, result)`
2. 计算 `score = 0.5*PDR + 0.3*Delay_norm + 0.2*Energy_norm`
3. 构造 11 维 `scene_vector`
4. **MySQL 先提交**（拿 `experience_id`）
5. FAISS 写入
6. **FAISS 失败 → MySQL 补偿删除**（双写一致性）

### 读取流程

1. 当前 scene → 11 维 vector
2. FAISS L2 距离 Top-K（k 由 config 控制，上限 `topk_max`）
3. MySQL `SELECT WHERE id IN (...)` 取完整记录
4. 按 `score DESC` 排序返回

### 配置

`experience/default_config.yaml`：

```yaml
dimension: 11
faiss_index_path: "./data/faiss/index.faiss"
faiss_id_map_path: "./data/faiss/id_map.json"
max_delay_ms: 1000.0
max_energy: 100.0
score_weights: { pdr: 0.5, delay: 0.3, energy: 0.2 }
topk_default: 5
topk_max: 50
auto_create_table: true
log_level: "INFO"
```

环境变量覆盖：

| 变量 | 默认 | 说明 |
|---|---|---|
| `EXPERIENCE_DISABLED` | `0` | `1` 跳过所有写入/检索（端点返 503） |
| `EXPERIENCE_FAISS_DIR` | `./data/faiss` | 索引和 id_map 输出目录 |
| `EXPERIENCE_MAX_DELAY_MS` | `1000.0` | 归一化分母 |
| `EXPERIENCE_MAX_ENERGY` | `100.0` | 归一化分母 |
| `EXPERIENCE_TOPK_DEFAULT` | `5` | search 默认 k |
| `EXPERIENCE_TOPK_MAX` | `50` | search k 上限 |
| `EXPERIENCE_LOG_LEVEL` | `INFO` | logger 级别 |

### 与 LLM Agent 的集成

`windows_server/main.py` 在 `/api/simulation/callback` 中：

1. 保存 JSON 原始数据
2. 写入 `simulation_records`
3. 写入经验库（FAISS + `experiences`）
4. 检索 Top-5 历史经验
5. 把 `prior_experiences` 透传给 `process_with_agent(...)`
6. `LlmAgent.think_with_loop` 把 `prior_experiences` 注入到 LLM system prompt

Agent 在做参数决策时能直接看到"类似场景下历史 score 最高的参数是什么"，无需修改任何工具。

### 测试

```bash
# 单元测试（无 IO）
pytest tests/test_experience_unit.py -v

# 集成测试（SQLite + 临时 FAISS 目录）
pytest tests/test_experience_integration.py -v

# API 测试（FastAPI TestClient）
pytest tests/test_experience_api.py -v

# 全部
pytest tests/ -v
```

### 端到端验证

```bash
# 1) 启 MySQL（新 DDL 自动执行）
docker compose -f docker-compose.mysql.yml up -d
docker exec -it uav-mysql mysql -uroot -prootpw uav_simulation \
  -e "SHOW TABLES; DESCRIBE experiences;"

# 2) 启动服务
uvicorn windows_server.main:app --host 0.0.0.0 --port 8000 --reload
# 启动日志应见：[experience] faiss index ready / experiences table ensured

# 3) 添加经验
curl -X POST http://localhost:8000/experience/add \
  -H "Content-Type: application/json" \
  -d '{
    "scene": {"speed":5.0,"energy":80.0,"queue_length":2,"neighbor_count":4,
              "distance_to_destination":120.0,"forward_candidate_ratio":0.6,
              "avg_neighbor_distance":50.0,"relative_speed_mean":1.2,
              "link_stability":0.9,"link_lifetime_mean":30.0,"traffic_load":0.3},
    "parameter": {"hello_interval":1.0,"candidate_num":2,
                  "w_distance":0.4,"w_linktime":0.3,"w_energy":0.2,"w_queue":0.1},
    "result": {"e2e_pdr":0.92,"e2e_delay":150.0,
               "routing_overhead":12.0,"energy_consumption":8.5}
  }'

# 4) 检索
curl -X POST http://localhost:8000/experience/search \
  -H "Content-Type: application/json" \
  -d '{"scene": {...}, "top_k": 3}'

# 5) 统计 / CSV 导出
curl http://localhost:8000/experience
curl "http://localhost:8000/experience?export=true" -o stats.csv
```



## 安装

```bash
pip install -r requirements.txt
```

依赖：`fastapi`、`uvicorn`、`pydantic`、`httpx`、`SQLAlchemy>=2.0`、`PyMySQL>=1.1`、`cryptography`。

## 运行方式

### 方式一：启动服务器（接收 ns3 回调）

```bash
python -m uvicorn windows_server.main:app --host 0.0.0.0 --port 8000 --reload
```

服务器启动后：
- 自动保存 ns3 推送的原始数据到 `data/{task_id}/{simulation_time}/node_{node_id}.json`
- 自动调用智能体处理数据

### 方式二：本地测试（使用保存的数据）

```bash
python -m agent.test_local
```

本地测试模式：
- 按时间顺序读取 `data/` 目录下所有保存的数据文件
- 不需要启动服务器
- 用于离线调试和算法验证

## 配置

默认 Ubuntu 端地址：

```text
UBUNTU_SERVER_URL=http://localhost:8001
```

默认 Windows 回调地址：

```text
WINDOWS_CALLBACK_URL=http://localhost:8000/api/simulation/callback
```

两端不在同一台机器时，把 `localhost` 改成 Ubuntu 能访问到的 Windows IP。

## 向 Ubuntu 启动仿真

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/simulation/start `
  -ContentType "application/json" `
  -Body '{"config":{"algorithm":"aodv","size":8,"duration":5,"seed":1},"ubuntu_server_url":"http://127.0.0.1:8001"}'
```

Windows 端会自动把 `task_id` 和 `callback_url` 写入提交给 Ubuntu 的 JSON。

