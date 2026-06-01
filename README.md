# Multi-Agent UAV System

Windows 端仿真数据处理服务，接收 ns3 推送的 UAV 网络仿真数据并进行全面分析。

## 项目结构

```
multi_agent_uav_system/
├── agent/                    # 智能体模块
│   ├── __init__.py
│   ├── agent.py              # Agent 类定义
│   ├── runner.py             # 主入口，process_with_agent 函数
│   ├── data_parser.py        # 数据解析（SimulationData, SceneParamsData）
│   └── test_local.py         # 本地测试工具
├── windows_server/           # FastAPI 服务器
│   └── main.py               # 回调接口
├── data/                     # 数据存储目录（按 task_id 分组）
│   └── {task_id}/
│       ├── ns3_data_*.json       # 仿真数据
│       └── scene_params_*.json   # 场景参数
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

## 安装

```bash
pip install -r requirements.txt
```

## 运行方式

### 方式一：启动服务器（接收 ns3 回调）

```bash
python -m uvicorn windows_server.main:app --host 0.0.0.0 --port 8000 --reload
```

服务器启动后：
- 自动保存 ns3 推送的原始数据到 `data/{task_id}/` 目录
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

