# AI Factory

AI Factory 是一个基于 LangGraph 和 LlamaIndex 的智能代理系统，支持 RAG 知识库检索、代码生成与沙箱执行。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                      API Layer                           │
│              FastAPI (REST /api/v1)                      │
├─────────────────────────────────────────────────────────┤
│                   Orchestrator                           │
│         LangGraph StateMachine (analyze→research→       │
│              code→validate→retry loop)                   │
├──────────────┬──────────────────────┬───────────────────┤
│   Agent      │     Knowledge        │    Execution      │
│  Research    │     RAG Pipeline     │   Docker Sandbox  │
│  Coder       │  pgvector + HNSW     │   File Ops        │
├──────────────┴──────────────────────┴───────────────────┤
│               Infrastructure                             │
│     PostgreSQL+pgvector │ LLM (llama.cpp) │ Docker      │
└─────────────────────────────────────────────────────────┘
```

## 功能特性

- **Orchestrator 层**: 使用 LangGraph 实现状态机工作流（analyze → research/code → validate → 重试/结束）
- **Agent 层**: ResearchAgent（知识检索 + LLM 总结）+ CoderAgent（代码生成 + 沙箱执行）
- **Knowledge 层**: RAG 知识库，支持 .txt/.md/.py 文档解析，pgvector HNSW 向量检索 + bge-reranker 重排序
- **Execution 层**: Docker 沙箱执行代码（带资源限制），安全文件操作
- **API 层**: FastAPI REST 接口，支持同步和流式执行

## 技术栈

- Python 3.11
- LangGraph (工作流编排)
- LlamaIndex (RAG 知识库)
- FastAPI (API 服务)
- PostgreSQL + pgvector (向量存储)
- llama.cpp (本地 LLM)

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 配置数据库和 LLM 参数
```

### 3. 启动基础设施

```bash
# 使用 Docker Compose 启动 PostgreSQL
docker-compose up -d postgres

# 初始化数据库表结构
bash scripts/init_db.sh
```

### 4. 启动 LLM 后端

```bash
# llama.cpp 需要单独启动
# 确保 Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf 模型可用
# 默认地址: http://localhost:8081/v1
```

### 5. 启动 API 服务

```bash
# 使用 Makefile
make run

# 或直接运行
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. 访问 API

```bash
# 健康检查
curl http://localhost:8000/health

# 执行请求
curl -X POST http://localhost:8000/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Write a Python function to calculate Fibonacci numbers"}'
```

## API 端点

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/ready` | 就绪探针 |
| POST | `/api/v1/execute` | 执行用户请求 |
| POST | `/api/v1/execute/stream` | 流式执行 |

## 项目结构

```
ai_factory/
├── src/
│   ├── api/                  # API 层
│   │   ├── main.py           # FastAPI 应用
│   │   ├── routes/           # 路由定义
│   │   └── endpoints/        # 端点实现
│   ├── orchestrator/         # Orchestrator 层
│   │   ├── state.py          # AgentState 定义
│   │   ├── graph.py          # LangGraph 工作流图
│   │   └── nodes.py          # 节点函数
│   ├── agents/               # Agent 层
│   │   ├── research.py       # ResearchAgent
│   │   └── coder.py          # CoderAgent
│   ├── knowledge/            # Knowledge 层
│   │   ├── ingestion.py      # 文档解析
│   │   ├── retrieval.py      # pgvector 向量检索
│   │   └── reranker.py       # 重排序
│   ├── execution/            # Execution 层
│   │   ├── sandbox.py        # Docker 沙箱
│   │   └── file_ops.py       # 文件操作
│   ├── config/               # 配置
│   │   └── settings.py       # Pydantic 配置
│   ├── common/               # 公共模块
│   │   ├── models.py         # 数据模型
│   │   ├── logging.py        # 日志
│   │   └── retry.py          # 重试机制
│   └── memory/               # 记忆管理
├── tests/                    # 测试
├── scripts/                  # 脚本
│   ├── init_db.sh            # 数据库初始化
│   └── migrate_db.sh         # 数据库迁移
├── docker-compose.yml        # Docker Compose
├── Dockerfile                # 应用镜像
├── requirements.txt          # Python 依赖
├── .env.example              # 环境变量模板
├── makefile                  # Makefile
└── README.md                 # 本文件
```

## 开发

```bash
# 运行测试
make test

# 代码格式化
make format

# 代码检查
make lint

# 完整 Docker 构建
make docker-full

# 清理
make clean
```

## LLM 配置

默认使用 llama.cpp 作为 LLM 后端：

- **地址**: `http://localhost:8081/v1`
- **模型**: `Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf`
- **上下文窗口**: 32768 tokens
- **最大输出**: 4096 tokens

## 数据库

PostgreSQL + pgvector 配置：

- **Host**: localhost:5432
- **用户**: openclaw
- **密码**: openclaw123
- **数据库**: openclaw
- **向量维度**: 768 (bge-small-en)
- **索引**: HNSW (m=16, ef_construction=64)

## License

MIT
