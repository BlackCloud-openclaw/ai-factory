# AI Factory

AI Factory 是一个基于 LangGraph 的事件溯源叙事运行时（Event‑Sourced Narrative Runtime），支持超长篇小说生成、代码生成与沙箱执行，以及 RAG 知识库检索。

## 核心设计理念

- **状态驱动叙事**：小说的唯一真相是结构化世界状态（WorldState），文本仅是其渲染结果。
- **事件溯源**：所有剧情变更以类型化事件（Typed Events）记录，支持回放、分叉、时间旅行。
- **确定性状态转移**：Planner 生成状态增量（StateDelta），Writer 仅负责渲染，Validator 保证一致性。
- **多层记忆**：L1 活跃状态、L2 压缩摘要、L3 永久知识，有效管理长期上下文。
- **工业级可靠性**：事件版本化、幂等写入、乐观锁、快照恢复、Prompt 防火墙、角色声纹等。

## 架构概览
```
┌─────────────────────────────────────────────────────────────────┐
│ API Layer (FastAPI) │
├────────────────────────────────────────────────────────```bash─────────┤
│ Orchestrator (LangGraph State Machine) │
│ load_memory → analyze → plan → writer → validate → save │
├───────────────┬─────────────────────────┬──────────────────────┤
│ Agent │ Writing │ Execution │
│ Planner │ WorldState │ Docker Sandbox │
│ Writer │ StateDelta │ File Operations │
│ Validator │ NarrativeEventStore │ ToolsRegistry │
│ Research │ SnapshotManager │ │
├───────────────┼─────────────────────────┼──────────────────────┤
│ Infrastructure │
│ PostgreSQL+pgvector │ llama.cpp │ Docker │
└─────────────────────────────────────────────────────────────────┘
```

```bash
## 功能特性

### 叙事引擎（新增）
- **状态驱动小说生成**：支持 50 万字以上长篇小说，剧情可重放、可编辑。
- **事件溯源**：所有剧情变更记录为类型化事件（境界突破、物品获得、关系变化等）。
- **自动快照与恢复**：每章/每卷自动保存世界快照，服务重启后可精确恢复进度。
- **分层验证**：规则校验 + 语义校验（embedding）确保正文符合计划。
- **角色声纹**：每个角色拥有独立的语言风格，防止角色同质化。
- **结构化输出防火墙**：LLM 输出 JSON 对象，防止提示词注入和格式错误。

### 通用能力（保留）```bash
- **RAG 知识库**：支持 .txt/.md/.py 文档向量化，使用 pgvector + HNSW + bge‑reranker。
- **代码生成与沙箱**：生成 Python 代码并在 Docker 容器中安全执行。
- **智能调度**：多任务优先级队列，指数退避重试，超时控制。

## 技术栈

- **Python 3.11** + **LangGraph**（工作流编排）
- **FastAPI**（REST API）
- **PostgreSQL + pgvector**（事件存储、向量检索、快照）
- **llama.cpp**（本地 LLM 推理，支持多容器模型池）
- **Docker**（代码执行沙箱）
- **Pydantic**（状态模型与配置）
```bash
## 快速开始

### 1. 环境准备

```bash
# 克隆仓库并进入目录
cd ai_factory

# 创建虚拟环境
python -m venv .conda
source .conda/bin/activate  # Linux/Mac```bash
# 或 .conda\Scripts\activate (Windows)

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env，填写数据库连接、LLM 地址、模型名称等```bash
```

### 3. 启动数据库
```bash
# 使用 Docker Compose 启动 PostgreSQL
docker-compose up -d postgres

# 初始化数据库表结构（包括事件表、快照表等）
python scripts/init_novel_db.py
```

### 4. 启动 LLM 容器
系统使用 llama.cpp 容器池，需要预先拉取镜像并启动至少一个模型容器，例如：
```bash
docker run -d --gpus all --name llamacpp-writing -p 8082:8081 \
  -v /models:/models ghcr.io/ggerganov/llama.cpp:server \
  -m /models/qwen3-32b-q5_k_m.gguf --host 0.0.0.0 --port 8081 -ngl 99
```
其他模型容器（plan、validate、code）按需启动。

### 5. 启动 API 服务
```bash
# 使用 Makefile
make run

# 或直接运行```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. 生成第一部长篇小说
```bash
curl -X POST http://localhost:8000/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "写一部修仙小说，共5卷，每卷10章，每章3个场景。主角林逸，从炼气期飞升。",
    "task_type": "novel_outline",
    "novel_id": "my_first_novel"
  }'

生成大纲后，再启动完整写作：

curl -X POST http://localhost:8000/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "开始写作",
    "task_type": "scene_plan",
    "novel_id": "my_first_novel",
    "resume": false
  }'

章节文件将保存在 data/novels/my_first_novel/vol_001/ 目录下。
```

## API 端点
方法	路径	描述
GET	/health	健康检查
GET	/ready	就绪探针
POST	/api/v1/execute	执行用户请求（支持 task_type：code, novel_outline, scene_plan）
POST	/api/v1/novel/resume	断点续写小说（需提供 novel_id 和可选的 from_event_id）
GET	/api/v1/novel/{novel_id}/events	获取指定小说的事件溯源记录
PATCH	/api/v1/novel/events/{event_uuid}	编辑某个事件并截断其后所有事件

## A项目结构
ai_factory/
├── src/
│   ├── api/                  # FastAPI 层
│   ├── agents/               # Planner, Writer, Validator, Research, Executor
│   ├── orchestrator/         # LangGraph 状态机 + 节点函数
│   ├── writing/              # 新架构核心
│   │   ├── world_state.py    # 规范化的世界状态模型
│   │   ├── events.py         # 类型化叙事事件
│   │   ├── delta.py          # 状态增量定义与应用
│   │   ├── event_store.py    # 叙事事件存储（幂等、版本化）
│   │   ├── snapshot.py       # 世界快照管理
│   │   ├── context_compiler.py # 智能上下文裁剪
│   │   ├── voiceprint.py     # 角色声纹系统
│   │   ├── prompt_firewall.py # 结构化输出防火墙
│   │   └── validators/       # 分层验证器（结构、安全、规则、语义）
│   ├── knowledge/            # RAG 检索与重排序
│   ├── execution/            # 代码沙箱、文件操作、工具注册表
│   ├── config/               # Pydantic 配置
│   └── common/               # 日志、重试、模型定义
├── config/                   # 外部配置（voiceprints.yaml 等）
├── data/novels/              # 生成的小说章节文件
├── scripts/                  # 数据库初始化、测试脚本
├── tests/                    # 单元测试与长期运行测试
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── Makefile
└── README.md

## 开发命令
命令	作用
make install	安装依赖
make run	启动开发服务器
make test	运行所有测试（含覆盖率）
make format	使用 black 格式化代码
make lint	检查代码风格
make clean	清理缓存与临时文件
make migrate-db	执行数据库表迁移（建表）
make status	查看数据库与容器状态

## 配置要点
.env 文件中的 LLM_API_URL 应指向 llama.cpp 容器池，例如 http://localhost:8081。

小说生成专用模型需要在 src/execution/llm_router_pool.py 中正确映射端口，并在 MODEL_CANDIDATES 中配置任务类型。

embedding_api_url 需提供支持 OpenAI 兼容接口的嵌入服务（如本地 sentence-transformers 或外部 API）。

角色声纹配置文件位于 config/voiceprints.yaml，可按需添加或修改。

## 许可证
此 README 完整描述了当前系统（包含事件溯源、叙事运行时、小说生成等核心功能），同时保留了原有代码生成和 RAG 的介绍。用户可直接替换项目根目录下的 `README.md`。