# AI Factory — 叙事运行时引擎

AI Factory 是一个基于事件溯源和状态驱动架构的长篇修仙小说智能生成系统。  
它从简单的“AI 写小说脚本”演进为生产级叙事运行时引擎，具备游戏服务器级别的状态管理能力，支持确定性续写、分支叙事和工业级可靠性。

---

## 🚀 核心特性

- **事件溯源 + 写时复制**  
  所有剧情变更通过类型化事件表达，支持时间旅行、分支和确定性重放。  
  世界状态是唯一真相，禁止直接修改。

- **三层叙事记忆**  
  L1 活跃状态、L2 卷级压缩摘要、L3 永久世界知识，保证百万字长文一致性。

- **断点续写与进度恢复**  
  基于 `writing_progress` 和 `scene_execution_units` 表，支持任意卷/章/场景的精确续写，计划与实际执行状态分离。

- **工业级可靠性**  
  幂等写入、乐观锁、快照恢复、Prompt 防火墙、角色声纹、模型容器池和内存熔断。

- **多模型容器管理**  
  自动启停 llama.cpp 容器，内存感知调度，支持模型降级与预热。

- **LangGraph 工作流**  
  可视化编排 `plan → writer → validate → save` 节点，支持场景级重试和自动章节推进。

---

## 📐 架构（文字描述）

- **API 层**：FastAPI 提供 `/execute`、`/resume`、健康检查等端点，内置优先级调度器和全局并发控制。  
- **编排层**：LangGraph 状态机定义工作流节点（加载记忆 → 分析意图 → 规划场景 → 写作 → 验证 → 保存），支持条件路由和重试循环。  
- **执行层**：包含 PlannerAgent（大纲/场景计划）、WritingAgent（正文生成）、ValidatorAgent（结构/安全/语义验证）、ResearchAgent（RAG+网络搜索）、ExecutorAgent（代码生成与沙箱）、MemoryAgent（内存上下文）。  
- **持久层**：PostgreSQL + pgvector 存储事件流、快照、写作进度、场景执行单元、向量知识库等。  
- **基础设施层**：Docker Compose 管理多个 llama.cpp 容器（按任务类型区分），使用 ROCm 加速 AMD GPU。

---

## 🛠️ 技术栈

- **语言**：Python 3.11  
- **Web 框架**：FastAPI + Uvicorn  
- **工作流编排**：LangGraph  
- **数据库**：PostgreSQL 15 + pgvector + asyncpg  
- **模型推理**：llama.cpp (ROCm) + Docker 容器池  
- **嵌入服务**：本地 BGE-small-zh (512维)  
- **日志**：JSON 格式 + 滚动压缩  
- **部署**：Docker Compose

---

## 📦 快速开始

### 1. 环境要求

- Docker & Docker Compose
- AMD GPU（ROCm）或 CPU（需修改配置）
- Python 3.11（用于开发模式）

### 2. 启动服务

```bash

# 克隆仓库
git clone <repo-url>
cd ai_factory

# 启动数据库和模型容器
docker-compose up -d postgres
docker-compose up -d llamacpp-writing llamacpp-plan llamacpp-embedding

# 安装依赖（推荐使用 conda 环境）
conda create -n ai_factory python=3.11
conda activate ai_factory
pip install -r requirements.txt

# 初始化数据库
python scripts/init_novel_db.py

# 启动 API 服务
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

```

### 3. 生成小说

```bash

# 生成大纲（5卷10章）
curl -X POST "http://localhost:8000/api/v1/execute" \
  -H "Content-Type: application/json" \
  -d '{"user_input": "写一部修仙小说，共5卷，每卷10章，每章3个场景。主角林逸，从炼气期飞升。", "task_type": "novel_outline", "novel_id": "my_novel"}'

# 开始写作（自动完成所有章节）
curl -X POST "http://localhost:8000/api/v1/execute" \
  -H "Content-Type: application/json" \
  -d '{"user_input": "开始写作，自动完成所有章节", "task_type": "scene_plan", "novel_id": "my_novel"}'

```

## 项目结构（简略）

ai_factory/
├── src/
│   ├── agents/           # Planner, Writer, Validator 等 Agent
│   ├── api/              # FastAPI 路由和端点
│   ├── orchestrator/     # LangGraph 工作流定义和节点
│   ├── writing/          # 事件溯源、世界状态、场景执行单元
│   ├── db/               # 数据库连接池和进度管理
│   ├── execution/        # 模型容器池、沙箱、工具注册
│   └── prompts/          # Planner/Validator 提示词模板
├── scripts/              # 数据库初始化、日志分析等工具
├── docker-compose.yml
├── requirements.txt
└── README.md

## 📄 许可证
MIT