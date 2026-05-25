from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_api_url: str = "http://localhost:8081"
    llm_model_name: str = "Qwen3.6-27B-Q5_K_M"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096
    llm_context_window: int = 32768

    # Embedding
    embedding_api_url: str = "http://localhost:8087"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 384
    embedding_mode: str = "api"  # api or local

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str          # 无默认值，必须从环境变量读取
    postgres_user: str        # 无默认值
    postgres_password: str    # 无默认值
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20
    postgres_max_queries: int = 50000
    postgres_max_inactive_connection_lifetime: float = 300.0

    # Sandbox
    sandbox_timeout: int = 120
    sandbox_cpu_limit: float = 1.0
    sandbox_memory_limit: int = 512
    sandbox_network_enabled: bool = False

    # LLM Pool
    llm_max_concurrent: int = 4
    llm_timeout: int = 600
    llm_timeout_planning: int = 1800
    llm_timeout_coding: int = 900
    llm_timeout_validation: int = 600
    llm_timeout_research: int = 600
    llm_timeout_writing: int = 900  # 15分钟

    # LLM 容器与内存管理
    memory_safety_margin_gb: int = 2
    idle_timeout: int = 86400
    force_evict_idle: bool = True

    # 调度器
    scheduler_max_retries: int = 3
    scheduler_task_timeout: int = 1800
    scheduler_worker_count: int = 3

    # API 与并发
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    global_workflow_semaphore: int = 2
    request_max_size: int = 1048576

    # Tools
    tools_dir: str = "/tmp/ai_factory/tools"

    # RAG
    rag_k: int = 10
    rerank_threshold: float = 0.5
    chunk_size: int = 512
    chunk_overlap: int = 50
    kb_quick_fail: bool = True

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-base"

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/ai_factory.log"
    log_max_bytes: int = 10 * 1024 * 1024
    log_backup_count: int = 5
    log_compress: bool = True

    # 小说生成
    max_active_characters: int = 20
    max_timeline_events: int = 100
    snapshot_interval_events: int = 1000
    auto_snapshot_on_chapter: bool = True
    must_events_similarity_threshold: float = 0.3
    langgraph_recursion_limit: int = 5000   #LangGraph递归限制
    
    # 可观测性
    enable_projection_metrics: bool = True
    dead_letter_alert_threshold: int = 3

    # 任务模型映射（可被环境变量覆盖为 JSON 字符串，未实现动态加载）
    task_model_map: Dict[str, List[str]] = {
        "code": ["Qwen2.5-Coder-32B-Instruct-Q5_K_M"],
        "writing": ["Qwen3-32B-Q5_K_M-writer"],
        "research": ["DeepSeek-R1-Distill-Llama-70B-Q5_K_M"],
        "validate": ["DeepSeek-R1-Distill-Qwen-32B-Q5_K_M"],
        "plan": ["Qwen3-32B-Q5_K_M"],
        "default": ["Qwen3.6-27B-Q5_K_M"],
    }

    @property
    def postgres_dsn(self) -> str:
        if not all([self.postgres_user, self.postgres_password, self.postgres_db]):
            raise ValueError("Database credentials missing. Set POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB in .env")
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def embedding_endpoint(self) -> str:
        return f"{self.embedding_api_url}/v1/embeddings"


config = Settings()
settings = config