from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_api_url: str = "http://localhost:8081"
    llm_model_name: str = "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096
    llm_context_window: int = 32768

    # PostgreSQL + pgvector
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "openclaw"
    postgres_user: str = "openclaw"
    postgres_password: str = "openclaw123"
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20

    # Sandbox
    sandbox_timeout: int = 30
    sandbox_cpu_limit: float = 1.0
    sandbox_memory_limit: int = 512  # MB
    sandbox_network_enabled: bool = False

    # LLM Pool
    llm_max_concurrent: int = 4
    llm_timeout: int = 120

    # Tools
    tools_dir: str = "/tmp/ai_factory/tools"

    # RAG
    rag_k: int = 10
    rerank_threshold: float = 0.5
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_dim: int = 768

    # Embedding / Reranker
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_retries: int = 3

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/ai_factory.log"
    log_max_bytes: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


config = Settings()
settings = config
