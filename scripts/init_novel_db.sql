-- 启用 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- 小说模块表（保留原有）
-- ============================================================

CREATE TABLE IF NOT EXISTS novels (
    novel_id VARCHAR(32) PRIMARY KEY,
    title VARCHAR(255),
    outline JSONB,
    current_volume INT,
    current_chapter INT,
    current_scene_index INT,
    current_state JSONB,
    last_sequence_id INT DEFAULT 0,
    revision INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chapters (
    chapter_id VARCHAR(32) PRIMARY KEY,
    novel_id VARCHAR(32),
    volume_num INT,
    chapter_num INT,
    file_path TEXT,
    word_count INT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS materials (
    material_id SERIAL PRIMARY KEY,
    novel_id VARCHAR(32),
    title VARCHAR(255),
    content TEXT,
    source_url TEXT,
    type VARCHAR(32),
    embedding vector(512),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chapter_summaries (
    chapter_id VARCHAR(32) PRIMARY KEY,
    novel_id VARCHAR(32),
    content TEXT,
    embedding vector(512),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS narrative_events (
    id BIGSERIAL PRIMARY KEY,
    event_uuid UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
    novel_id VARCHAR(32) NOT NULL,
    volume_num INT,
    chapter_num INT,
    scene_id INT,
    event_type VARCHAR(64) NOT NULL,
    event_data JSONB NOT NULL,
    event_version INT NOT NULL DEFAULT 1,
    timestamp TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_narrative_events_novel ON narrative_events(novel_id, id);
CREATE INDEX IF NOT EXISTS idx_narrative_events_chapter ON narrative_events(novel_id, volume_num, chapter_num);

CREATE TABLE IF NOT EXISTS world_snapshots (
    novel_id VARCHAR(32) NOT NULL,
    snapshot_id INT NOT NULL,
    volume_num INT,
    chapter_num INT,
    last_event_id BIGINT NOT NULL,
    world_state JSONB NOT NULL,
    compressed_state JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (novel_id, snapshot_id)
);

CREATE TABLE IF NOT EXISTS event_embeddings (
    event_id BIGINT PRIMARY KEY REFERENCES narrative_events(id) ON DELETE CASCADE,
    embedding vector(512) NOT NULL,
    summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_event_embeddings_vector ON event_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS narrative_causality (
    cause_event_id BIGINT NOT NULL,
    effect_event_id BIGINT NOT NULL,
    relation VARCHAR(32) NOT NULL,
    PRIMARY KEY (cause_event_id, effect_event_id)
);

CREATE TABLE IF NOT EXISTS compressed_states (
    novel_id VARCHAR(32) NOT NULL,
    volume_num INT NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (novel_id, volume_num)
);

CREATE TABLE IF NOT EXISTS lore_state (
    id VARCHAR(32) PRIMARY KEY DEFAULT 'global',
    world_rules JSONB DEFAULT '[]',
    realm_system JSONB DEFAULT '{}',
    major_characters JSONB DEFAULT '{}',
    geography JSONB DEFAULT '{}',
    cultivation_methods JSONB DEFAULT '{}',
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- 知识库表（与 retrieval.py 中的 asyncpg 版本兼容）
-- ============================================================

CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(255) PRIMARY KEY,
    title TEXT NOT NULL,
    source_path TEXT NOT NULL,
    file_type VARCHAR(20) NOT NULL,
    content TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255),
    chunk_index INTEGER,
    content TEXT NOT NULL,
    embedding vector(512),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);

-- 写作进度表（任务级进度，快速恢复使用）
CREATE TABLE IF NOT EXISTS writing_progress (
    project_id TEXT PRIMARY KEY,
    current_volume INTEGER NOT NULL DEFAULT 1,
    current_chapter INTEGER NOT NULL DEFAULT 1,
    current_scene INTEGER NOT NULL DEFAULT 0,
    chapter_completed BOOLEAN NOT NULL DEFAULT FALSE,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- 新增：场景执行单元表（执行计划持久化）
-- ============================================================
CREATE TABLE IF NOT EXISTS scene_execution_units (
    id BIGSERIAL PRIMARY KEY,
    novel_id VARCHAR(32) NOT NULL,
    volume_num INT NOT NULL,
    chapter_num INT NOT NULL,
    scene_index INT NOT NULL,          -- 0-based 场景索引

    status VARCHAR(32) NOT NULL DEFAULT 'pending',  -- pending, running, succeeded, failed, skipped

    plan_json JSONB NOT NULL,          -- 完整的场景计划（goal, conflict, must_events, characters, state_delta 等）
    planned_state_delta JSONB,         -- Planner 期望的状态变更（来自 plan_json.state_delta）
    actual_state_delta JSONB,          -- Writer 实际生成的状态变更（聚合自 events）
    applied_event_ids JSONB,           -- 最终应用到事件存储的事件 ID 列表（数组）

    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 2,
    error_message TEXT,

    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(novel_id, volume_num, chapter_num, scene_index)
);

CREATE INDEX IF NOT EXISTS idx_seu_novel_chapter ON scene_execution_units(novel_id, volume_num, chapter_num);
CREATE INDEX IF NOT EXISTS idx_seu_status ON scene_execution_units(status);
CREATE INDEX IF NOT EXISTS idx_seu_updated ON scene_execution_units(updated_at);

-- ============================================================
-- 任务调度表（可选，保留）
-- ============================================================
CREATE TABLE IF NOT EXISTS task_jobs (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    subtask_id TEXT NOT NULL,
    status TEXT NOT NULL,
    result TEXT,
    error TEXT,
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 2,
    dependencies TEXT,
    subtask_type TEXT,
    description TEXT,
    created_at DOUBLE PRECISION,
    updated_at DOUBLE PRECISION,
    started_at DOUBLE PRECISION,
    completed_at DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);