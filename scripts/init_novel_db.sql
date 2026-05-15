-- ============================================================
-- AI Factory 小说模块完整数据库 Schema
-- 基于新架构（事件溯源、快照、向量检索等）
-- ============================================================

-- 启用 pgvector 扩展（如果尚未启用）
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- 保留的原有表（仍在使用，但可逐步迁移）
-- ============================================================

-- 小说主表（已包含 revision 乐观锁字段）
CREATE TABLE IF NOT EXISTS novels (
    novel_id VARCHAR(32) PRIMARY KEY,
    title VARCHAR(255),
    outline JSONB,
    current_volume INT,
    current_chapter INT,
    current_scene_index INT,
    current_state JSONB,
    scene_plan_list JSONB,
    last_sequence_id INT DEFAULT 0,
    revision INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 章节索引表（可选，用于外部查询）
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

-- 素材库（可选）
CREATE TABLE IF NOT EXISTS materials (
    material_id SERIAL PRIMARY KEY,
    novel_id VARCHAR(32),
    title VARCHAR(255),
    content TEXT,
    source_url TEXT,
    type VARCHAR(32),
    embedding vector(768),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 章节摘要向量表（用于语义检索）
CREATE TABLE IF NOT EXISTS chapter_summaries (
    chapter_id VARCHAR(32) PRIMARY KEY,
    novel_id VARCHAR(32),
    content TEXT,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- 新架构表（事件溯源、快照、向量检索等）
-- ============================================================

-- 叙事事件表（新版事件溯源）
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

-- 世界快照表
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

-- 事件向量表（用于语义检索）
CREATE TABLE IF NOT EXISTS event_embeddings (
    event_id BIGINT PRIMARY KEY REFERENCES narrative_events(id) ON DELETE CASCADE,
    embedding vector(768) NOT NULL,
    summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_event_embeddings_vector ON event_embeddings USING hnsw (embedding vector_cosine_ops);

-- 因果图表（未来使用）
CREATE TABLE IF NOT EXISTS narrative_causality (
    cause_event_id BIGINT NOT NULL,
    effect_event_id BIGINT NOT NULL,
    relation VARCHAR(32) NOT NULL,
    PRIMARY KEY (cause_event_id, effect_event_id)
);

-- 压缩状态表（L2 记忆）
CREATE TABLE IF NOT EXISTS compressed_states (
    novel_id VARCHAR(32) NOT NULL,
    volume_num INT NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (novel_id, volume_num)
);

-- 长期知识表（L3 记忆）
CREATE TABLE IF NOT EXISTS lore_state (
    id VARCHAR(32) PRIMARY KEY DEFAULT 'global',
    world_rules JSONB DEFAULT '[]',
    realm_system JSONB DEFAULT '{}',
    major_characters JSONB DEFAULT '{}',
    geography JSONB DEFAULT '{}',
    cultivation_methods JSONB DEFAULT '{}',
    updated_at TIMESTAMP DEFAULT NOW()
);