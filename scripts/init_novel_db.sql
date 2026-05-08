-- scripts/init_novel_db.sql

-- 事件表（所有状态变更）
CREATE TABLE IF NOT EXISTS events (
    sequence_id BIGSERIAL PRIMARY KEY,
    event_id UUID NOT NULL,
    type VARCHAR(64) NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    novel_id VARCHAR(32) NOT NULL,
    chapter_id VARCHAR(32)
);

CREATE INDEX IF NOT EXISTS idx_events_novel_sequence ON events(novel_id, sequence_id);

-- 小说主表（元数据 + 快照）
CREATE TABLE IF NOT EXISTS novels (
    novel_id VARCHAR(32) PRIMARY KEY,
    title VARCHAR(255),
    outline JSONB,
    current_volume INT,
    current_chapter INT,
    current_scene INT,
    current_state JSONB,
    last_sequence_id INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 章节索引（新增 updated_at 列）
CREATE TABLE IF NOT EXISTS chapters (
    chapter_id VARCHAR(32) PRIMARY KEY,
    novel_id VARCHAR(32),
    volume_num INT,
    chapter_num INT,
    file_path TEXT,
    word_count INT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()   -- 新增列
);

-- 素材库（下载的小说、参考材料）
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

-- 章节摘要向量（用于语义检索）
CREATE TABLE IF NOT EXISTS chapter_summaries (
    chapter_id VARCHAR(32) PRIMARY KEY,
    novel_id VARCHAR(32),
    content TEXT,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT NOW()
);