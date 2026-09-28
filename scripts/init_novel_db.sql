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
    PRIMARY KEY (novel_id, snapshot_id),
    snapshot_schema_version INT DEFAULT 1
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
-- 场景执行单元表（执行计划持久化）
-- ============================================================
CREATE TABLE IF NOT EXISTS scene_execution_units (
    id BIGSERIAL PRIMARY KEY,
    novel_id VARCHAR(32) NOT NULL,
    volume_num INT NOT NULL,
    chapter_num INT NOT NULL,
    scene_index INT NOT NULL,

    status VARCHAR(32) NOT NULL DEFAULT 'pending',

    plan_json JSONB NOT NULL,
    planned_state_delta JSONB,
    actual_state_delta JSONB,
    applied_event_ids JSONB,

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

-- ============================================================
-- 因果引擎新增表与字段 (v3.0)
-- ============================================================

-- 为 narrative_events 增加语义字段和 schema 版本
ALTER TABLE narrative_events ADD COLUMN IF NOT EXISTS semantic VARCHAR(32);
ALTER TABLE narrative_events ADD COLUMN IF NOT EXISTS event_schema_version INT DEFAULT 1;

-- 为 world_snapshots 增加快照 schema 版本
ALTER TABLE world_snapshots ADD COLUMN IF NOT EXISTS snapshot_schema_version INT DEFAULT 1;

-- 谓词投影缓存表
CREATE TABLE IF NOT EXISTS predicates (
    id BIGSERIAL PRIMARY KEY,
    novel_id VARCHAR(32) NOT NULL,
    event_id BIGINT NOT NULL,
    source_event_type VARCHAR(64),
    source_event_semantic VARCHAR(32),
    projection_version INT DEFAULT 1,
    subject VARCHAR(128),
    relation VARCHAR(64),
    object JSONB,
    negated BOOLEAN DEFAULT false,
    confidence FLOAT DEFAULT 1.0,
    priority VARCHAR(16) DEFAULT 'narrative',
    scope VARCHAR(32) DEFAULT 'scene',
    is_active BOOLEAN DEFAULT true,
    valid_from_event_id BIGINT,
    valid_to_event_id BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    identity_key VARCHAR(512),   -- 新增列：谓词唯一标识
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
-- 为 identity_key 创建索引（加速查询和失效操作）
CREATE INDEX IF NOT EXISTS idx_predicates_identity_key ON predicates(identity_key);
-- （可选）部分唯一索引：确保同一 identity_key 最多只有一条活跃记录
CREATE UNIQUE INDEX IF NOT EXISTS idx_predicates_active_identity 
ON predicates (novel_id, identity_key) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_predicates_novel_version ON predicates(novel_id, projection_version);
CREATE INDEX IF NOT EXISTS idx_predicates_novel_subject_relation ON predicates(novel_id, subject, relation);
CREATE INDEX IF NOT EXISTS idx_predicates_active_priority ON predicates(novel_id, is_active, priority);
CREATE INDEX IF NOT EXISTS idx_predicates_identity ON predicates(novel_id, subject, relation, (object->>0)) WHERE is_active = true;

-- 投影幂等记录表
CREATE TABLE IF NOT EXISTS projection_applied (
    delta_id VARCHAR(64) PRIMARY KEY,
    novel_id VARCHAR(32) NOT NULL,
    event_id BIGINT NOT NULL,
    applied_at TIMESTAMPTZ DEFAULT NOW()
);

-- 投影健康检查表
CREATE TABLE IF NOT EXISTS projection_health (
    novel_id VARCHAR(32) PRIMARY KEY,
    last_projected_event_id BIGINT,
    projection_lag_events INT,
    last_full_rebuild_at TIMESTAMPTZ,
    core_predicates_hash TEXT,
    drift_level VARCHAR(16),
    validator_mode VARCHAR(16),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 投影死信队列
CREATE TABLE IF NOT EXISTS projection_dead_letters (
    id BIGSERIAL PRIMARY KEY,
    novel_id VARCHAR(32) NOT NULL,
    event_id BIGINT NOT NULL,
    error TEXT,
    traceback TEXT,
    retry_count INT DEFAULT 0,
    status VARCHAR(16) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
-- 添加唯一约束（确保幂等性）
ALTER TABLE projection_dead_letters ADD CONSTRAINT IF NOT EXISTS projection_dead_letters_novel_event_unique UNIQUE (novel_id, event_id);

-- 章节一致性预算表
CREATE TABLE IF NOT EXISTS chapter_budget (
    novel_id VARCHAR(32) NOT NULL,
    volume_num INT NOT NULL,
    chapter_num INT NOT NULL,
    remaining_warnings INT DEFAULT 3,
    remaining_soft INT DEFAULT 1,
    PRIMARY KEY (novel_id, volume_num, chapter_num)
);

-- 可供性使用记录表（冷却）
CREATE TABLE IF NOT EXISTS affordance_usage (
    novel_id VARCHAR(32) NOT NULL,
    affordance_id VARCHAR(64) NOT NULL,
    last_used_chapter INT NOT NULL,
    PRIMARY KEY (novel_id, affordance_id)
);

CREATE TABLE IF NOT EXISTS projection_metrics (
    id BIGSERIAL PRIMARY KEY,
    novel_id VARCHAR(32) NOT NULL,
    event_id BIGINT NOT NULL,
    latency_seconds FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 状态审计表
CREATE TABLE IF NOT EXISTS state_audit (
    id BIGSERIAL PRIMARY KEY,
    novel_id VARCHAR(32) NOT NULL,
    node_name VARCHAR(64) NOT NULL,
    step_count INT NOT NULL,
    last_event_id BIGINT,                    -- 当前状态对应的最后一个事件ID
    state_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_state_audit_novel ON state_audit(novel_id);
CREATE INDEX IF NOT EXISTS idx_state_audit_node ON state_audit(node_name);
CREATE INDEX IF NOT EXISTS idx_state_audit_created ON state_audit(created_at);

-- 为 projection_health 添加 last_full_rebuild_at 列（如果不存在）
ALTER TABLE projection_health ADD COLUMN IF NOT EXISTS last_full_rebuild_at TIMESTAMPTZ;


-- 叙事版本表（Phase 6 新增）
CREATE TABLE IF NOT EXISTS narrative_versions (
    id BIGSERIAL PRIMARY KEY,
    novel_id VARCHAR(32) NOT NULL,
    volume_num INT NOT NULL,
    chapter_num INT NOT NULL,
    scene_idx INT NOT NULL,
    version_type VARCHAR(8) NOT NULL,  -- 'A', 'B', 'C'
    scene_text TEXT NOT NULL,
    world_state JSONB,                 -- 场景结束时的世界状态（可选）
    kpi_scores JSONB,                  -- KPI 预计算分数（可选）
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(novel_id, volume_num, chapter_num, scene_idx, version_type)
);

CREATE INDEX IF NOT EXISTS idx_narrative_versions_novel ON narrative_versions(novel_id);
CREATE INDEX IF NOT EXISTS idx_narrative_versions_chapter ON narrative_versions(novel_id, volume_num, chapter_num);
CREATE INDEX IF NOT EXISTS idx_narrative_versions_type ON narrative_versions(version_type);


CREATE TABLE IF NOT EXISTS narrative_projection_snapshots (
    id VARCHAR(32) PRIMARY KEY,
    novel_id VARCHAR(32) NOT NULL,
    chapter INT NOT NULL,
    event_id BIGINT NOT NULL,
    projection_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_projection_novel_chapter
    ON narrative_projection_snapshots(novel_id, chapter);

CREATE INDEX IF NOT EXISTS idx_projection_created
    ON narrative_projection_snapshots(created_at DESC);


CREATE TABLE IF NOT EXISTS loop_store (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    novel_id VARCHAR(64) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('active', 'resolved', 'abandoned')),
    progress FLOAT NOT NULL DEFAULT 0.0 CHECK (progress BETWEEN 0 AND 1),
    owner_character_id VARCHAR(64),
    priority INT DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at TIMESTAMPTZ
);
CREATE INDEX idx_loop_novel_status ON loop_store(novel_id, status);

-- Phase 15.3 — Shadow Rewrite Log
-- 用于记录 Shadow Rewrite 实验数据

CREATE TABLE IF NOT EXISTS shadow_rewrite_log (
    id BIGSERIAL PRIMARY KEY,

    -- 场景标识
    scene_id VARCHAR(64) NOT NULL,

    -- 原始文本（用于人工复核）
    original_text TEXT,
    rewritten_text TEXT,

    -- 长度统计
    original_length INT,
    rewritten_length INT,

    -- Validator 结果
    original_passed BOOLEAN,
    rewritten_passed BOOLEAN,
    original_violations JSONB,
    rewritten_violations JSONB,

    -- 执行状态
    status VARCHAR(32) NOT NULL,  -- success / validation_failed / llm_error / validator_error / timeout / skipped
    error_message TEXT,

    -- 实验元数据
    experiment_id VARCHAR(64),
    prompt_version VARCHAR(32) DEFAULT 'phase15.3.v1',
    contract_id VARCHAR(64),       -- 关联的 Contract ID
    model VARCHAR(64),             -- 使用的模型

    -- ========== A-1: 新增 contract_data ==========
    contract_data JSONB,           -- 完整的 PlanningContract，用于离线重放
    -- ============================================

    -- ========== Phase 15.5: 新增 writer_events ==========
    writer_events JSONB,           -- Writer 产生的结构化 events (List[Dict])，用于 Contract Compliance 诊断
    -- ====================================================

    -- 时间戳
    executed_at TIMESTAMP DEFAULT NOW()
);
-- 索引保持不变
CREATE INDEX IF NOT EXISTS idx_shadow_log_scene_id ON shadow_rewrite_log(scene_id);
CREATE INDEX IF NOT EXISTS idx_shadow_log_experiment_id ON shadow_rewrite_log(experiment_id);
CREATE INDEX IF NOT EXISTS idx_shadow_log_status ON shadow_rewrite_log(status);
CREATE INDEX IF NOT EXISTS idx_shadow_log_contract_id ON shadow_rewrite_log(contract_id);
CREATE INDEX IF NOT EXISTS idx_shadow_log_executed_at ON shadow_rewrite_log(executed_at DESC);



-- ============================================================
-- Phase 15.7-C3.4.2 — Validator V2 Audit Persistence
-- 记录每次 ValidationV2 的完整判定链：
--   Structural → Retrieval → LLM → Final
-- 设计原则：
--   1. 只追加，不更新
--   2. 每个 claim 一行
--   3. 完整保留 LLM 原始 verdict（审计"降级"关键）
--   4. Retriever / LLM 证据分层存储
--   5. 三态语义（None / [] / [N]）严格保留
-- ============================================================

CREATE TABLE IF NOT EXISTS validator_v2_audit (
    id BIGSERIAL PRIMARY KEY,

    -- ---------- 上下文 ----------
    novel_id        VARCHAR(64) NOT NULL,
    volume_num      INT,
    chapter_num     INT,
    scene_idx       INT,
    scene_id        VARCHAR(64),

    -- source = 调用入口: production | c3_3b_eval | b2_2_bridge
    source          VARCHAR(32) NOT NULL DEFAULT 'production',
    -- mode   = 语义模式: production | shadow_only
    mode            VARCHAR(16) NOT NULL DEFAULT 'production',

    -- ---------- Claim 标识 ----------
    claim_id                 VARCHAR(64) NOT NULL,
    state_change_type        VARCHAR(32) NOT NULL,
    raw_state_change_type    VARCHAR(32),

    -- ---------- 结构层（realm_change 专用） ----------
    -- 取值: exact_match | mismatch | no_event | NULL
    structural_check         VARCHAR(16),

    -- ---------- 证据层（Retriever） ----------
    evidence_candidates_found BOOLEAN NOT NULL DEFAULT FALSE,
    retrieved_evidence_count  INT     NOT NULL DEFAULT 0,
    retrieved_evidence_ids    JSONB,     -- 三态: NULL | [] | [...]

    -- ---------- LLM 层（原始输出，不降级） ----------
    llm_invoked              BOOLEAN NOT NULL DEFAULT FALSE,
    llm_raw_verdict          VARCHAR(16),    -- SUPPORTED | CONTRADICTED | INSUFFICIENT
    llm_raw_confidence       FLOAT,
    llm_raw_reason           TEXT,
    llm_evidence_ids         JSONB,

    -- ---------- 最终层（对外契约） ----------
    final_verdict            VARCHAR(16) NOT NULL,
    matched_layer            VARCHAR(16) NOT NULL,
    final_confidence         FLOAT       NOT NULL,
    fallback_applied         BOOLEAN     NOT NULL DEFAULT FALSE,
    final_reason             TEXT,

    -- ---------- 时间与版本 ----------
    executed_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    validator_version        VARCHAR(16) NOT NULL DEFAULT 'v2.0',
    contract_hash            VARCHAR(64)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_vva_novel_chapter
    ON validator_v2_audit(novel_id, volume_num, chapter_num, scene_idx);
CREATE INDEX IF NOT EXISTS idx_vva_claim
    ON validator_v2_audit(claim_id);
CREATE INDEX IF NOT EXISTS idx_vva_final_verdict
    ON validator_v2_audit(final_verdict);
CREATE INDEX IF NOT EXISTS idx_vva_fallback
    ON validator_v2_audit(fallback_applied) WHERE fallback_applied = TRUE;
CREATE INDEX IF NOT EXISTS idx_vva_executed
    ON validator_v2_audit(executed_at DESC);
-- 双轨观测索引：快速定位"LLM 判定与最终判定背离"的样本
CREATE INDEX IF NOT EXISTS idx_vva_llm_final_divergence
    ON validator_v2_audit(llm_raw_verdict, final_verdict)
    WHERE llm_invoked = TRUE;



-- ============================================================
-- C3.4.3B — Bridge Outcome Audit
--
-- 记录 B2-2 Production Bridge 的每次调用结果。
-- 与 validator_v2_audit 严格分离：
--   - validator_v2_audit：claim-level validation
--   - bridge_outcome_audit：bridge-level outcome
--
-- 职责边界：
--   trigger_status    = 有没有进入 B2-2 rescue 逻辑
--   execution_status  = 进入后有没有正常完成
--   rescued           = 正常完成后是否救援
--
-- 不覆盖：
--   False Rescue / Rescue Precision（需要独立 correctness oracle）
-- ============================================================

CREATE TABLE IF NOT EXISTS bridge_outcome_audit (
    id BIGSERIAL PRIMARY KEY,

    -- ---------- 上下文 ----------
    novel_id        VARCHAR(64) NOT NULL,
    volume_num      INT,
    chapter_num     INT,
    scene_idx       INT,
    scene_id        VARCHAR(64),

    -- ---------- 输入快照 ----------
    claim_types             JSONB NOT NULL,   -- 契约中全部 state_change 类型
    production_claim_types  JSONB NOT NULL,   -- 白名单过滤后的类型

    -- ---------- 触发判定 ----------
    -- 取值：
    --   original_passed      原 Validator 已通过，未进入 rescue
    --   no_state_changes     contract 无 state_changes
    --   no_production_types  白名单过滤后无 production 类型
    --   triggered            实际进入 try_rescue
    trigger_status          VARCHAR(32) NOT NULL,
    triggered               BOOLEAN     NOT NULL DEFAULT FALSE,

    -- ---------- 执行判定（仅 trigger_status='triggered' 时有意义） ----------
    -- 取值：
    --   completed  try 块正常返回 results
    --   error      try 块抛异常，返回 verdict="ERROR"
    execution_status        VARCHAR(16) NOT NULL DEFAULT 'completed',
    error_reason            TEXT,

    -- ---------- rescue 判定（仅 triggered + completed 时有值） ----------
    all_supported           BOOLEAN,
    min_confidence          FLOAT,
    confidence_threshold    FLOAT,
    confidence_ok           BOOLEAN,
    rescued                 BOOLEAN,

    -- ---------- 代表 claim（审计展示用） ----------
    representative_claim_id     VARCHAR(64),
    representative_claim_type   VARCHAR(32),
    representative_reason       TEXT,
    representative_evidence_ids JSONB,

    -- ---------- 版本与时间 ----------
    executed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    bridge_version          VARCHAR(16) NOT NULL DEFAULT 'b2_2c2.v1'
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_boa_novel_chapter
    ON bridge_outcome_audit(novel_id, volume_num, chapter_num, scene_idx);
CREATE INDEX IF NOT EXISTS idx_boa_trigger_status
    ON bridge_outcome_audit(trigger_status);
CREATE INDEX IF NOT EXISTS idx_boa_triggered
    ON bridge_outcome_audit(triggered) WHERE triggered = TRUE;
CREATE INDEX IF NOT EXISTS idx_boa_rescued
    ON bridge_outcome_audit(rescued) WHERE rescued = TRUE;
CREATE INDEX IF NOT EXISTS idx_boa_execution_status
    ON bridge_outcome_audit(execution_status) WHERE execution_status = 'error';
CREATE INDEX IF NOT EXISTS idx_boa_executed
    ON bridge_outcome_audit(executed_at DESC);
-- 核心 KPI 索引：Rescue Rate = rescued / triggered_completed
CREATE INDEX IF NOT EXISTS idx_boa_rescue_rate
    ON bridge_outcome_audit(triggered, execution_status, rescued);