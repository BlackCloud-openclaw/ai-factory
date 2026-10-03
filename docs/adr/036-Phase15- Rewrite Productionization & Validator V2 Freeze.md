# ADR-036: Phase 15 — Rewrite Productionization & Validator V2 Freeze

| 项目 | 内容 |
|------|------|
| **状态** | ✅ 已冻结（Frozen） |
| **日期** | 2026-10-03 |
| **决策者** | Phase 15 团队 |
| **前置依赖** | Phase 14 (Narrative Contract Hardening) — ✅ 已完成 |
| **影响范围** | `ControlledWriter`, `StructuralLock`, `SemanticValidator`, `ProductionBridge`, `validator_v2_audit`, `bridge_outcome_audit`, `shadow_rewrite_log`, `writer_artifact`, `AgentState` |

---

## 一、背景与问题陈述

### 1.1 Phase 14 后的状态

Phase 14 完成了 Narrative Contract Hardening：

| 子阶段 | 内容 | 状态 |
|--------|------|------|
| 14.0A | Contract Signal Foundation | ✅ 冻结 |
| 14.0B | Validation Control（三态 ValidatorOutput） | ✅ 冻结 |
| 14.0C | Writer Activation（部分延后） | ⚠️ |

Phase 14 已验证：

- `observables.state_changes` 覆盖率 0% → 95%+
- `force_pass` 比例（生产）~100% → 0%
- Validator `strict` 模式 0% → 80%+

但**遗留三个未解问题**：

1. **Rewrite 从未在生产路径启用**：Phase 15.3 引入 Shadow Rewrite 层，但在 Phase 14 之前始终为纯实验（观测），生产文本永远是 `original_text`。
2. **Validator V2 缺乏审计链**：判定结果只输出最终 verdict，无 LLM 原始输出、无 Retriever 证据分层、无 fallback 标记，导致 rescue precision 无法评估。
3. **B2-2 Production Bridge 缺乏 KPI 度量**：白名单 `["plot_flag"]`、阈值 0.90 的 rescue 逻辑无持久化审计，无法计算真实 rescue rate。

### 1.2 Phase 15 的核心命题

> **将 Rewrite 从"纯观测实验"升级为"生产默认路径"，同时为 Validator V2 与 B2-2 Bridge 建立完整的审计链。**

三条主线：

| 主线 | 目标 |
|------|------|
| **A. Validator V2 Audit** | 三值判定 + 21 列 audit + Bridge KPI 度量 |
| **B. Rewrite Productionization** | Rewrite Selection Contract + Structural Lock + Flip |
| **C. Hardening** | P0-11/12/13/14 修复 + 日志降级 + 数据修复 |

---

## 二、决策

### 2.1 总体架构决策

**Phase 15 建立"双轨制 Rewrite 生产架构"：**

```
ControlledWriter.execute()
       │
       ▼
original_text ─────────────────────┐
       │                            │
       ▼                            │
_select_rewrite()  ── 6 级决策 ────┤
       │                            │
       ▼                            │
  ┌────────────────────────┐        │
  │ Rewriter (ShadowRewriter)│      │
  └────────────────────────┘        │
       │                            │
       ▼                            │
  rewritten_text ─── StructuralLock.check_async() ────┐
       │                            │                │
       ▼                            │                │
  ┌────────────────────────┐        │                │
  │ RewriteSelectionResult │        │                │
  │  - selected_source     │        │                │
  │  - selection_reason    │        │                │
  │  - structural_safe     │        │                │
  └────────────────────────┘        │                │
       │                            │                │
       ▼                            ▼                ▼
  final_text = original | rewritten  (由 selected_source 决定)
       │
       ▼
  writer_artifact v1.2（双轨保留：original_text + rewritten_text）
       │
       ▼
  ValidatorAgent（SemanticValidator + B2-2 Bridge）
       │
       ▼
  SceneCompletionService（事件持久化 + Projection 更新）
```

**核心原则**：

> **Rewrite 是"选择性生效"，不是"必然生效"。** 每个场景独立决策，selection 结果作为一等数据契约记录。

---

### 2.2 子决策一：Phase 15.7 — Validator V2 Freeze（CLOSED）

**问题**：Phase 14 的 Validator 只输出三态结论（PASSED / DEGRADED / FAILED），缺乏：
- LLM 原始判定（审计"降级"不可追溯）
- Retriever 证据分层（无法判断是"未检索"还是"检索为空"）
- Fallback 标记（无法区分 EXACT/ALIAS/SEMANTIC 与 L4 兜底）
- B2-2 Bridge rescue 的可量化 KPI

**决策**：建立**双审计表**：

| 审计表 | 粒度 | 职责 |
|--------|------|------|
| `validator_v2_audit` | claim-level（21 列） | Structural → Retrieval → LLM → Final 完整判定链 |
| `bridge_outcome_audit` | bridge-level | trigger_status / execution_status / rescued 三分 |

**关键设计**：

#### C3.3 Validator V2 三值判定

```python
class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"
```

**5 条不变量**：
1. realm_change 必须走结构验证（EXACT_MATCH / MISMATCH / NO_EVENT），LLM 只能判 CONTRADICTED，不能判 SUPPORTED
2. 三态语义严格保留：`None` / `[]` / `[N]` 不可归一
3. LLM 原始 verdict 永不降级存储
4. `fallback_applied=True` 时必标记 matched_layer=NONE
5. `bridge_audit` 与 `claim_audit` 严格分离，不互相写入

#### C3.4.3B Claim + Bridge Observation

`validator_v2_audit` **21 列**设计：

```
上下文:   novel_id, volume_num, chapter_num, scene_idx, scene_id
来源:     source (production|c3_3b_eval|b2_2_bridge), mode (production|shadow_only)
Claim:    claim_id, state_change_type, raw_state_change_type
结构层:   structural_check (exact_match|mismatch|no_event|NULL)
证据层:   evidence_candidates_found, retrieved_evidence_count, retrieved_evidence_ids
LLM 层:   llm_invoked, llm_raw_verdict, llm_raw_confidence, llm_raw_reason, llm_evidence_ids
最终层:   final_verdict, matched_layer, final_confidence, fallback_applied, final_reason
版本:     executed_at, validator_version, contract_hash
```

#### C3.4.3B.5 Replay — v5 final baseline

| 指标 | 值 |
|------|-----|
| Rescue Rate | **0.7443** |
| Triggered | 442 |
| Rescued | 329 |
| Execution error | 0 |
| `no_production_types` 占比 | 14.2% |

#### B2-2 Production Bridge 冻结配置

| 项 | 值 |
|---|---|
| 白名单 | `["plot_flag"]` |
| Shadow-only | realm_change, knowledge_gain, relationship_change, inventory_acquire, location_change |
| confidence 阈值 | **0.90** |
| multi-claim 安全 | 使用 `min(confidence)` 而非 `max` |
| audit 预算 | 1 秒（**仅覆盖 audit persistence**，不含 validate_contract） |

#### R-03 修复

原 B2-2 Bridge 的 `try_rescue` 在 `validate_contract()` 内部耗时不可控。修复方案：**deadline 在 `validate_contract()` 返回后才建立**，只覆盖 claim audit + bridge audit。skip 分支各自建立独立 deadline。

#### C3.5 Independent Correctness Oracle（DEFERRED）

原计划实现独立 Oracle 来度量 False Rescue / Rescue Precision。Phase 15.7 决定 **DEFERRED**——当前 rescue rate 是充分基线，precision 需要独立数据源支持。

---

### 2.3 子决策二：Phase 15.8 — Rewrite Productionization（CLOSED）

**问题**：Rewrite 从 Phase 15.3 引入以来，始终停留在 Shadow 层，生产文本永远是 `original_text`。Flip 需要**三个前置条件**：

1. 有明确的 Selection Contract（否则无法区分"未尝试"和"尝试失败"）
2. 有确定性的结构守卫（否则 rewrite 可能破坏场景结构）
3. 有可审计的语义保留检查（否则 rewrite 可能悄悄改剧情）

**决策**：分 4 个 Commit 逐步落地。

#### Commit 1：Rewrite Selection Contract（`f049e39`）

**新建 6 字段契约**：

```python
@dataclass
class RewriteSelectionResult:
    selected_source: str          # "original" | "rewritten"
    selection_reason: str         # RewriteSelectionReason.value
    rewrite_available: bool
    structural_safe: bool         # Commit 1 恒 False
    rewrite_attempted: bool
    rewrite_failure_reason: Optional[str]
```

**6 级决策顺序**（`_select_rewrite()`）：

| 顺序 | 条件 | Reason |
|------|------|--------|
| 1 | rewriter 未注入 | `no_rewriter_injected` |
| 2 | text < 50 chars | `text_too_short` |
| 3 | execution_contract 为空 | `missing_execution_contract` |
| 4 | rewrite 异常/返回空 | `rewrite_unavailable` |
| 5 | structural_safe = False | `structural_unsafe`（Commit 1 硬编码） |
| 6 | structural_safe = True | `selected`（Commit 3 才启用） |

**writer_artifact schema 1.1 → 1.2**：

```python
{
    "schema_version": "1.2",
    "scene_text": ...,
    "events": ...,
    "original_text": ...,      # 新增
    "rewritten_text": ...,     # 新增
    "rewrite_attempted": ...,  # 新增
    "selection": {...},        # 新增
    "final_text": ...,         # 新增
    "selected_source": ...,    # 新增
    "selection_reason": ...,   # 新增
    "validation_original": None,
    "validation_rewritten": None,
}
```

**Rewrite 重试策略**：`MAX_REWRITE_ATTEMPTS = 3`，超时/空返回重试，其他异常不重试。

#### Commit 2：Structural Lock（观测）

**新建 `StructuralLock` 类**，原计划 5 条规则，经校准**弃用 3 条**：

| 规则 | 状态 | 原因 |
|------|------|------|
| `characters_preserved` | ✅ 保留 | 契约中角色必须出现在 rewritten |
| `units_preserved` | ❌ 弃用 | 语义不变量，字面匹配必然误杀 |
| `state_changes_preserved` | ❌ 弃用 | 结构化字段名不会在正文出现 |
| `no_new_characters` | ❌ 弃用 | jieba nr 识别不可靠 |
| `length_floor` | ✅ 保留 | rewritten/original ≥ 0.6 |

#### Commit 3A：语义规则（观测）

**新增 async 语义规则**：`key_events_semantically_preserved`

- 只检查 4 类可验证的 state_change：`inventory_acquire` / `location_change` / `knowledge_gain` / `realm_change`
- 每条 state_change 构造自然语言描述 → embedding
- 对 rewritten 分句 → embedding
- 最大 cosine ≥ **0.35** 视为保留

**阈值校准**：0.35 是基于 bge-small-zh-v1.5 中文短文本对的实验值。

#### Commit 3B：Rewrite Flip 生产

**唯一 flip 点**：

```python
# 从硬编码 False → 真实 StructuralLock 判定
_lock = await StructuralLock().check_async(
    original_text=original_text,
    rewritten_text=rewritten_text,
    contract=execution_contract,
)
structural_safe = _lock.structural_safe
```

**final_text 决定逻辑**：

```python
if selection.selected_source == "rewritten" and rewritten_text:
    final_text = rewritten_text
else:
    final_text = original_text
```

**删除 `commit1_invariant_forced` 防御块**——Commit 1 的"永远 original"是过渡态，Commit 3B 后 selection 是权威。

---

### 2.4 子决策三：Phase 15.8 期间的 P0/P1 修复

**8+ Bug 修复清单**：

| # | 问题 | 修复 |
|---|---|---|
| P0-11 | ControlledWriter 无 grammar → JSON 解析失败 | `_do_call` 附加 GBNF grammar |
| P0-12 | ControlledWriter 用 NoOp SemanticValidator | `build_writer_runtime` 单例构造共享 LocalHttpEmbeddingProvider |
| P0-13 | 段级验证做场景级契约匹配 → 3x retry | 段级只做结构验证（长度 + 单元） |
| P0-14 | sanity 拒绝时残留 scene_text | 清空 + `sanity_blocked=True` |
| P0-14-fix2 | LangGraph `KeyError: __end__` | `add_conditional_edges` 加 `END: END` |
| P0-14-fix3 | Planner realm_change "中期不稳定" 被强转 1 | 关键词匹配 + 无法解析时丢弃 |
| P1 | Rewrite 无重试 | `MAX_REWRITE_ATTEMPTS=3` |
| P1 | embedding_dim 384 vs 512 | 统一 512 |
| P1 | 场景保存未跳过门 | 去除门控 |
| P1 | Shadow 完整性 | fallback + tag |
| P1 | 跨章重复（chap_035） | `previous_scene_tail` 章末清空 |
| P1 | `shadow/runner.py` 缺 `import json` | 补充 |
| Data | `world_snapshot` ch72：`炼气7层 → 金丹1层` | 手工 SQL |

---

### 2.5 子决策四：Phase 15.9-E1 — 日志降级（部分完成）

**问题**：Phase 15.7/15.8 期间为调试加了 **112 处 `logger.critical`** + 情绪化标记（`🔥🔥🔥` / `🚨🚨` / `!!!...!!!`） + `stderr.write` 残留。

**决策**：

- **E1a**：`logger.critical` → `logger.debug`（项目级，112 → 0）
- **E1b**：删除 `print(..., file=sys.stderr)` / `sys.stderr.write` / flush
- **E1c**：删除 `nodes.py` 的 DIAG-C2 块 + `event_store.py` 的未用 `NarrativeProjector` import
- **E1d（关键修复）**：`validator.py` 的 `_check_loop_advancement` 因 E1 前的 sed 误删，导致模块级函数 → 后续所有类方法不可达 → **`ValidatorAgent._get_validator` 报 AttributeError**。修复后 SemanticValidator 首次在 ValidatorAgent 路径真正生效。
- **E1e**：`_check_loop_advancement` 改走 `llm_router_pool`（原直连 8081 coder 端点）

**E2（物理删除 debug 块）DEFERRED** —— E1 已让日志可读，E2 是美学优化，优先级低。

**Phase 15.9 其他目标（DEFERRED）**：
- 对话占比（prompt 层面无效，需 QualityGate 结构性 retry）
- 场景长度下限
- 意象去重
- Narrative Control Plane

---

## 三、最终架构状态

```
Application / API (src/api/main.py)
    ↓
Orchestrator (LangGraph)
    ├── plan_node → ScenePlanningService
    │     ├── PlannerAgent → planner_outputs（含 NarrativeIntent + PlanningContract）
    │     ├── SceneEventValidator (P1 降级版，只阻断占位符)
    │     ├── ContractNormalizer (推断 state_changes)
    │     ├── ContractConsistencyValidator
    │     ├── DramaPlannerAgent
    │     └── _persist_scene_plans → scene_execution_units
    │
    ├── writer_node
    │     ├── ContractSanityGuard (B2-1A fail-closed)
    │     ├── ControlledWriter.execute()
    │     │     ├── _execute_segment (GBNF + QualityGate + StructuralLock)
    │     │     └── _select_rewrite (Commit 1/3A/3B)
    │     ├── writer_artifact v1.2
    │     └── _run_shadow_rewrite (异步)
    │
    └── validate_node
          ├── B2-2 ProductionBridge.try_rescue（白名单 plot_flag, 阈值 0.90）
          ├── 双轨观察（original + rewritten）
          ├── Shadow 写入
          └── SceneCompletionService
                ├── StateDelta.apply_to
                ├── append_event (含 PerceptionPropagation)
                ├── ProjectionUpdater (事务内)
                ├── 相变检测
                ├── 快照保存
                └── writing_progress 推进

Writer Runtime (Composition Root)
    ├── RuntimeCapabilities (Audit + Snapshot)
    ├── RuntimeServices (含 production_rewriter)
    ├── SemanticValidator (P0-12 共享实例)
    ├── ValidationPolicy (dev/prod)
    └── Shadow (Runner + Recorder)

Validator V2 Runtime
    ├── claim_builder.py
    ├── evidence_retriever.py
    ├── semantic_judge.py
    ├── production_bridge.py (B2-2)
    └── audit_writer.py / bridge_audit_writer.py

Event Sourcing / Projection
    ├── narrative_events (append-only)
    ├── predicates (projection cache)
    ├── world_snapshots
    └── chapter_budget

Persistence (PostgreSQL + pgvector)
```

---

## 四、API 冻结清单

### 4.1 Rewrite Selection Contract

```python
class RewriteSelectionReason(str, Enum):
    NO_REWRITER_INJECTED       = "no_rewriter_injected"
    TEXT_TOO_SHORT             = "text_too_short"
    MISSING_EXECUTION_CONTRACT = "missing_execution_contract"
    REWRITE_UNAVAILABLE        = "rewrite_unavailable"
    STRUCTURAL_UNSAFE          = "structural_unsafe"
    SELECTED                   = "selected"

@dataclass
class RewriteSelectionResult:
    selected_source: str
    selection_reason: str
    rewrite_available: bool
    structural_safe: bool
    rewrite_attempted: bool
    rewrite_failure_reason: Optional[str]
    def to_dict(self) -> Dict[str, Any]: ...
```

### 4.2 Structural Lock

```python
@dataclass(frozen=True)
class LockCheck:
    name: str
    passed: bool
    detail: str

@dataclass(frozen=True)
class LockResult:
    structural_safe: bool
    checks: List[LockCheck]
    failure_summary: str
    def to_dict(self) -> dict: ...

class StructuralLock:
    LENGTH_RATIO_MIN = 0.6
    SEMANTIC_THRESHOLD = 0.35

    def check(original, rewritten, contract) -> LockResult: ...
    async def check_async(original, rewritten, contract) -> LockResult: ...
```

### 4.3 ControlledWriteResult

```python
@dataclass
class ControlledWriteResult:
    text: str
    events: List[Dict]
    segments_used: int
    segments_succeeded: int
    fallback_used: bool
    execution_time: float

    # Phase 15.7-A 双轨字段
    original_text: str = ""
    rewritten_text: Optional[str] = None
    rewrite_attempted: bool = False
    rewrite_failure_reason: Optional[str] = None

    # Phase 15.8 Commit 1 Selection Contract
    selection: Optional[RewriteSelectionResult] = None
```

### 4.4 writer_artifact v1.2

```python
{
    "schema_version": "1.2",
    "scene_text": str,
    "events": List[Dict],
    "foreshadowing": List[str],
    "original_text": str,
    "rewritten_text": Optional[str],
    "rewrite_attempted": bool,
    "rewrite_failure_reason": Optional[str],
    "selection": Optional[Dict],
    "final_text": str,
    "selected_source": str,
    "selection_reason": str,
    "validation_original": Optional[Dict],
    "validation_rewritten": Optional[Dict],
}
```

### 4.5 Validator V2 Output

```python
@dataclass(frozen=True)
class ValidationResultV2:
    claim_id: str
    state_change_type: str
    verdict: Verdict
    matched_layer: MatchLayer
    confidence: float = 0.0
    evidence: Optional[Evidence] = None
    judgement: Optional[SemanticJudgement] = None
    reason: str = ""
    evidence_candidates_found: bool = False
    # C3.4.1 观测
    structural_check: Optional[str] = None
    fallback_applied: bool = False
    # C3.4.2 观测
    retrieved_evidence_count: int = 0
    retrieved_evidence_ids: Optional[List[str]] = None
```

### 4.6 Audit Writers

```python
@dataclass(frozen=True)
class AuditContext:
    novel_id: str
    volume_num: Optional[int] = None
    chapter_num: Optional[int] = None
    scene_idx: Optional[int] = None
    scene_id: Optional[str] = None
    source: str = "production"
    mode: str = "production"
    contract_hash: Optional[str] = None

@dataclass(frozen=True)
class BridgeAuditContext:
    novel_id: str
    volume_num: Optional[int] = None
    chapter_num: Optional[int] = None
    scene_idx: Optional[int] = None
    scene_id: Optional[str] = None

async def record_audit_batch(conn, results, context) -> int: ...
async def record_bridge_outcome(conn, context, ...) -> bool: ...
```

### 4.7 B2-2 Production Bridge

```python
class B2_2ProductionBridge:
    PRODUCTION_TYPES: List[str] = ["plot_flag"]
    SHADOW_ONLY_TYPES: List[str] = [...]
    CONFIDENCE_THRESHOLD: float = 0.90
    AUDIT_WRITE_TIMEOUT: float = 1.0

    async def try_rescue(
        contract, writer_events, scene_text,
        original_validation_result, scene_id,
        audit_context=None, bridge_audit_context=None,
    ) -> B2_2ProductionResult: ...
```

---

## 五、验证结果

### 5.1 Phase 15.7 v5 冻结基线

| 指标 | 值 |
|------|-----|
| Rescue Rate | **0.7443** |
| Triggered | 442 |
| Rescued | 329 |
| Execution error | **0** |
| `no_production_types` 占比 | 14.2% |

### 5.2 Phase 15.8 生产验证

| 指标 | 值 |
|------|-----|
| Rewrite flip 率 | ~50%（ch72-ch85） |
| StructuralLock 拦截率 | ~25% |
| SemanticValidator `passed=True` | ~100% |
| `contract_observables matched` | 修复后 ≈ 100% |
| 场景平均长度 | 128-544 字（不稳定） |
| 对话占比 | 1-5%（**远低于 25-60% 目标**） |
| 场景数 | 2 或 3（Planner 决策，非 bug） |

### 5.3 Phase 15.9-E1 验证

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| `logger.critical` | 112 | 0 |
| `logger.debug` | - | 169 |
| `stderr.write` | 1 | 0 |
| `_get_validator` 可达 | ❌ | ✅ |
| SemanticValidator 在 ValidatorAgent 路径 | AttributeError | passed=True, matched=N |

**ch77 重跑验证**：

```
[15.7-B1] Dual observation: original_pass=True, rewritten_pass=True
[15.8-C1] Selection observation: selected_source=rewritten, selection_reason=selected, structural_safe=True
SceneCompletion succeeded, chapter_finished=False
```

---

## 六、后果

### 正面

- ✅ **Rewrite 首次在生产路径启用**：`selected_source=rewritten` 在 ch75 生产确认
- ✅ **Rewrite Selection 是显式契约**：6 字段 + 6 级决策，无隐式状态
- ✅ **Structural Lock 是确定性守卫**：2 条 sync 规则 + 1 条 async 语义规则，无 LLM 依赖
- ✅ **Validator V2 有完整审计链**：21 列 claim audit + bridge audit 双表分离
- ✅ **B2-2 Bridge 有可量化 KPI**：Rescue Rate 0.7443（v5 baseline）
- ✅ **writer_artifact v1.2 双轨保留**：`original_text` + `rewritten_text` 均可追溯
- ✅ **P0-11/12/13/14 全修复**：grammar / 共享 validator / 段级验证 / sanity residual
- ✅ **Phase 15.9-E1 让日志可读**：112 `critical` → 0

### 负面 / 已知遗留

- ⚠️ **场景长度不均**：128-544 字波动
- ⚠️ **对话占比 1-5%**：远低于 25-60% 目标（prompt 层面无效，需结构性 QualityGate）
- ⚠️ **意象疲劳**：青苔/青铜/玉佩等重复
- ⚠️ **场景因果断裂**：Planner 生成无依赖，跨场景/跨章跳变
- ⚠️ **章节连贯性**：地点乱跳
- ⚠️ **E2（物理删除 debug 块）DEFERRED**
- ⚠️ **C3.5 Independent Correctness Oracle DEFERRED**
- ⚠️ **`WritingContract` 三重定义**（`contracts.py` / `contracts/__init__.py` / `contracts/models.py`）
- ⚠️ **DramaPlanner 无重试**（3 次全失败 fallback）
- ⚠️ **`_validate_contract_observables` 生产失效**（Writer 自由 type 无法匹配 enum，SemanticValidator 是权威）

### 冻结后建议（Phase 16+）

| 主题 | 目标 |
|------|------|
| Narrative Control Plane | Planner 层因果依赖图 + 跨场景 location 一致性 |
| 对话占比结构性控制 | QualityGate retry（非 prompt 强制） |
| 意象去重 | 负向约束注入 context_compiler |
| WritingContract 去重 | 三处定义归一 |
| Independent Correctness Oracle | 度量 False Rescue / Rescue Precision |

---

## 七、相关 ADR

| ADR | 关系 |
|-----|------|
| ADR-023 | Planning Contract 稳定接口（Writer 消费 Contract） |
| ADR-024 | Incremental Execution（ControlledWriter） |
| ADR-025 | Validator as Controller（Validator 三层 Control） |
| ADR-026 | Empirical Control Model（SceneSpec 控制读者体验） |
| ADR-031 | Phase 10 Audit & Snapshot Runtime（审计基础设施） |
| ADR-034 | Phase 13 Narrative Coherence Layer & Runtime Closure |
| ADR-035 | Phase 14 Narrative Contract Hardening |
| **ADR-036** | **Phase 15 Rewrite Productionization & Validator V2 Freeze（本文）** |

---

## 八、冻结状态

| 子阶段 | 设计状态 | 实施状态 |
|--------|----------|----------|
| Phase 15.7 | ✅ Design Freeze | ✅ Implemented（CLOSED） |
| Phase 15.8 Commit 1 | ✅ Design Freeze | ✅ Implemented |
| Phase 15.8 Commit 2 | ✅ Design Freeze | ✅ Implemented |
| Phase 15.8 Commit 3A | ✅ Design Freeze | ✅ Implemented |
| Phase 15.8 Commit 3B | ✅ Design Freeze | ✅ Implemented |
| Phase 15.8-fix（P0-11~14） | ✅ Design Freeze | ✅ Implemented |
| Phase 15.9-E1 | ✅ Design Freeze | ✅ Implemented |
| Phase 15.9-E2 | ✅ Design Freeze | ⏸ DEFERRED |
| Phase 15.9 内容质量 | ✅ Design Freeze | ⏸ DEFERRED |
| Phase 15.9 Narrative Control | ✅ Design Freeze | ⏸ DEFERRED |

**Phase 15 已冻结。**
**Tag: `phase15-final`**

---

## 九、一句话总结

> **Phase 15 完成了 Rewrite 的生产化——从"Shadow 层纯观测"升级为"6 级决策 + 确定性结构守卫 + 语义保留检查 + 双轨留痕 + 完整审计链"的生产默认路径。同时为 Validator V2 与 B2-2 Bridge 建立了双表审计（21 列 claim audit + bridge outcome audit），并在 Rescue Rate 0.7443 基线上冻结。Phase 15.8 期间修复 8+ P0/P1 bug，Phase 15.9-E1 完成日志降级并修复 validator.py 缩进 bug（首次让 SemanticValidator 在 ValidatorAgent 路径真正生效）。内容质量（对话占比、场景长度、意象去重）与 Narrative Control Plane 明确移交 Phase 16+。**

---

**写完。Phase 15 CLOSED。**