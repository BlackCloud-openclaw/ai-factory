# ADR-034: Phase 13 Narrative Coherence Layer & Runtime Closure

| 项目 | 内容 |
|------|------|
| **状态** | ✅ 已接受（已冻结） |
| **日期** | 2026-08-01 |
| **决策者** | Phase 13 团队 |
| **影响范围** | `Planner`, `Writer`, `Validator`, `ControlledWriter`, `AgentState`, `Orchestrator`, `PlanningContract` |

---

## 一、背景与问题陈述

### 1.1 回顾：Phase 12 后的状态

经过 Phase 10-12 的建设，AI Factory 已具备：
- 编译器驱动的 Runtime（Phase 6）
- 能力驱动的 Surface Framework（Phase 7-8）
- 完整的审计与快照基础设施（Phase 10）
- 标准化 Benchmark 与质量评估（Phase 12）

然而，**Phase 12 测试与长期运行日志表明**，长篇小说连续生成仍存在显著的“割裂感”：
- 场景缺乏明确的叙事角色（`scene_role` 缺失）
- 章节之间存在“重启感”（未解决的冲突消失，读者失去参照系）
- Planner 输出不稳定，偶发格式错误导致断裂
- 叙事焦点偶尔丢失，使后续章节缺乏方向

### 1.2 根本原因

上述症状不是孤立 Bug，而是**同一个深层次问题的不同表现**：

> 系统缺少一个显式的“叙事控制平面”（Narrative Control Plane）。

此前链路是扁平的：

```
Planner → SceneSpec → Writer → SceneText
```

`SceneSpec` 描述“写什么”，但没有描述“为什么现在写”以及“写完后会改变什么”。

更严重的问题是 **状态传输隐式化**：`planner_outputs`、`narrative_intent` 等核心业务状态通过 `metadata` 字典隐式传递，无类型检查、无契约约束，导致 Writer 和 Validator 无法可靠地接收 Planner 的意图。

---

## 二、决策

### 2.1 总体架构决策

**引入 Narrative Control Plane（叙事控制平面）** 作为 `Planner` 与 `Writer` 之间的显式语义层，同时建立 **强类型运行时闭环（Runtime Closure）** 确保所有核心状态通过类型化契约传递。

新链路：

```
WorldState + CharacterState + NarrativeProjection
        │
        ▼
Planner
        │
        ▼
NarrativeIntent + PlanningContract
        │
        ▼
Writer Runtime (ControlledWriter)
        │
        ▼
Scene Output
        │
        ▼
Validator → QualityGate → Retry Feedback Loop
```

**核心原则**：

> 场景不是孤立的文本单元，而是叙事状态机中的一次状态转移。

即：
- 输入：当前叙事状态
- 执行：场景生成
- 输出：新的叙事状态

---

### 2.2 子决策一：Runtime State Contract（13.2.2）

#### 问题
`planner_outputs` 和 `narrative_intent` 通过 `metadata` 隐式传递，Writer 和 Validator 通过 fallback 恢复，无类型安全。

#### 决策
- `PlannerAgent.run()` 返回 `planner_outputs: List[PlannerOutput]`
- `ScenePlanningResult` 包含 `planner_outputs` 字段
- `plan_node` 通过 `StatePatch.planner_outputs` 写入 `AgentState`
- `writer_node` 和 `validate_node` 从 `state.planner_outputs` 强类型读取
- `metadata` 不再作为核心业务状态的主传输通道

#### 数据流
```
PlannerAgent
        │
        ▼
ScenePlanningResult.planner_outputs
        │
        ▼
plan_node → StatePatch.planner_outputs
        │
        ▼
AgentState.planner_outputs
        │
        ├── writer_node（强类型读取）
        │
        └── validate_node（强类型读取）
```

---

### 2.3 子决策二：Contract Completion & Signal Provenance（13.2.3A）

#### 问题
Planner 生成的 `PlanningContract` 中 `observables.state_changes` 和 `constraints` 常为空，Validator 无信号可验。

#### 决策
- 引入 `ContractNormalizer`，在 `ScenePlanningService` 中标准化 Contract
- 引入 `EventClassifier`，将 `must_events` 分类为 `EventType`（`REALM_ADVANCE`, `ITEM_ACQUIRE`, `LOCATION_CHANGE` 等）
- 引入 `StateChangeFactory`，根据 `EventType` 生成 `StateChange` 对象
- 增加 `SignalSource` 枚举追踪信号来源：
  - `LLM`：Planner 直接生成
  - `INFERRED`：系统推断
  - `UNKNOWN`：未声明来源
- 扩展 `StateChange` 增加 `id` 和 `source` 字段
- 扩展 `PlanningContract` 增加 `enrichment` 元数据（`sources`, `rules_applied`, `input_hash`）

#### 关键约束
- Normalizer 不覆盖 LLM 生成字段
- 推断信号标记 `source=INFERRED`
- 幂等性：`normalize(normalize(c)) == normalize(c)`
- 确定性：相同输入产生相同输出

---

### 2.4 子决策三：Semantic Validator Upgrade（13.2.3B）

#### 问题
Validator 对 `must_events` 使用精确字符串匹配，自然语言变体被误判为未完成。

#### 决策
- 引入 `SemanticValidator` 主管道
- 实现三阶段 Matcher Pipeline（按顺序执行，匹配即停止）：
  1. **ExactMatcher**：精确字符串匹配
  2. **NormalizedMatcher**：去除停用词后匹配
  3. **KeywordCoverageMatcher**：关键词覆盖率 ≥ 60%
  4. **EmbeddingMatcher**：Sentence-level embedding 相似度 ≥ 0.30（仅低置信度区域）
- 扩展 `ValidationResult` 增加 `blocking_missing` 字段
- 策略 B：`LLM`、`SYSTEM`、`INFERRED` 来源缺失阻断；`UNKNOWN`、`NORMALIZED` 不阻断
- 引入 `SignalWeightPolicy`：`LLM=1.0`, `SYSTEM=0.8`, `INFERRED=0.6`, `NORMALIZED=0.5`, `UNKNOWN=0.3`
- 引入 `ValidationEvidence` 提供可解释验证

---

### 2.5 子决策四：Quality Gate & Control Loop（13.2.3C）

#### 问题
Validator 输出二值（通过/不通过），Writer 无法获得连续的控制信号。

#### 决策
- 引入 `QualityGate` 组件，将 `ValidationResult` 转化为 Writer 控制信号
- 决策规则：
  - `pass`：`score >= 0.8` 且无阻断性缺失
  - `retry`：有阻断性缺失或 `score < 0.5`，且重试次数未耗尽
  - `force_pass`：重试次数耗尽，强制通过（避免无限循环）
- 反馈注入：`retry` 时 `QualityGate.feedback` 作为 `error_hint` 注入下一轮 Prompt
- `error_hint` 跨 attempt 保留，实现 **Guided Retry Loop**

#### 闭环链路
```
attempt 0 → LLM → Validator → QualityGate.retry → feedback
        │
        ▼
attempt 1 → LLM (with feedback) → Validator → QualityGate.pass
```

---

### 2.6 子决策五：Metadata Cleanup（13.2.3D）

#### 问题
`metadata` 仍残留部分核心状态的 fallback 读写（`MigrationFallback` 日志）。

#### 决策
- 从 `writer_node` 和 `validate_node` 中删除所有 metadata fallback 逻辑
- 从 `ScenePlanningService` 和 `PlannerAgent` 中删除 `metadata["planner_outputs"]` 写入
- 从 `execute.py` 中删除 `metadata["scene_plan_list"]` 写入，改用 `state.scene_plan_list`
- 明确 `None` vs `[]` 语义：
  - `None` = 状态缺失（致命错误）
  - `[]` = 合法空状态（允许继续）
- 更新 `AgentState.metadata` 注释为“仅辅助/调试”

---

### 2.7 子决策六：ControlledWriter Coverage（13.2.3E）

#### 问题
`ControlledWriter` 的核心路径（激活、分段、重试、降级、状态传递）缺乏集成测试覆盖。

#### 决策
- 新增 `test_controlled_writer_coverage.py`，覆盖 13 个场景：
  - 激活阈值（1 单元→单次，3+ 单元→分段）
  - 分段计数（`_determine_segments`）
  - 重试流（首次失败→二次成功）
  - 反馈注入验证
  - 重试用尽→`force_pass`
  - 降级路径（fallback）
  - 段间状态积累
  - 首次成功路径
  - 真实 SemanticValidator 端到端集成

---

## 三、API 冻结清单

### 3.1 `AgentState` 核心字段
```python
planner_outputs: List[Dict[str, Any]]   # 强类型契约
narrative_intent: Optional[NarrativeIntent]
scene_plan_list: List[Dict[str, Any]]
metadata: Dict[str, Any]                # 仅辅助/调试
```

### 3.2 `ValidationResult`
```python
@dataclass
class ValidationResult:
    passed: bool
    missing: List[str]
    matched: List[ValidationEvidence]
    blocking_missing: List[str]         # Phase 13.2.3C
    overall_confidence: float
    weight_applied: float
```

### 3.3 `QualityGate`
```python
class QualityGate:
    def evaluate(result: ValidationResult, retry_count: int) -> QualityGateResult:
        # decision: "pass" | "retry" | "force_pass"
```

### 3.4 `SignalSource`
```python
class SignalSource(str, Enum):
    UNKNOWN = "unknown"
    LLM = "llm"
    INFERRED = "inferred"
    SYSTEM = "system"
    NORMALIZED = "normalized"
```

---

## 四、后果

### 正面
- ✅ **显式状态契约**：`planner_outputs` 通过 `StatePatch` → `AgentState` 强类型传递，消除隐式依赖
- ✅ **信号可追踪**：每个 `StateChange` 标记来源（LLM/INFERRED/UNKNOWN），审计可溯源
- ✅ **可控生成循环**：`QualityGate` 实现 Guided Retry Loop，Writer 根据反馈修正
- ✅ **语义验证**：Validator 理解自然语言变体，减少误判
- ✅ **可重放性**：确定性 Matcher 和 Normalizer 保证相同输入→相同输出
- ✅ **可观测性**：`ValidationEvidence` 和 `ContractEnrichment` 提供验证证据

### 负面/风险
- ⚠️ **系统复杂性增加**：新增 `ContractNormalizer`、`SemanticValidator`、`QualityGate` 等组件
- ⚠️ **测试工程债**：需要维护集成测试的 mock 行为
- ⚠️ **分段算法锁定**：测试锁定了 `_determine_segments` 的阈值（未来优化需同步更新测试）

---

## 五、相关 ADR

| ADR | 关系 |
|-----|------|
| ADR-023 | Planning Contract as Stable Interface（Writer 消费 Contract） |
| ADR-024 | Incremental Execution as Core Capability（ControlledWriter） |
| ADR-025 | Validator as Controller（Validator 三层 Control） |
| ADR-026 | Empirical Control Model（SceneSpec 控制读者体验） |
| ADR-031 | Phase 10 Audit & Snapshot Runtime（审计基础设施） |

---

## 六、冻结状态

| 子阶段 | 内容 | 状态 |
|--------|------|------|
| 13.2.2 | Runtime State Contract | ✅ 冻结 |
| 13.2.3A | Contract Completion & Signal Provenance | ✅ 冻结 |
| 13.2.3B | Semantic Validator | ✅ 冻结 |
| 13.2.3C | Quality Gate & Control Loop | ✅ 冻结 |
| 13.2.3D | Metadata Cleanup | ✅ 冻结 |
| 13.2.3E | ControlledWriter Coverage | ✅ 冻结 |

**Phase 13 完成标志**：100 个测试通过，Runtime Closure 完整闭环验证。

---

*本 ADR 记录了 Phase 13 的完整架构决策，标志着 AI Factory 从“可生成”系统正式演进为“可控制、可观测、可追溯”的叙事生成平台。*