基于您的要求，我已将所有 Phase 9 相关 ADR 合成一份统一的架构文档：**ADR-030: Narrative Intelligence Foundation**。

这份文档整合了 Phase 9.0 至 9.4 的所有关键决策，涵盖：

1. **阶段划分**：从语义协议 (9.0) 到自适应运行时 (9.4) 的完整演进路径。
2. **核心组件**：NarrativeIntent, IntentDimension, ConflictResolver, AdaptiveSelector 等。
3. **架构决策**：ResolutionPlan 作为执行事实源、自适应闭环、模型分配策略。
4. **部署建议**：AI Max 395 128G 硬件下的模型选型与灰度发布方案。

文档已整理为适合归档的形式，为后续 Phase 10 及未来优化奠定基础。

---

# ADR-030: Narrative Intelligence Foundation

## 状态
Accepted

## 日期
2026-07-18

---

## 一、背景与问题

AI Factory 从 Phase 8（Capability Runtime）演进后，已具备结构化的意图表达能力，但存在以下问题：

1. **意图缺乏语义结构** —— NarrativeIntent 仅为文本描述，无法比较、冲突检测或维度化。
2. **满意度评估粗糙** —— 仅依赖关键词匹配，无法理解意图是否被真正实现。
3. **冲突解决单一** —— 仅支持 PRIORITY 策略，无法处理更复杂的叙事权衡。
4. **策略选择静态** —— 无法根据历史执行效果优化策略选择。

Phase 9 的目标是将叙事生成从“规则驱动”升级为“语义驱动 + 反馈自适应”的系统。

---

## 二、阶段划分

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 9.0a | Semantic Protocol（叙事基础类型） | ✅ |
| Phase 9.0b | Reference Realizer（参考实现） | ✅ |
| Phase 9.1 | Intent Compiler（意图编译器） | ✅ |
| Phase 9.2.1 | Semantic Intent Upgrade（语义意图） | ✅ |
| Phase 9.2.2 | Semantic Satisfaction（语义满意度） | ✅ |
| Phase 9.3.1 | Conflict-Aware Realization（冲突感知实现） | ✅ |
| Phase 9.3.2 | Conflict Strategy Framework（冲突策略框架） | ✅ |
| Phase 9.4 | Adaptive Narrative Runtime（自适应运行时） | ✅ |

---

## 三、核心架构

### 3.1 完整闭环

```
NarrativeIntentSet
        │
        ▼
IntentResolver
        │
        ▼
Conflict Detection
        │
        ▼
StrategyDecisionProvider
        │
        ├── RuleSelector (deterministic)
        │
        └── AdaptiveSelector (feedback-driven)
        │
        ▼
ResolutionPlan
        │
        ▼
CompositeResolver
        │
        ▼
ResolutionContext
        │
        ▼
ReferenceNarrativeRealizer
        │
        ▼
Prompt / Artifact
        │
        ▼
SatisfactionReport
        │
        ▼
StrategyFeedbackCollector
        │
        ▼
StrategyPerformanceTracker
        │
        ▼
PerformanceRepository
        │
        └── (feedback loop to AdaptiveSelector)
```

### 3.2 关键组件

| 组件 | 职责 |
|------|------|
| **NarrativeIntent** | 声明式目标，包含 `dimension`（语义维度）和 `desired_effect`（描述） |
| **IntentDimension** | 语义维度（DIALOGUE, EMOTION, TRANSITION 等）+ 方向（INCREASE/DECREASE） |
| **IntentResolver** | 检测意图冲突，生成 ResolutionPlan |
| **ResolutionPlan** | 执行事实源，包含 `conflicts` 和 `resolutions` |
| **ConflictResolver** | 策略接口（PRIORITY, BALANCE, SYNTHESIS） |
| **StrategySelector** | 根据冲突特征选择策略（规则/自适应） |
| **CompositeResolver** | 注入 StrategyDecisionProvider，执行解析 |
| **NarrativeContext** | 包含 ResolutionContext，供 Realizer 使用 |
| **ReferenceNarrativeRealizer** | 根据 plan 和 context 生成 Prompt，调用 LLM |
| **IntentSatisfaction** | 评估意图实现程度（关键词/LLM 语义） |
| **StrategyPerformanceTracker** | 记录策略执行历史与满意度 |
| **AdaptiveSelector** | 基于历史性能 + 特征 Eligibility 自适应选择策略 |

---

## 四、关键架构决策

### 4.1 ResolutionPlan 是执行事实源
- QualityLoop **不**持有 IntentResolver，只消费预解析的 ResolutionPlan。
- 保证决策一致性，避免重复解析。

### 4.2 ConflictResolver 只负责解决冲突
- 不修改 NarrativeIntent，不生成新意图。
- SYNTHESIS 策略在 Phase 9.3.2 只生成 `rationale` 文本，不依赖 LLM。

### 4.3 AdaptiveSelector 与 RuleSelector 并列
- 二者均实现 `StrategyDecisionProvider` 协议。
- AdaptiveSelector 根据冲突特征计算 **Eligible Strategies**，然后选择历史满意度最高的策略。
- 支持 `deterministic` 模式确保测试可重复。

### 4.4 Feedback 与决策解耦
- `StrategyPerformanceTracker` 只负责记录，不负责分析。
- `StrategyFeedbackCollector` 作为中间层，解耦 QualityLoop 与 Tracker。
- FeedbackEvent 与 ResolutionPlan 通过 `resolution_id` 关联，不嵌套。

### 4.5 数据模型不可变
- `StrategyPerformance`, `ConflictResolution`, `StrategyDecision` 等均为不可变 dataclass。
- 符合 ADR-035（RuntimeSnapshot Immutable）原则。

---

## 五、关键接口冻结

```python
# 意图
NarrativeIntent(dimension: IntentDimension, desired_effect: str)

# 维度
IntentDimension(id: str, direction: IntentDirection)

# 解析
IntentResolver.resolve(intents) -> ResolutionPlan

# 决议
ResolutionPlan(primary_intents, conflicts, resolutions)

# 策略决策
StrategyDecision(strategy, confidence, reason, selected_by)

# 提供者
StrategyDecisionProvider.decide(conflicts, intents) -> StrategyDecision

# 评估
IntentSatisfaction.evaluate(artifact, intents) -> SatisfactionReport

# 跟踪
StrategyPerformanceTracker.record(strategy, satisfaction, iterations, conflict_id)
StrategyPerformanceTracker.get_performance(strategy) -> StrategyPerformance

# 收集
StrategyFeedbackCollector.collect(plan, report, iterations)

# 工厂
create_adaptive_resolver(mode, repository, min_records, confidence_threshold)
create_deterministic_resolver()
```

---

## 六、性能与部署

### 6.1 模型选型（AI Max 395 128G）

| 任务 | 推荐模型 | 内存 |
|------|----------|------|
| Writer | Qwen2.5-72B-Instruct-Q4_K_M | ~45 GB |
| Planner | Qwen2.5-32B-Instruct-Q5_K_M | ~22 GB |
| Semantic Eval | Qwen2.5-32B-Instruct-Q5_K_M | ~22 GB |
| Deep Validator | DeepSeek-R1-70B-Q4_K_M | ~42 GB（按需） |

### 6.2 灰度发布

```
Staging (10章)
    ↓
灰度 10%
    ↓
灰度 50%
    ↓
全量 100%
```

### 6.3 关键开关

```python
ADAPTIVE_RUNTIME_ENABLED = False   # kill switch
ADAPTIVE_ROLLOUT_PERCENTAGE = 0    # 0-100
```

---

## 七、测试状态

```
============================= 186 passed in 0.20s ==============================
```

- Phase 9.2.1 基础测试：111 项
- Phase 9.2.2 语义满意度：新增 ~12 项
- Phase 9.3.1 冲突解析：新增 ~14 项
- Phase 9.3.2 策略框架：新增 ~15 项
- Phase 9.4 自适应运行时：新增 ~34 项

---

## 八、后果与演进

### 正面
- 叙事系统从“规则驱动”升级为“语义驱动 + 反馈自适应”
- 冲突策略可扩展（PRIORITY/BALANCE/SYNTHESIS）
- 策略选择可基于历史效果优化
- 完整的 Feedback → Adaptation 闭环

### 负面
- 增加 LLM 语义评估调用（可配置关闭）
- 需要维护性能数据（内存/数据库）
- 自适应模式下需观察策略分布是否合理

### 未来演进
- Phase 10.1: 性能数据持久化（Database Repository）
- Phase 10.2: 策略分析与可视化 Dashboard
- Phase 10.3: A/B 策略实验框架
- Phase 10.4: 策略合成增强（LLM 生成 SYNTHESIS 描述）

---

## 九、相关 ADR

| ADR | 关系 |
|-----|------|
| ADR-023 | Planning Contract 稳定接口 |
| ADR-038 | Satisfaction Evaluation Result |
| ADR-039 | Semantic Evaluator Injection |
| ADR-040 | EvaluationResult Immutable |
| ADR-041 | Conflict-Aware Realization |
| ADR-042 | Conflict Resolution Strategy Framework |
| ADR-043 | Adaptive Narrative Runtime |

---

**ADR-030 记录了 Phase 9 的完整架构决策，为 Phase 10 及未来演进提供基础。**