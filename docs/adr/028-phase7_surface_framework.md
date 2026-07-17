# Phase 7 Surface Framework

## 1. Objective

Phase 7 的目标是**将 Runtime 从功能耦合模型迁移到 Surface 扩展模型**，使新增能力（如 Dialogue、Emotion、Pacing 等）不需要修改 Runtime 核心代码。

核心命题：

> **新增 Surface，Runtime 核心零修改。**

### 1.1 解决的问题

在 Phase 6 中，Runtime 与具体功能（Reasoning、Construction、Prediction）是硬编码耦合的。新增一个控制维度需要修改 Runtime 核心，导致：

- 扩展成本高
- 回归风险大
- 无法独立交付

### 1.2 Phase 7 的答案

引入 **Surface Framework**：

- Surface 是自描述的能力单元
- Runtime 只负责组合和执行
- Surface 通过 Manifest 声明，通过 Registry 加载

---

## 2. Framework Contract

### 2.1 四层架构 [修订]
Surface
│
│ declares
▼
Capability IDs
│
│ referenced by
▼
RuntimeSnapshot
│
│ consumed by
▼
Consumers
(ObservationCompiler / Validator / EditCompiler)


**说明**：
- Surface 声明需要哪些 Capability ID。
- Capability ID 目前只是声明载体，不是独立运行层。
- RuntimeSnapshot 是编译后的执行快照。
- Consumers 只消费 RuntimeSnapshot，不感知具体 Surface 或 Capability 实现。

### 2.2 依赖约束

| 层级 | 允许 | 禁止 |
| :--- | :--- | :--- |
| **Surface** | 声明能力、声明规则、声明修复策略 | 调用 Compiler、修改 Runtime、访问其他 Surface、参与 composition 决策 |
| **Runtime** | Discovery、Aggregation、Snapshot Construction | 包含功能逻辑（Dialogue/Reasoning 等） |
| **Compiler** | 消费 RuntimeSnapshot | 依赖具体 Surface |

### 2.3 关键冻结接口

- `RuntimeBuilder.build()` → `RuntimeSnapshot`
- `ObservationCompiler.compile(draft, snapshot)` → `ObservationIR`
- `Validator.validate(snapshot, ir)` → `ComplianceReport`
- `EditCompiler.compile_with_snapshot(snapshot, report, text, ir)` → `EditPlan`

---

## 3. Surface Contract

### 3.1 Surface 数据结构

```python
SurfaceDefinition:
    metadata:
        id: str
        display_name: str
    observation:
        patterns: List[PatternDefinition]
    validation:
        layer_rules: List[LayerRule]
    repair:
        repair_strategies: List[RepairStrategy]
```

### 3.2 示例：DialogueSurface
DialogueSurface = SurfaceDefinition(
    metadata=SurfaceMetadata(id="dialogue", display_name="对话控制"),
    observation=ObservationSpec(
        patterns=[
            PatternDefinition(
                name="dialogue_marker",
                matcher=Matchers.QUOTATION,
                config={}
            )
        ]
    ),
    validation=ValidationSpec(
        layer_rules=[
            LayerRule(
                layer="dialogue",
                required_types=["dialogue_marker"],
                metrics=[MetricDefinition(name="dialogue_exists", operator="gte", target=1)]
            )
        ]
    ),
    repair=RepairSpec(
        repair_strategies=[
            RepairStrategy(
                target_layer="dialogue",
                trigger="non_compliant",
                operation=Repairs.INSERT_DIALOGUE,
                payload_type="dialogue_marker"
            )
        ]
    )
)

### 3.3 关键约束 [修订]
3.3 关键约束 [修订]

    Surface 只声明，不执行

    Surface 引用 Capability ID，不包含实现

    Surface 不参与 Runtime composition 决策

说明：

    Surface 声明“我是什么”。

    RuntimeConfig 决定“启用什么”。

    RuntimeBuilder 决定“如何组合”。

    RuntimeSnapshot 是“冻结后的事实”。

4. Phase 7B Validation Evidence
4.1 7B-1: Discovery

目标：验证 Surface 能够被自动发现、加载、聚合。

方法：

    PluginLoader 从 __manifest__.py 加载所有 Surface

    SurfaceRegistry 创建不可变目录

    RuntimeBuilder 构建 RuntimeSnapshot

结果：

    DialogueSurface 被成功加载

    RuntimeSnapshot 包含 reasoning 和 dialogue 两个 Surface

    ObservationCompiler 通过遍历 snapshot.surfaces 提取 dialogue_marker

证明：

    ✅ Runtime 不感知具体 Surface

    ✅ 新增 Surface 仅需 Manifest 更新

    ✅ Runtime 核心代码零修改

4.2 7B-2: Validation [修订]

目标：验证 Validator 能从 RuntimeSnapshot 加载 Layer 规则并正确判定合规。

方法：

    Validator 从 snapshot.surfaces 遍历所有 Layer 规则

    对每个 LayerRule，读取 ObservationIR 中对应的 pattern observation

    根据 MetricDefinition 计算 Layer compliance

结果：

    有 Dialogue 的文本：dialogue_exists=True → 合规

    无 Dialogue 的文本：dialogue_exists=False → 不合规

证明：

    ✅ Validator 完全由 Snapshot 驱动

    ✅ 无 layer_targets 依赖

    ✅ 无 Surface 特判

4.3 7B-3: Repair

目标：验证 EditCompiler 能从 RuntimeSnapshot 加载 Repair 策略并生成 EditAction。

方法：

    EditCompiler 从 snapshot.surfaces 遍历所有 Repair 策略

    检查对应层是否不合规

    生成对应的 EditAction

结果：

    无 Dialogue → 生成 INSERT_DIALOGUE Action

    已有 Dialogue → 不生成 Dialogue Action

证明：

    ✅ EditCompiler 完全由 Snapshot 驱动

    ✅ Repair 策略来自 Surface

    ✅ 无 Surface 特判

5. Contract Regression Tests
5.1 测试清单
测试文件	验证内容
test_add_surface_without_runtime_change.py	新增 Surface 不修改 Runtime 核心
test_dialogue_validation.py	Validator 从 Snapshot 加载 Layer 规则
test_dialogue_repair.py	EditCompiler 从 Snapshot 加载 Repair 策略
5.2 核心保证

    ✅ RuntimeBuilder 接口不变

    ✅ RuntimeSnapshot 接口不变

    ✅ ObservationCompiler API 不变

    ✅ Validator API 不变

    ✅ EditCompiler API 不变

    ✅ 新增 Surface 仅需 Manifest 更新

6. Phase 8 Boundary
6.1 Phase 7 完成的验证
能力	状态
Discovery	✅
Aggregation	✅
Observation	✅
Validation	✅
Repair	✅
Runtime 零修改	✅
Framework Contract 冻结	✅
6.2 Phase 8 方向

Phase 7 验证的是：
text

Surface → Capability Identity

Phase 8 将验证：
text

Capability Identity → Capability Semantics

当前 "builtin.quotation" 只是一个 ID。未来需要证明：
python

CapabilitySpec(
    id="builtin.quotation",
    matcher="quotation",
    version="1.0",
    metadata={...}
)

是否需要作为独立架构实体存在。
6.3 Phase 7 不包含的内容

    完整的情感、节奏、记忆等 Surface 实现

    Capability Registry 的 Spec 化

    Capability 版本管理

    跨 Surface 依赖管理

7. 合并前检查
bash

# 1. 运行所有测试
pytest tests/

# 2. 确认 Phase 6 regression 通过
python experiments/phase6/phase6_3c_benchmark.py --mock

# 3. 确认 Phase 7 tests 通过
python experiments/phase7/phase7b1_load.py
python experiments/phase7/phase7b2_validation.py
python experiments/phase7/phase7b3_repair.py
python tests/test_add_surface_without_runtime_change.py

# 4. 提交
git add src/ tests/ experiments/ docs/
git commit -m "Phase 7 complete: Surface Framework with Snapshot-driven Validation and Repair"

8. Phase 7 状态
text

Phase 7 COMPLETE

Architecture:
    Surface-driven Runtime

Verified:
    ✓ Discovery
    ✓ Aggregation
    ✓ Observation
    ✓ Validation
    ✓ Repair
    ✓ Runtime zero-modification

Frozen:
    RuntimeSnapshot as execution boundary
    RuntimeBuilder.build() as composition interface
    Compiler (Observation/Validation/Repair) as Snapshot consumers

Next:
    Phase 8: CapabilitySpec


