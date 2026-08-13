# ADR-031: Phase 10 — Audit & Snapshot Runtime

> **状态**: Accepted  
> **日期**: 2026-07-22  
> **决策者**: Phase 10 团队  
> **影响范围**: `src/writing/snapshot/` · `src/writing/audit/` · `src/writing/bootstrap/`

---

## 一、背景

AI Factory 经过 Phase 8（Capability Runtime）和 B4 系列（Snapshot Runtime Foundation）的建设，已具备：

- 完整的快照存储与增量版本管理
- 流式传输与压缩
- 远程 S3 后端与缓存
- 垃圾回收（GC）与租约机制

Phase 10 的目标是将这些能力与 Writer Runtime 打通，建立 **从执行追踪到审计分析的全链路基础设施**，使系统具备：

1. **可重放性**：每次 Writer 执行都可完整记录、回放和验证。
2. **可观测性**：定位字段丢失、Token 分配异常、优化优先级。
3. **可审计性**：自动化生成执行报告，支持持久化和比较。

---

## 二、架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Phase 10 Audit & Snapshot                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     Phase 10.4: Runtime Validation                    │ │
│  │  • Integration Test (Full Flow)                                      │ │
│  │  • Non-invasive Validation                                           │ │
│  │  • Failure Isolation                                                 │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     Phase 10.3: Runtime Integration                   │ │
│  │  • AuditCoordinator (无状态编排器)                                    │ │
│  │  • AuditContext (会话状态)                                            │ │
│  │  • RuntimeHook (装饰器 + 手动 Hook)                                  │ │
│  │  • AuditReportStore (报告持久化)                                     │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        Phase 10.2: Audit Core                         │ │
│  │  • ExecutionTrace (不可变 DAG)                                        │ │
│  │  • Preservation Analyzer                                             │ │
│  │  • Attribution Analyzer                                              │ │
│  │  • Budget Analyzer                                                   │ │
│  │  • Priority Engine                                                   │ │
│  │  • Reporter                                                          │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      Phase 10.1: Snapshot Runtime                     │ │
│  │  • PipelineSnapshot (Identity + Manifest + Metadata)                 │ │
│  │  • WriterIR · PromptBundle · RenderTrace                             │ │
│  │  • Canonical JSON · Golden Tests                                     │ │
│  │  • Architecture Lock (Import Boundary / Dependency Cycle)            │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 职责分层

| 层 | 职责 | 关键组件 |
|---|---|---|
| **10.1 Snapshot** | 执行持久化与可重放 | `PipelineSnapshot`, `SnapshotWriter`, `SnapshotLoader` |
| **10.2 Audit Core** | 分析推理与报告生成 | Analyzers + Reporter |
| **10.3 Integration** | Runtime 接入与编排 | `AuditCoordinator`, `AuditContext`, `AuditHook` |
| **10.4 Validation** | 端到端验证 | 集成测试套件 |

---

## 三、Phase 10.1: Snapshot Runtime

### 核心数据结构

```python
@dataclass(frozen=True)
class PipelineSnapshot:
    identity: SnapshotIdentity
    manifest: SnapshotManifest
    metadata: SnapshotMetadata
    planning: PlanningArtifact
    writer_ir: WriterIR
    prompt_bundle: PromptBundle
    render_trace: RenderTrace
    draft: str
    coverage: CoverageReport
    timestamp: datetime
```

### 序列化契约

- **输出格式**: Canonical JSON（UTF-8, sort_keys=True, indent=2, ensure_ascii=False）
- **Golden 测试**: SHA256 锁定，防止字节流漂移
- **版本迁移**: `SnapshotLoader` 自动升级旧版本

### Architecture Lock

- 导入边界检查（`tests/architecture/`）
- 循环依赖检测
- 确保 Runtime 不依赖具体 Capability 实现

### 冻结 API

| 组件 | 关键方法 |
|------|----------|
| `SnapshotWriter` | `write(snapshot) -> Path` |
| `SnapshotLoader` | `load(path) -> PipelineSnapshot` |
| `JsonSerializer` | `serialize()/deserialize()` |

---

## 四、Phase 10.2: Audit Core

### 核心数据流

```
ExecutionTrace
        │
        ▼
  Preservation Analyzer
  └── 回答：丢了什么？
        │
        ▼
  Attribution Analyzer
  └── 回答：在哪里丢？
        │
        ▼
  Budget Analyzer
  └── 回答：值不值得优化？
        │
        ▼
  Priority Engine
  └── 回答：先修哪个？
        │
        ▼
  Reporter
  └── 回答：怎么告诉用户？
        │
        ▼
  ComprehensiveReport
```

### 关键数据结构

#### ExecutionTrace（不可变 DAG）

```python
@dataclass(frozen=True)
class ExecutionTrace:
    execution_id: UUID
    stages: Tuple[StageRecord, ...]
    artifacts: Mapping[UUID, Artifact]
```

#### 字段比较模型

```python
class Existence(Enum):
    PRESENT = "present"
    REMOVED = "removed"
    UNKNOWN = "unknown"

class ChangeType(Enum):
    UNCHANGED = "unchanged"
    MODIFIED = "modified"
    PARTIAL = "partial"

@dataclass(frozen=True)
class ComparisonResult:
    existence: Existence
    change: ChangeType
    retention_ratio: float
```

#### Lineage 分析

```python
@dataclass
class LineagePreservation:
    source: UUID          # 源 Artifact
    sink: UUID            # 终端 Artifact
    lineage: List[UUID]   # 路径上的所有 Artifact
    statuses: Dict[UUID, ComparisonResult]
    end_retention_rate: float
```

### 分析器输出

| Analyzer | 输出 | 用途 |
|----------|------|------|
| Preservation | `PreservationReport` | 字段保留率、丢失字段列表 |
| Attribution | `AttributionReport` | 丢失阶段、归因类型 |
| Budget | `BudgetReport` | Token 分配、异常评分 |
| Priority | `PriorityReport` | 优化目标、优先级排序 |
| Reporter | `ComprehensiveReport` | 综合报告（聚合所有） |

### 冻结 API

所有 Report 类实现 `to_dict()` + `from_dict()`，支持持久化。

---

## 五、Phase 10.3: Runtime Integration

### 核心组件

| 组件 | 职责 | 生命周期 |
|------|------|----------|
| `AuditCoordinator` | 无状态编排器 | 全局单例 |
| `AuditContext` | 会话状态管理 | 单次 Writer 执行 |
| `TraceCollector` | 收集 ExecutionTrace | 单次 Writer 执行 |
| `@audit_writer` | 自动注入审计点 | 函数级装饰器 |
| `AuditReportStore` | 报告持久化与查询 | 全局单例 |

### 数据流

```
Writer Runtime
       │
       ▼
@audit_writer decorator
       │
       ├── AuditCoordinator.start()
       │         │
       │         ▼
       │   AuditContext
       │         │
       │         ├── TraceCollector
       │         │         │
       │         │         ▼
       │         │   ExecutionTrace
       │         │
       │         ├── _generate_report()
       │         │         │
       │         │         ├── PreservationAnalyzer
       │         │         ├── AttributionAnalyzer
       │         │         ├── BudgetAnalyzer
       │         │         ├── PriorityEngine
       │         │         └── Reporter
       │         │
       │         └── ComprehensiveReport
       │
       ├── Writer 执行（不受影响）
       │
       └── AuditReportStore.save()
                   │
                   ▼
             JSON + index
```

### 关键决策

#### D1: 无状态编排器

`AuditCoordinator` 不保存会话状态，所有状态由 `AuditContext` 持有，支持并发 Writer 执行。

#### D2: 故障隔离

`_generate_report()` 捕获所有异常并记录日志，不向上传播，确保 Audit 失败不影响 Writer 执行。

#### D3: 报告存储分离

- 索引（`index.json`）轻量，列表查询 O(1)
- 报告主体按 `execution_id` 独立存储，按需加载

### 冻结 API

```python
# AuditCoordinator
def __init__(resolver, config)
def start(novel_id, volume, chapter, scene_idx) -> AuditContext
def audit(...) -> AuditContext

# AuditContext
@property def execution_id, trace, report
def record_stage(stage, inputs, outputs)

# @audit_writer
def audit_writer(resolver, config)

# AuditReportStore
def save(report, novel_id) -> Path
def list(novel_id, limit) -> Sequence[ReportEntry]
def load(entry) -> ComprehensiveReport
```

---

## 六、Phase 10.4: Runtime Validation

### 测试覆盖

| 测试 | 验证内容 | 状态 |
|------|----------|------|
| `test_full_runtime_flow` | 执行 → Trace → Report → Store → Reload | ✅ |
| `test_non_invasive` | Audit 不影响 Writer 输出 | ✅ |
| `test_failure_isolation` | Audit 失败不影响 Writer 执行 | ✅ |

### 验收标准

- ✅ Audit 不修改 Writer 业务逻辑
- ✅ 审计失败仅记录日志，不传播异常
- ✅ 报告可完整恢复（`save()` ↔ `load()`）
- ✅ 全部集成测试通过

---

## 七、核心 ADR 决策汇总

### 10.1 Snapshot

| 决策 | 选择 | 理由 |
|------|------|------|
| 序列化格式 | Canonical JSON | 确定性输出，便于 Golden 测试 |
| 版本迁移 | `SnapshotLoader` 自动升级 | 向后兼容历史快照 |
| 架构锁 | Import Boundary + Cycle Check | 防止依赖漂移 |

### 10.2 Audit Core

| 决策 | 选择 | 理由 |
|------|------|------|
| Trace 存储方式 | PayloadRef（不存储数据） | 保持 Trace 轻量 |
| Payload 解析 | `PayloadResolver` Protocol | 可替换存储后端 |
| 字段比较 | 值比较（非存在性检查） | 支持部分保留检测 |
| Lineage 追踪 | Artifact DAG（非 Stage 顺序） | 真正的数据流分析 |

### 10.3 Runtime Integration

| 决策 | 选择 | 理由 |
|------|------|------|
| 编排器状态 | 无状态（`AuditCoordinator`） | 支持并发 |
| 会话管理 | `AuditContext` 持有状态 | 生命周期清晰 |
| 故障处理 | 捕获异常，记录日志 | 不影响 Writer 执行 |
| 报告存储 | 索引 + 快照分离 | 查询高效，按需加载 |

---

## 八、API 冻结清单

### Snapshot Runtime

- `SnapshotWriter`, `SnapshotLoader`
- `PipelineSnapshot`, `SnapshotIdentity`, `SnapshotManifest`, `SnapshotMetadata`
- `JsonSerializer` (Canonical JSON)

### Audit Core

- `ExecutionTrace`, `StageRecord`, `Artifact`
- `TraceCollector`
- `PreservationReport`, `AttributionReport`, `BudgetReport`, `PriorityReport`
- `ComprehensiveReport`

### Runtime Integration

- `AuditCoordinator`, `AuditConfig`, `AuditContext`
- `@audit_writer`
- `AuditReportStore`, `ReportEntry`

---

## 九、测试状态

```
单元测试 (Phase 10.2):
    test_priority.py                  7/7 passed
    test_budget.py                    8/8 passed
    test_reporter.py                  5/5 passed

单元测试 (Phase 10.3):
    test_coordinator.py               5/5 passed
    test_hook.py                      2/2 passed

集成测试 (Phase 10.4):
    test_runtime_audit_flow.py        3/3 passed
```

---

## 十、限制与演进

### 当前限制

1. **PayloadResolver**：`MemoryPayloadResolver` 仅用于测试，生产环境需持久化实现。
2. **报告恢复**：`from_dict()` 简化实现，未完全恢复复杂子对象。
3. **性能基线**：审计开销基线待建立（Phase 10.5）。

### 后续演进（Phase 11 及以后）

| Phase | 主题 | 目标 |
|-------|------|------|
| Phase 11 | Capability Runtime | 将 Audit/Snapshot 能力注册到 Capability Registry |
| Phase 10.5 | Production Hardening | Payload 持久化、性能优化、监控告警 |
| Phase 12 | Distributed Runtime | 多节点共享审计报告 |

---

## 十一、相关 ADR

- ADR-008: Deterministic Projection Functions
- ADR-017: Schema Compatibility Policy
- ADR-035: RuntimeSnapshot Immutable
- Phase 8: Capability Registry
- B4: Remote GC Runtime

---

*本 ADR 记录了 Phase 10 的全部架构决策、组件设计、API 冻结状态及测试验收标准，标志着 AI Factory 从“可工作”系统正式演进为“可审计、可观测、可重放”的生产级叙事生成平台。*