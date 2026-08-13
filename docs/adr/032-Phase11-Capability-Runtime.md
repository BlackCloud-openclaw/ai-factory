# ADR-032: Phase 11 — Capability Runtime 迁移

> **状态**：已接受  
> **日期**：2026-07-22  
> **决策者**：Phase 11 团队  
> **影响范围**：`src/capabilities/runtime/` · `src/writing/bootstrap/` · `src/writing/controlled_writer.py` · `src/orchestrator/`

---

## 一、背景

Phase 7 引入了 Surface Framework，Phase 8 将 Capability 从字符串标识升级为 `CapabilitySpec` + `CapabilityImplementation` 双层架构。然而，Runtime 能力（如 Audit、Snapshot）与 Surface Capability（如 `keyword`、`quotation`）仍存在以下问题：

1. **Runtime 能力未纳入 Capability 体系**：`AuditCoordinator`、`ChunkRepository`、`VersionStore`、`IncrementalTransport` 等 Runtime 服务散落在各处，业务层通过直接实例化或模块级导入获取，破坏了依赖倒置原则。

2. **业务层与 Runtime 实现耦合**：`ControlledWriter` 通过裸实例化 `ControlledWriter()` 获取，无法注入不同的 Runtime 配置，测试和扩展困难。

3. **缺少统一的 Runtime 服务入口**：没有 `RuntimeServices` 这样的聚合对象，业务层需要分别导入 `AuditCoordinator`、`SnapshotManager` 等，增加了认知负担和耦合。

4. **命名冲突**：`src/writing/snapshot.py`（旧版 `SnapshotManager`）与 `src/writing/snapshot/`（新版 Snapshot Runtime 包）同名，导致导入歧义和循环引用。

Phase 11 的目标是将 Runtime 能力纳入 Capability 体系，建立 `RuntimeCapabilityRegistry` → `RuntimeServices` → `ControlledWriter` 的注入链，实现业务层与 Runtime 实现的解耦。

---

## 二、决策

### 1. 双层 Capability 架构（ADR-029 延续）

**Runtime Capability** 与 **Surface Capability** 分离：

| 类型 | 接口 | 职责 | 示例 |
|------|------|------|------|
| Surface Capability | `CapabilityImplementation.match()` | 文本匹配 | `keyword`, `quotation` |
| Runtime Capability | `RuntimeCapability.get()` | 返回运行时服务对象 | `AuditCoordinator`, `ChunkRepository` |

**理由**：两者调用模式不同，强行统一会导致接口污染。Runtime Capability 不需要 `match` 方法，只需要提供服务实例。

### 2. RuntimeCapabilityRegistry（ADR-052/053 冻结）

```python
class RuntimeCapabilityRegistry:
    def register(spec: CapabilitySpec, capability: RuntimeCapability) -> None
    def require(capability_id: str) -> RuntimeCapability
    def freeze() -> FrozenRuntimeCapabilityRegistry
```

**理由**：与 Surface Capability 的注册机制对齐，但独立管理 Runtime 能力。冻结后不可变，保证 Runtime 环境的确定性。

### 3. RuntimeServices 聚合层

```python
@dataclass(frozen=True)
class RuntimeServices:
    def audit(self) -> AuditService
    # 未来扩展: def snapshot(self) -> SnapshotService
```

**理由**：业务层不直接操作 `RuntimeCapabilityRegistry`，而是通过 `RuntimeServices` 获取能力。这提供了类型安全的访问接口，降低耦合。

### 4. WriterRuntime 组合根

```python
@dataclass(frozen=True)
class WriterRuntime:
    runtime_capabilities: FrozenRuntimeCapabilityRegistry
    runtime_services: RuntimeServices

def build_writer_runtime() -> WriterRuntime
```

**理由**：`composition_root.py` 是唯一的 Runtime 组装入口，`WriterRuntime` 持有所有 Runtime 对象的生命周期。业务层通过注入 `WriterRuntime` 获取所需服务。

### 5. ControlledWriter 依赖注入

```python
class ControlledWriter:
    def __init__(self, runtime_services: Optional[RuntimeServices] = None):
        self._runtime_services = runtime_services
```

**理由**：`ControlledWriter` 只依赖 `RuntimeServices` 协议，不关心其来源。`runtime_services=None` 时保持向后兼容，使迁移平滑。

### 6. Orchestrator 注入 Runtime

```python
async def writer_node(state: AgentState, runtime: WriterRuntime) -> dict:
    cw = ControlledWriter(runtime_services=runtime.runtime_services)
```

通过 `partial(writer_node, runtime=runtime)` 绑定，不引入全局状态。

**理由**：LangGraph 节点函数签名固定为 `(state) -> dict`，通过 `partial` 在 `create_workflow` 阶段注入依赖，保持节点函数的纯净性。

### 7. Snapshot 命名冲突修复

将 `src/writing/snapshot.py` 重命名为 `src/writing/snapshot_manager.py`，消除与 `src/writing/snapshot/` 包的命名冲突。

**理由**：Python 导入解析在模块和包同名时会产生歧义，导致循环引用。重命名后语义清晰：`snapshot_manager` 为旧版快照管理，`snapshot` 为新版 Runtime 包。

---

## 三、架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Application Bootstrap                            │
│                                                                             │
│   src/writing/bootstrap/composition_root.py                                 │
│                                                                             │
│   build_writer_runtime()                                                    │
│         │                                                                   │
│         ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        WriterRuntime                                │   │
│   │                                                                      │   │
│   │   runtime_capabilities: FrozenRuntimeCapabilityRegistry             │   │
│   │   runtime_services: RuntimeServices                                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Orchestrator Layer                                 │
│                                                                             │
│   src/orchestrator/graph.py                                                 │
│   workflow.add_node("writer", partial(writer_node, runtime=runtime))        │
│                                                                             │
│   src/orchestrator/nodes.py                                                 │
│   async def writer_node(state, runtime):                                   │
│       cw = ControlledWriter(runtime_services=runtime.runtime_services)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Business Layer                                    │
│                                                                             │
│   src/writing/controlled_writer.py                                          │
│   class ControlledWriter:                                                   │
│       def __init__(self, runtime_services: Optional[RuntimeServices]):     │
│           self._runtime_services = runtime_services                        │
│                                                                             │
│   # 未来扩展: 在 execute() 中使用 self._runtime_services.audit()           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Capability Layer                                     │
│                                                                             │
│   src/capabilities/runtime/                                                 │
│   ├── protocol.py      # RuntimeCapability Protocol                        │
│   ├── registry.py      # RuntimeCapabilityRegistry                         │
│   └── frozen.py        # FrozenRuntimeCapabilityRegistry                   │
│                                                                             │
│   src/capabilities/audit/   # AuditCapability                              │
│   src/capabilities/runtime/snapshot/  # Snapshot Capabilities              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Runtime Services                                     │
│                                                                             │
│   src/writing/runtime/services.py                                           │
│   class RuntimeServices:                                                    │
│       def audit(self) -> AuditService                                      │
│                                                                             │
│   src/writing/runtime/protocols.py                                          │
│   class AuditService(Protocol): ...                                        │
│                                                                             │
│   # 具体实现: AuditCoordinator, ChunkRepository, VersionStore, Transport   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 四、后果

### 正面

- ✅ Runtime 能力（Audit、Snapshot）纳入 Capability 体系，统一管理
- ✅ 业务层与 Runtime 实现解耦，通过 `RuntimeServices` 协议访问
- ✅ `ControlledWriter` 支持依赖注入，测试时可注入 Mock Runtime
- ✅ `WriterRuntime` 作为唯一的 Runtime 生命周期入口，消除隐式全局状态
- ✅ 命名冲突修复，`snapshot_manager` 与 `snapshot` 包职责清晰
- ✅ 为 Phase 12（写作质量评估）奠定可观测性基础

### 负面/限制

- ⚠️ `ControlledWriter.execute()` 当前尚未消费 `self._runtime_services`，仅保留入口供未来扩展
- ⚠️ 旧版 `SnapshotManager` 仍在使用（`snapshot_manager.py`），未来需逐步迁移到新 Snapshot Runtime
- ⚠️ `RuntimeServices` 当前仅提供 `audit()` 服务，Snapshot 服务未暴露（由 Composition Root 内部持有）

---

## 五、验证证据

### 单元测试

| 测试套件 | 数量 | 状态 |
|----------|------|------|
| `tests/unit/capabilities/runtime/` | 16 | ✅ 通过 |
| `tests/unit/writing/bootstrap/` | 8 | ✅ 通过 |
| `tests/architecture/test_runtime_injection.py` | 4 | ✅ 通过 |

### 集成测试

| 测试 | 内容 | 状态 |
|------|------|------|
| 服务启动 | `python -m src.api.main` | ✅ 无导入错误 |
| 长篇小说生成 | `python tests/simple_long_novel.py` | ✅ 500 章大纲完整，续写启动成功 |

### 迁移检查清单

| 检查项 | 状态 |
|--------|------|
| `ControlledWriter` 支持 `runtime_services` 参数 | ✅ |
| `composition_root.py` 提供 `build_writer_runtime()` | ✅ |
| `graph.py` 使用 `partial` 绑定 `writer_node` 与 `runtime` | ✅ |
| `writer_node` 无内部 `build_writer_runtime()` fallback | ✅ |
| `snapshot.py` → `snapshot_manager.py` 重命名完成 | ✅ |
| 所有旧导入迁移到 `snapshot_manager` | ✅ |
| `snapshot/__init__.py` 无循环导入 | ✅ |

---

## 六、相关 ADR

| ADR | 关系 |
|-----|------|
| ADR-023 | Planning Contract 稳定接口（Writer 消费 Contract） |
| ADR-024 | Incremental Execution（ControlledWriter 执行增量写入） |
| ADR-027 | Phase 6 Runtime（Runtime 纯计算化） |
| ADR-028 | Phase 7 Surface Framework（Surface Capability 声明） |
| ADR-029 | Phase 8 Capability Runtime（CapabilitySpec 双层架构） |
| ADR-031 | Phase 10 Audit & Snapshot Runtime（Trace/Report 基础设施） |

---

## 七、后续演进（Phase 12+）

| Phase | 内容 | 目标 |
|-------|------|------|
| Phase 12 | Factory 2.0 写作质量评估 | 验证架构升级对章节割裂感的改善 |
| Phase 12+ | `ControlledWriter.execute()` 消费 `RuntimeServices` | 启用内部审计能力 |
| Phase 13 | 旧版 `SnapshotManager` 废弃 | 迁移到新 Snapshot Runtime |

---

**本 ADR 记录了 Phase 11 的全部架构决策、实施细节及验证状态，标志着 AI Factory Runtime 能力从分散实现演进为统一的 Capability 驱动架构。**