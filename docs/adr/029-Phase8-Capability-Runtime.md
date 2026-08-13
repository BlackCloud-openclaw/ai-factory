# ADR-029 — Phase 8：Capability Runtime 迁移

> **状态**：已接受  
> **决策日期**：2026-07-17  
> **生效阶段**：Phase 8  
> **取代**：无  
> **被取代**：无  

---

## 背景

Phase 7 引入了 Surface Framework，其中 `PatternDefinition` 使用字符串类型的匹配器（如 `"keyword"`、`"regex"`、`"quotation"`）来声明观测能力。这种方式虽然可用，但存在以下架构限制：

1. **字符串耦合** —— Compiler 依赖硬编码的 `_matcher_registry` 将字符串映射到函数。
2. **无版本管理** —— Capability 仅有字符串名称，缺乏版本标识。
3. **不可扩展** —— 增加新的匹配器需要修改 Runtime 核心代码。
4. **绕过 Loader** —— 兼容逻辑存在于 Runtime 中，而非 Loader。

Phase 8 的目标是将 Capability 从**字符串标识**升级为一等公民的**语义对象**，同时不破坏现有 Surface 定义，也不改变 Runtime 行为。

---

## 决策

### 1. CapabilitySpec 是纯数据（ADR-029）

**CapabilitySpec** 仅包含声明式元数据：

```python
@dataclass(frozen=True)
class CapabilitySpec:
    id: str
    version: Version
    metadata: CapabilityMetadata
    config_schema: Optional[dict] = None
```

**不包含任何可调用对象（Callable）。** 执行逻辑与规格分离，由 `CapabilityImplementation` 承担。

**理由：** 规格应支持序列化、插件加载和版本管理，而不携带执行依赖。

---

### 2. CapabilityRef 作为稳定引用（ADR-033）

`PatternDefinition` 通过 `CapabilityRef` 引用能力：

```python
@dataclass(frozen=True)
class CapabilityRef:
    id: str
    version: Optional[Version] = None
```

这为未来支持版本锁定和可选版本解析预留了空间。

---

### 3. CapabilityLookup 协议（ADR-034）

Runtime 依赖的是**协议（Protocol）**，而非具体的 Registry 实现：

```python
@runtime_checkable
class CapabilityLookup(Protocol):
    def get_spec(self, ref: CapabilityRef) -> CapabilitySpec: ...
    def get_impl(self, ref: CapabilityRef) -> CapabilityImplementation: ...
    def has(self, ref: CapabilityRef) -> bool: ...
```

**理由：** Runtime 永远不应该知道 `CapabilityRegistry` 或任何内置实现。这是依赖倒置原则在 Runtime 架构中的应用。

---

### 4. Registry 不可变且属于 Snapshot（ADR-030、ADR-035）

`CapabilityRegistry` 在 Composition Root 创建，并通过 `MappingProxyType` **冻结**。

`RuntimeSnapshot` 持有 Registry：

```python
@dataclass(frozen=True)
class RuntimeSnapshot:
    ...
    capability_registry: CapabilityLookup
```

**理由：** Runtime 执行上下文完全封装在 `RuntimeSnapshot` 中，没有运行时修改。这也使快照可重现、可测试。

---

### 5. Loader 是唯一的兼容层（ADR-032）

**兼容逻辑位于 `src/surfaces/compatibility.py`，仅由 `PluginLoader` 调用。**

Runtime 核心（`ObservationCompiler`、`Validator`、`EditCompiler`）**永远不需要**看到 `matcher` 字段。Loader 使用 `dataclasses.replace()` 升级旧的 `PatternDefinition` 对象：

```python
def upgrade_pattern(pattern: PatternDefinition) -> PatternDefinition:
    if pattern.capability_ref is not None:
        return pattern
    ref = _matcher_to_ref(pattern.matcher)
    return replace(pattern, capability_ref=ref, matcher=None)
```

**理由：** Loader 是持久化配置（YAML/Surface 模块）与 Runtime 数据结构之间的边界。这样保证了所有进入 Runtime 的 `PatternDefinition` 对象都包含 `capability_ref`。

---

### 6. Capability 实现必须无状态（ADR-037）

所有 `CapabilityImplementation` 实例：

- 没有实例状态
- 可安全地在多线程/多进程间复用
- 不得缓存数据

```python
class KeywordCapability:
    def match(self, text: str, config: dict):
        # 纯函数 — 没有 self.state
        ...
```

**理由：** 无状态实现为未来的插件架构、沙箱化和远程能力执行奠定基础。

---

### 7. Registry 生命周期由 Composition Root 管理（ADR-031、ADR-036）

`CapabilityRegistry` 在应用启动时**创建一次**，并注入 `RuntimeBuilder`：

```python
builder = RuntimeBuilder(catalog, capability_registry)
```

任何 Runtime 组件都不得自行创建 Registry。这保证了所有能力的单一事实来源。

---

### 8. RuntimeSnapshot 不可变（ADR-035）

`RuntimeSnapshot` 一旦构建完成，不可修改。包括：

- `surfaces`
- `capability_registry`
- `config`
- `metrics`

所有字段均为 `frozen=True` dataclass 或 `MappingProxyType`。

**理由：** 不可变性使快照可安全地用于缓存、重放和并行执行。

---

### 9. Compiler 仅依赖协议（ADR-034）

`ObservationCompiler` 现在：

- 不导入 `src.capabilities.builtin.*`
- 不访问 `.matcher` 属性
- 不使用 `_matcher_registry`
- 仅使用 `snapshot.capability_registry.get_impl(ref)`

**架构强制检查：** 架构测试（`tests/architecture/test_runtime_boundaries.py`）使用 AST 解析验证这些约束。

---

## 后果

### 正面

- **Runtime 与具体能力解耦。** 新增能力只需注册新的 `CapabilitySpec` + `CapabilityImplementation`，无需修改 Runtime。
- **版本管理成为可能。** CapabilityRef 支持未来的 `id@version` 语义。
- **Loader 边界清晰。** 兼容逻辑不再泄漏到 Runtime 中。
- **可测试性提升。** Golden Test 验证 Runtime 行为保持不变。

### 负面

- **需要迁移现有 Surface。** `reasoning.py` 和 `dialogue.py` 已更新为使用 `capability_ref`。
- **基础设施增多。** 新增模块：`spec.py`、`reference.py`、`protocol.py`、`registry.py`、`compatibility.py`、`errors.py`。

---

## 验证证据

| 类型 | 文件 |
|------|------|
| 单元测试 | `tests/unit/capabilities/test_registry.py`（16 通过） |
| 单元测试 | `tests/unit/surfaces/test_compatibility.py`（8 通过） |
| 集成测试 | `tests/integration/test_capability_integration.py`（6 通过） |
| 架构测试 | `tests/architecture/test_runtime_boundaries.py`（4 通过） |
| 回归测试 | `tests/regression/test_runtime_behavior.py`（2 通过） |
| 端到端 | 第 35 章生成成功 |
| Smoke 检查 | `_matcher_registry` 已移除；Runtime 不再引用 `matcher` |

### 架构边界已验证

- ✅ Runtime 不导入 `src.capabilities.builtin.*`
- ✅ Runtime 不导入 `CapabilityRegistry`（使用 `CapabilityLookup`）
- ✅ Runtime 不导入 `surfaces.compatibility`
- ✅ Runtime 不访问 `.matcher` 属性

---

## 已知问题（Phase 8 后）

```
WARNING: Runtime workflow failed: cannot import name 'build_default_snapshot'
```

| 字段 | 值 |
|------|-----|
| **范围** | 仅 `RevisionWorkflow` |
| **影响** | 不阻塞主生成流程 |
| **优先级** | 低 |
| **处理方式** | 在 `src/runtime/builder.py` 中添加 `build_default_snapshot` 辅助函数 |
| **不属于 Phase 8 验收范围** | 是 — Phase 8 验证的是 Capability Runtime 迁移，而非 RevisionWorkflow 集成 |

---

## 迁移说明

### 已修改文件

| 文件 | 变更 |
|------|------|
| `src/surfaces/definition.py` | 添加 `capability_ref`，标记 `matcher` 为已废弃 |
| `src/surfaces/compatibility.py` | 新建 — Loader 升级逻辑 |
| `src/surfaces/reasoning.py` | 将 `matcher` 替换为 `capability_ref` |
| `src/surfaces/dialogue.py` | 将 `matcher` 替换为 `capability_ref` |
| `src/runtime/loader.py` | 集成 `upgrade_surfaces()` |
| `src/runtime/observation_compiler.py` | 移除 `_matcher_registry`，使用 `CapabilityLookup` |

### 新增文件

```
src/capabilities/
├── __init__.py
├── spec.py
├── reference.py
├── implementation.py
├── registry.py
├── protocol.py
└── errors.py
```

---

## 参考文档

- **ADR-001**：事件是真相源
- **ADR-002**：谓词是投影缓存
- **ADR-023**：规划契约作为稳定接口
- **ADR-028**：Phase 7 Surface Framework

---

## 标签

`phase8`, `capability`, `runtime`, `migration`, `accepted`, `baseline`

---

*本 ADR 代表 Phase 8 Capability Runtime 迁移基线。所有子决策（029–037）在此合并为统一参考文档。*