# src/bootstrap/snapshot.py
"""
B2: Snapshot Composition Root

职责：组装 Snapshot 加载链，不包含任何迁移定义或业务逻辑。
Bootstrap 仅调用注册函数，不直接知道 MigrationEdge / Upcaster / VersionNode。
"""

from src.writing.snapshot import SnapshotLoader
from src.writing.snapshot.migration import (
    MigrationRegistry,
    MigrationRuntime,
    StaticSchemaProvider,
    DefaultMigrationContextFactory,
    CURRENT_SCHEMA_VERSION,
    register_builtin_migrations,
)

from .services import SnapshotServices


def configure_snapshot() -> SnapshotServices:
    """
    Composition Root 入口：构建完整的 Snapshot 加载器。

    流程：
    1. 创建 Registry
    2. 注册所有内置迁移（Bootstrap 不知道具体内容）
    3. 构建 MigrationGraph → MigrationRuntime
    4. 注入依赖 → 返回 SnapshotServices
    """
    # 1. 创建 Registry
    registry = MigrationRegistry()

    # 2. 注册所有内置迁移
    register_builtin_migrations(registry)

    # 3. 构建 Graph → Runtime
    graph = registry.build()
    runtime = MigrationRuntime(graph)

    # 4. 组装 Loader
    schema_provider = StaticSchemaProvider(CURRENT_SCHEMA_VERSION)
    context_factory = DefaultMigrationContextFactory()

    loader = SnapshotLoader(
        migrator=runtime,
        schema_provider=schema_provider,
        context_factory=context_factory,
    )

    return SnapshotServices(
        loader=loader,
        runtime=runtime,
        registry=registry,
    )