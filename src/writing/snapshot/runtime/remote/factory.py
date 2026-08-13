# src/writing/snapshot/runtime/remote/factory.py
"""
B4.4: Remote 存储组装工厂
"""

import os
from typing import Optional

from ..id import SnapshotId
from ..incremental import IncrementalTransport
from ..serializers import SerializerRegistry, SerializerResolver
from ..compression import CompressionRegistry, CompressionResolver
from ..chunking import ChunkingStrategy, FixedChunkStrategy
from .s3 import S3Config, S3Client, S3ChunkStore, S3VersionStore, S3KeyLayout
from .repository import RemoteChunkRepository
from .cache import CachedChunkRepository
from .errors import RemoteStoreError


def create_default_registry() -> tuple[SerializerRegistry, CompressionRegistry]:
    """创建默认的 Serializer 和 Compression Registry（Composition Root 辅助）。"""
    return SerializerRegistry.with_builtin(), CompressionRegistry.with_builtin()


def create_remote_transport(
    s3_config: S3Config,
    *,
    serializer_registry: SerializerRegistry,
    compression_registry: CompressionRegistry,
    chunking_strategy: Optional[ChunkingStrategy] = None,
    use_cache: bool = True,
    max_cache_entries: int = 128,
    default_serializer_id: str = "builtin.json",
    default_codec_id: str = "builtin.identity",
    max_chain_depth: int = 32,
) -> IncrementalTransport:
    """
    创建基于 S3 的远程存储 Transport。

    Args:
        s3_config: S3 配置
        serializer_registry: 序列化器注册表（必须由调用方注入）
        compression_registry: 压缩注册表（必须由调用方注入）
        chunking_strategy: 分块策略（默认 FixedChunkStrategy 1MB）
        use_cache: 是否启用缓存
        max_cache_entries: 缓存最大条目数
        default_serializer_id: 默认序列化器 ID
        default_codec_id: 默认压缩算法 ID
        max_chain_depth: 版本链最大深度

    Returns:
        配置好的 IncrementalTransport
    """
    # 1. 创建 S3 客户端
    s3_client = S3Client(s3_config)

    # 2. 创建共享的 KeyLayout
    layout = S3KeyLayout(s3_config.prefix)

    # 3. 创建 S3 存储组件（共享同一个 layout）
    chunk_store = S3ChunkStore(s3_client, layout)
    version_store = S3VersionStore(s3_client, layout)

    # 4. 创建 RemoteChunkRepository
    remote_repo = RemoteChunkRepository(chunk_store, version_store)

    # 5. 可选缓存装饰
    if use_cache:
        remote_repo = CachedChunkRepository(remote_repo, max_entries=max_cache_entries)

    # 6. 创建 Resolver（从 Registry 获取）
    serializer_resolver: SerializerResolver = serializer_registry
    compression_resolver: CompressionResolver = compression_registry

    if chunking_strategy is None:
        chunking_strategy = FixedChunkStrategy(1024 * 1024)  # 1MB

    # 7. 创建 IncrementalTransport
    transport = IncrementalTransport(
        repository=remote_repo,
        serializer_resolver=serializer_resolver,
        compression_resolver=compression_resolver,
        strategy=chunking_strategy,
        default_serializer_id=default_serializer_id,
        default_codec_id=default_codec_id,
        max_chain_depth=max_chain_depth,
    )

    return transport


def create_s3_transport_from_env(
    *,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    region: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    serializer_registry: Optional[SerializerRegistry] = None,
    compression_registry: Optional[CompressionRegistry] = None,
    **kwargs,
) -> IncrementalTransport:
    """
    从环境变量或显式参数创建 S3 Transport。

    环境变量优先级（从高到低）：
        1. 显式参数
        2. AWS 标准环境变量 (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
        3. S3_* 自定义环境变量 (S3_BUCKET, S3_PREFIX, S3_REGION, S3_ENDPOINT_URL)
    """
    bucket = bucket or os.environ.get("S3_BUCKET")
    if not bucket:
        raise ValueError("S3 bucket must be provided via argument or S3_BUCKET env var")

    prefix = prefix or os.environ.get("S3_PREFIX", "snapshots/")
    region = region or os.environ.get("AWS_REGION") or os.environ.get("S3_REGION")
    endpoint_url = endpoint_url or os.environ.get("S3_ENDPOINT_URL")
    access_key = access_key or os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("S3_ACCESS_KEY")
    secret_key = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("S3_SECRET_KEY")

    config = S3Config(
        bucket=bucket,
        prefix=prefix,
        region=region,
        endpoint_url=endpoint_url,
        access_key=access_key,
        secret_key=secret_key,
    )

    # 如果未提供 Registry，使用默认
    if serializer_registry is None or compression_registry is None:
        serializer_registry, compression_registry = create_default_registry()

    return create_remote_transport(
        config,
        serializer_registry=serializer_registry,
        compression_registry=compression_registry,
        **kwargs,
    )