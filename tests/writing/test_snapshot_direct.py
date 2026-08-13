import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 直接从子模块导入，避免 __init__.py 的循环依赖
from src.writing.snapshot.models import PipelineSnapshot, SnapshotIdentity, SnapshotManifest, SnapshotMetadata
from src.writing.snapshot.serializer import JsonSerializer
from src.writing.snapshot.writer import SnapshotWriter
from src.writing.snapshot.loader import SnapshotLoader

def test_imports():
    assert PipelineSnapshot is not None
    assert JsonSerializer is not None
    print("✅ All imports successful")

if __name__ == "__main__":
    test_imports()
