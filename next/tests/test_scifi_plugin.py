import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from next.plugin.scifi_plugin import SciFiPlugin
from next.kernel.factory import KernelFactory

def test_scifi_plugin():
    plugin = SciFiPlugin()
    kernel = KernelFactory.create(plugin)

    print("科幻题材 Kernel 初始化成功")
    print(f"主角: {list(kernel.entities.keys())}")
    print(f"能力: {list(kernel.capabilities.keys())}")
    print(f"世界规则数量: {len(kernel.metadata.get('world_rules', []))}")
    print(f"主题: {kernel.metadata.get('themes')}")

    assert "凯尔" in kernel.entities
    assert "凯尔|rank" in kernel.capabilities
    assert kernel.capabilities["凯尔|rank"].value == "新兵"

if __name__ == "__main__":
    test_scifi_plugin()