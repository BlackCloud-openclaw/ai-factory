import sys
from unittest.mock import MagicMock

# 在导入任何代码前 mock 掉 nodes，避免循环导入
sys.modules['src.orchestrator.nodes'] = MagicMock()