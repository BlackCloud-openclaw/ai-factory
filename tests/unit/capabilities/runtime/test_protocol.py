# tests/unit/capabilities/runtime/test_protocol.py

import pytest
from src.capabilities.runtime import RuntimeCapability


def test_protocol_accepts_get_method():
    """验证 RuntimeCapability Protocol 接受有 get() 的类。"""

    class ValidCapability:
        def get(self):
            return "service"

    cap = ValidCapability()
    assert isinstance(cap, RuntimeCapability)


def test_protocol_rejects_missing_get():
    """验证 RuntimeCapability Protocol 拒绝没有 get() 的类。"""

    class InvalidCapability:
        def something_else(self):
            pass

    cap = InvalidCapability()
    assert not isinstance(cap, RuntimeCapability)