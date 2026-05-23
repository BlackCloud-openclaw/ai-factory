import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "characterization: tests that capture current behavior")