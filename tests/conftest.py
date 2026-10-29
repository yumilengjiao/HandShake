import pytest

from backend.mock import run_synthetic
from shared.config import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def smoke_records(config):
    return run_synthetic(config, code_version="test-only")
