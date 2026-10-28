import os
from pathlib import Path

import pytest


def _env_root(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not set")
    return Path(value)


@pytest.fixture
def dataset_root() -> Path:
    """Root of the released dataset tree (PIT_DATASET)."""
    return _env_root("PIT_DATASET")


@pytest.fixture
def sources_root() -> Path:
    """Root of the raw sources tree (PIT_SOURCES)."""
    return _env_root("PIT_SOURCES")
