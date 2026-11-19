"""Shared test fixtures."""

import pytest

from kipp.embedder import MockEmbedder


@pytest.fixture
def embedder():
    return MockEmbedder(dims=48)
