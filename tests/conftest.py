"""Shared test fixtures."""

import pytest

import kipp.cli as cli_mod
from kipp.client import ALL_COLLECTIONS
from kipp.embedder import MockEmbedder


@pytest.fixture
def embedder():
    return MockEmbedder(dims=48)


@pytest.fixture
def cli_env(embedder, monkeypatch):
    """In-memory Qdrant + mock embedder wired into the cli module.

    Collections are created at the mock embedder's dimension (not via
    ensure_collections, which hardcodes the 384-dim real model size).
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    dim = len(embedder.embed("dimension probe"))
    client = QdrantClient(":memory:")
    for name in ALL_COLLECTIONS:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    monkeypatch.setattr(cli_mod, "get_client", lambda *a, **k: client)
    monkeypatch.setattr(cli_mod, "_get_embedder", lambda *a, **k: embedder)
    return client
