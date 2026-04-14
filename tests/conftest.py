import pytest
from yippy import Coronagraph
from yippy.datasets import fetch_coronagraph


@pytest.fixture(scope="session")
def coronagraph_path():
    """Session-scoped path to a real coronagraph YIP fetched via pooch."""
    return fetch_coronagraph()


@pytest.fixture(scope="session")
def yippy_coronagraph(coronagraph_path):
    """Session-scoped yippy Coronagraph built from the fetched YIP."""
    return Coronagraph(coronagraph_path)
