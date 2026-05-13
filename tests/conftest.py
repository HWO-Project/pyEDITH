import pytest
from yippy import Coronagraph, fetch_yip


@pytest.fixture(scope="session")
def coronagraph_path():
    """Session-scoped path to a real coronagraph YIP fetched from Zenodo."""
    return fetch_yip("eac1_aavc_2d")


@pytest.fixture(scope="session")
def yippy_coronagraph(coronagraph_path):
    """Session-scoped yippy Coronagraph built from the fetched YIP."""
    return Coronagraph(coronagraph_path)
