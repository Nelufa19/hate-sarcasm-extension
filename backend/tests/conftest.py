"""
Session-scoped TestClient fixture. FastAPI's lifespan (which calls
registry.load_all()) runs once when the client is created and stays
resident for every test in the session -- loading two DistilBERT
checkpoints per test would make the suite unusably slow.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
