"""
Session-scoped TestClient fixture. FastAPI's lifespan (which calls
registry.load_all()) runs once when the client is created and stays
resident for every test in the session -- loading two DistilBERT
checkpoints per test would make the suite unusably slow.
"""
import os
import tempfile
from pathlib import Path

import pytest

# Must be set BEFORE app.config is imported (it reads this at module level).
# Without it the suite would append to -- and permanently pollute -- the
# developer's real backend/data/feedback.jsonl.
os.environ["FEEDBACK_PATH"] = str(Path(tempfile.mkdtemp(prefix="hs-feedback-")) / "feedback.jsonl")

from fastapi.testclient import TestClient  # noqa: E402

from app.feedback import store as feedback_store  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def clean_feedback(client):
    """
    Empty the taught-example store around a test. The store is process-global
    (it has to be -- it's an inference-path cache), so without this a test that
    teaches an example would silently change the verdicts every later test
    sees.
    """
    def _clear():
        for entry in feedback_store.all_entries():
            feedback_store.remove(entry.text, entry.language)

    _clear()
    yield feedback_store
    _clear()
