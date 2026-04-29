"""
tests/conftest.py — Pytest configuration and shared fixtures.
"""
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Ensure project root is importable
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Mock heavy third-party deps so tests run without GPU/network ──
MOCK_MODULES = [
    "httpx", "chromadb", "chromadb.utils",
    "chromadb.utils.embedding_functions",
    "sentence_transformers", "spacy",
    "trafilatura", "feedparser", "datasketch",
    "pystray", "PIL", "PIL.Image", "PIL.ImageDraw",
    "pyttsx3", "whisper", "sounddevice", "soundfile",
    "keyboard", "torch", "transformers", "datasets",
    "peft", "trl", "bitsandbytes", "unsloth",
    "watchdog", "watchdog.observers", "watchdog.events",
    "langdetect", "requests", "uvicorn",
    "fastapi", "fastapi.responses", "fastapi.staticfiles",
    "pydantic", "aiofiles", "click", "rich",
    "rich.console", "rich.table",
]

for mod in MOCK_MODULES:
    if mod not in sys.modules:
        sys.modules[mod] = mock.MagicMock()

# Mock httpx.AsyncClient to return sensible defaults
httpx_mock = sys.modules["httpx"]
httpx_mock.AsyncClient.return_value.__aenter__ = mock.AsyncMock(
    return_value=mock.MagicMock(
        post=mock.AsyncMock(return_value=mock.MagicMock(
            json=mock.MagicMock(return_value={"response": "mocked answer"})
        )),
        get=mock.AsyncMock(return_value=mock.MagicMock(
            status_code=200, text="<html>mocked page</html>",
            json=mock.MagicMock(return_value={"results": []})
        )),
    )
)
httpx_mock.AsyncClient.return_value.__aexit__ = mock.AsyncMock(return_value=False)
httpx_mock.ConnectError = ConnectionError
httpx_mock.get = mock.MagicMock(return_value=mock.MagicMock(
    json=mock.MagicMock(return_value={}),
    raise_for_status=mock.MagicMock(),
))
httpx_mock.post = mock.MagicMock(return_value=mock.MagicMock(
    json=mock.MagicMock(return_value={}),
    raise_for_status=mock.MagicMock(),
))


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Fixture: redirect context DB to a temp path."""
    import core.context as ctx_mod
    monkeypatch.setattr(ctx_mod, "DB_PATH", tmp_path / "test_context.sqlite")
    return tmp_path


@pytest.fixture
def fresh_context(tmp_db):
    """Fixture: a fresh ContextMemory backed by a temp SQLite file."""
    from core.context import ContextMemory
    return ContextMemory()


@pytest.fixture
def router():
    """Fixture: a ModelRouter instance."""
    from core.model_router import ModelRouter
    return ModelRouter()


@pytest.fixture
def orchestrator(fresh_context, router):
    """Fixture: a minimal Orchestrator with mocked modules."""
    from core.orchestrator import Orchestrator
    mock_module = mock.MagicMock()
    mock_module.health.return_value = {
        "name": "knowledge", "stage": "bootstrap",
        "maturity_score": 0.0, "query_count": 0,
        "db_ok": True, "kb_chunks": 0,
    }
    modules = {"knowledge": mock_module}
    return Orchestrator(modules, fresh_context, router)
