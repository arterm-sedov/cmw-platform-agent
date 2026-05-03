"""Pytest: load workspace ``.env`` so tests match app env."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOTENV_PATH = _REPO_ROOT / ".env"


@pytest.fixture(scope="session", autouse=True)
def load_workspace_dotenv() -> None:
    """Load ``.env`` into ``os.environ`` if present (does not override OS env)."""
    if _DOTENV_PATH.is_file():
        load_dotenv(_DOTENV_PATH, override=False)
