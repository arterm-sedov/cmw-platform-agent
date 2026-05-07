"""Tests for operational UI env flags (download prep timing)."""

import pytest

from agent_ng.agent_config import get_ui_download_prep_after_stream


def _clear_ui_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CMW_UI_DOWNLOAD_PREP_AFTER_STREAM", raising=False)


def test_download_prep_after_stream_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_ui_env(monkeypatch)
    assert get_ui_download_prep_after_stream() is False


def test_download_prep_after_stream_env_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMW_UI_DOWNLOAD_PREP_AFTER_STREAM", "1")
    assert get_ui_download_prep_after_stream() is True
