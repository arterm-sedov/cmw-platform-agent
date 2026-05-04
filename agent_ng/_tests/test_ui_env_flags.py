"""Tests for operational UI env flags (home tab order, download prep timing)."""

import pytest

from agent_ng.agent_config import (
    get_ui_download_prep_after_stream,
    get_ui_home_first,
)


def _clear_ui_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "CMW_UI_HOME_FIRST",
        "CMW_UI_DOWNLOAD_PREP_AFTER_STREAM",
    ):
        monkeypatch.delenv(key, raising=False)


def test_home_first_default_home_leftmost(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_ui_env(monkeypatch)
    assert get_ui_home_first() is True


def test_home_first_truthy_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMW_UI_HOME_FIRST", "1")
    assert get_ui_home_first() is True


def test_home_first_opt_out_chat_before_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy eager Chat mount order when Chat must precede Home in ``tab_modules``."""
    monkeypatch.setenv("CMW_UI_HOME_FIRST", "0")
    assert get_ui_home_first() is False


def test_download_prep_after_stream_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_ui_env(monkeypatch)
    assert get_ui_download_prep_after_stream() is False


def test_download_prep_after_stream_env_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMW_UI_DOWNLOAD_PREP_AFTER_STREAM", "1")
    assert get_ui_download_prep_after_stream() is True
