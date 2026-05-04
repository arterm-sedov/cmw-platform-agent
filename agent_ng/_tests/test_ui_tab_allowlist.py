"""Behavior tests for ``CMW_UI_TABS`` allowlist parsing."""

from __future__ import annotations

import pytest

from agent_ng.agent_config import (
    get_ui_disable_auto_timers,
    get_ui_download_prep_after_stream,
    get_ui_export_html_after_turn,
    get_ui_home_first,
    get_ui_stack_home_chat,
    get_ui_tab_allowlist,
)


@pytest.fixture(autouse=True)
def clear_ui_bisect_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "CMW_UI_TABS",
        "CMW_UI_TAB_LIMIT",
        "CMW_UI_STACK_HOME_CHAT",
        "CMW_UI_DISABLE_AUTO_TIMERS",
        "CMW_UI_EXPORT_HTML_AFTER_TURN",
        "CMW_UI_HOME_FIRST",
        "CMW_UI_DOWNLOAD_PREP_AFTER_STREAM",
    ):
        monkeypatch.delenv(key, raising=False)


def test_allowlist_unset_means_all_tabs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CMW_UI_TABS", raising=False)
    assert get_ui_tab_allowlist() is None


def test_allowlist_normalizes_case_and_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMW_UI_TABS", " home , STATS , sidebar ")
    assert get_ui_tab_allowlist() == frozenset({"home", "stats", "sidebar"})


def test_tab_limit_first_three(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMW_UI_TAB_LIMIT", "3")
    assert get_ui_tab_allowlist() == frozenset({"home", "chat", "logs"})


def test_tabs_explicit_overrides_tab_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMW_UI_TAB_LIMIT", "3")
    monkeypatch.setenv("CMW_UI_TABS", "chat")
    assert get_ui_tab_allowlist() == frozenset({"chat"})


def test_tab_limit_zero_means_all_tabs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMW_UI_TAB_LIMIT", "0")
    assert get_ui_tab_allowlist() is None


def test_tab_limit_invalid_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMW_UI_TAB_LIMIT", "x")
    assert get_ui_tab_allowlist() is None


def test_stack_home_chat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CMW_UI_STACK_HOME_CHAT", raising=False)
    assert get_ui_stack_home_chat() is False
    monkeypatch.setenv("CMW_UI_STACK_HOME_CHAT", "1")
    assert get_ui_stack_home_chat() is True


def test_disable_auto_timers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CMW_UI_DISABLE_AUTO_TIMERS", raising=False)
    assert get_ui_disable_auto_timers() is False
    monkeypatch.setenv("CMW_UI_DISABLE_AUTO_TIMERS", "true")
    assert get_ui_disable_auto_timers() is True


def test_export_html_after_turn_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CMW_UI_EXPORT_HTML_AFTER_TURN", raising=False)
    assert get_ui_export_html_after_turn() is False


def test_export_html_after_turn_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMW_UI_EXPORT_HTML_AFTER_TURN", "1")
    assert get_ui_export_html_after_turn() is True


def test_home_first_default_home_leftmost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CMW_UI_HOME_FIRST", raising=False)
    assert get_ui_home_first() is True


def test_home_first_truthy_still_home_leftmost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMW_UI_HOME_FIRST", "1")
    assert get_ui_home_first() is True


def test_home_first_chat_before_home_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMW_UI_HOME_FIRST", "0")
    assert get_ui_home_first() is False


def test_download_prep_after_stream_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CMW_UI_DOWNLOAD_PREP_AFTER_STREAM", raising=False)
    assert get_ui_download_prep_after_stream() is False


def test_download_prep_after_stream_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CMW_UI_DOWNLOAD_PREP_AFTER_STREAM", "1")
    assert get_ui_download_prep_after_stream() is True
