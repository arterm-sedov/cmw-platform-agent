"""CMW_USE_DOTENV gates platform credentials source, not Config tab visibility."""

import os
from unittest.mock import patch

from agent_ng.tabs.config_tab import ConfigTab


def test_use_dotenv_for_platform_truthy_values():
    for v in ("true", "True", "1", "yes"):
        with patch.dict(os.environ, {"CMW_USE_DOTENV": v}, clear=False):
            assert ConfigTab.use_dotenv_for_platform() is True


def test_use_dotenv_for_platform_falsey_values():
    for v in ("false", "0", "no", "off", ""):
        with patch.dict(os.environ, {"CMW_USE_DOTENV": v}, clear=False):
            assert ConfigTab.use_dotenv_for_platform() is False


def test_session_config_payload_merges_env_when_dotenv(monkeypatch):
    monkeypatch.setenv("CMW_USE_DOTENV", "true")
    monkeypatch.setenv("CMW_BASE_URL", "https://env.example/")
    monkeypatch.setenv("CMW_LOGIN", "envuser")
    monkeypatch.setenv("CMW_PASSWORD", "placeholder-from-env")
    tab = ConfigTab(event_handlers={}, language="en")
    browser = {
        "url": "https://browser/",
        "username": "b",
        "password": "placeholder-from-browser",
        "llm_provider_api_keys": {"provider_a": "k1"},
    }
    out = tab._session_config_payload(browser)
    assert out["url"] == "https://env.example/"
    assert out["username"] == "envuser"
    assert out["password"] == os.environ["CMW_PASSWORD"]
    assert out["llm_provider_api_keys"]["provider_a"] == "k1"


def test_session_config_payload_uses_browser_when_not_dotenv(monkeypatch):
    monkeypatch.setenv("CMW_USE_DOTENV", "false")
    tab = ConfigTab(event_handlers={}, language="en")
    browser = {
        "url": "https://browser/",
        "username": "u",
        "password": "placeholder-from-browser-row",
        "llm_provider_api_keys": {},
    }
    out = tab._session_config_payload(browser)
    assert out["url"] == "https://browser/"
    assert out["username"] == "u"
    assert out["password"] == browser["password"]


def test_browser_state_public_snapshot_strips_platform_when_dotenv(monkeypatch):
    monkeypatch.setenv("CMW_USE_DOTENV", "true")
    tab = ConfigTab(event_handlers={}, language="en")
    merged = {
        "url": "https://x",
        "username": "u",
        "password": "p",
        "llm_provider_api_keys": {"a": "1"},
    }
    pub = tab._browser_state_public_snapshot(merged)
    assert pub["url"] == ""
    assert pub["username"] == ""
    assert pub["password"] == ""
    assert pub["llm_provider_api_keys"]["a"] == "1"
