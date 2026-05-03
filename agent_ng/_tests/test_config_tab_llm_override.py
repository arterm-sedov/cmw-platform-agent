"""
Test Config Tab LLM Override Feature
====================================

Tests for LLM provider key override functionality in config tab.
TDD: Tests written first, implementation to follow.
"""

import sys
import os
from pathlib import Path
from unittest.mock import patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_config_tab_browser_state_schema():
    """Test that BrowserState schema includes LLM override fields"""
    from agent_ng.tabs.config_tab import ConfigTab

    # Read the source code and check for the fields
    import inspect

    source = inspect.getsource(ConfigTab._create_config_interface)

    assert "llm_provider_override" in source, "llm_provider_override not in _create_config_interface"
    assert "llm_api_key_override" in source, "llm_api_key_override not in BrowserState default"
    assert "llm_provider_api_keys" in source, "llm_provider_api_keys not in BrowserState default"
    assert "llm_provider_key_inputs" in source, "per-provider key inputs not wired"
    assert 'type="password"' in source, "API key fields should use password masking"
    assert 'llm_provider_override"] = gr.Dropdown' not in source, (
        "active provider should not be a config-tab dropdown"
    )
    print("✅ BrowserState schema includes LLM override fields")


def test_config_tab_save_load_includes_llm_override():
    """Test that _save_to_state and _load_from_state handle LLM override fields"""
    from agent_ng.tabs.config_tab import ConfigTab
    import inspect

    # Check _save_to_state accepts llm override params
    save_source = inspect.getsource(ConfigTab._save_to_state)
    assert "llm_provider_override" in save_source, "llm_provider_override not in _save_to_state"
    assert "llm_api_key_override" in save_source, "llm_api_key_override not in _save_to_state"
    assert "llm_provider_api_keys" in save_source, "llm_provider_api_keys not in _save_to_state"
    assert "_key_values_to_map" in save_source, "key tuple merge not in _save_to_state"
    assert "_default_active_llm_provider" in save_source, (
        "active provider should come from startup default"
    )

    # Check _load_from_state returns LLM override fields
    load_source = inspect.getsource(ConfigTab._load_from_state)
    assert "llm_provider_override" in load_source, "llm_provider_override not in _load_from_state"
    assert "llm_api_key_override" in load_source, "llm_api_key_override not in _load_from_state"
    assert "llm_provider_api_keys" in load_source or "_normalize_llm_provider_api_keys" in load_source
    assert "_default_active_llm_provider" in load_source, (
        "load should apply startup default provider to session"
    )

    print("✅ Save/Load methods handle LLM override fields")


def test_normalize_llm_provider_api_keys_from_map_and_legacy():
    from agent_ng.tabs.config_tab import ConfigTab

    assert ConfigTab._normalize_llm_provider_api_keys(None) == {}
    mp = ConfigTab._normalize_llm_provider_api_keys(
        {
            "llm_provider_api_keys": {" openrouter ": " sk-aa "},
            "llm_provider_override": "",
            "llm_api_key_override": "",
        }
    )
    assert mp == {"openrouter": "sk-aa"}

    leg = ConfigTab._normalize_llm_provider_api_keys(
        {
            "llm_provider_api_keys": {},
            "llm_provider_override": "groq",
            "llm_api_key_override": "sk-g",
        }
    )
    assert leg == {"groq": "sk-g"}

    both = ConfigTab._normalize_llm_provider_api_keys(
        {
            "llm_provider_api_keys": {"openrouter": "sk-o"},
            "llm_provider_override": "groq",
            "llm_api_key_override": "sk-g",
        }
    )
    assert both["openrouter"] == "sk-o"
    assert both["groq"] == "sk-g"


def test_parse_llm_keys_dataframe_pandas_and_lists():
    import pandas as pd

    from agent_ng.tabs.config_tab import ConfigTab

    provs = ["openrouter", "groq"]
    df = pd.DataFrame([["openrouter", " k1 "], ["groq", ""]], columns=["a", "b"])
    m1 = ConfigTab._parse_llm_keys_dataframe(df, provs)
    assert m1 == {"openrouter": "k1", "groq": ""}

    m2 = ConfigTab._parse_llm_keys_dataframe(
        [["openrouter", "x"], ["groq", "y"]],
        provs,
    )
    assert m2 == {"openrouter": "x", "groq": "y"}


def test_key_values_to_map():
    from agent_ng.tabs.config_tab import ConfigTab

    assert ConfigTab._key_values_to_map([], ["a"]) == {"a": ""}
    assert ConfigTab._key_values_to_map(["x", "y"], ["a", "b"]) == {"a": "x", "b": "y"}


def test_save_to_state_merges_into_llm_provider_api_keys_stub():
    from agent_ng.tabs.config_tab import ConfigTab

    tab = ConfigTab(event_handlers={}, language="en", i18n_instance=None)
    tab._config_llm_providers = ["openrouter", "groq"]

    with patch.object(tab, "_default_active_llm_provider", return_value="openrouter"):
        merged = tab._save_to_state(
            "https://example.com/",
            "",
            "",
            "sk-one",
            "sk-g-new",
            {
                "llm_provider_api_keys": {"groq": "should-be-replaced"},
                "llm_provider_override": "",
                "llm_api_key_override": "",
                "url": "",
                "username": "",
                "password": "",
            },
        )
    assert merged["llm_provider_api_keys"]["openrouter"] == "sk-one"
    assert merged["llm_provider_api_keys"]["groq"] == "sk-g-new"
    assert merged["llm_api_key_override"] == "sk-one"


def test_save_to_state_accepts_legacy_dataframe_single_arg():
    """Single DataFrame in *rest still merges (legacy BrowserState round-trips)."""
    import pandas as pd

    from agent_ng.tabs.config_tab import ConfigTab

    tab = ConfigTab(event_handlers={}, language="en", i18n_instance=None)
    tab._config_llm_providers = ["openrouter", "groq"]

    df = pd.DataFrame(
        [["openrouter", "from-df"], ["groq", "g2"]],
        columns=["Provider", "API Key"],
    )
    with patch.object(tab, "_default_active_llm_provider", return_value="openrouter"):
        merged = tab._save_to_state(
            "https://example.com/",
            "",
            "",
            df,
            {
                "llm_provider_api_keys": {},
                "llm_provider_override": "",
                "llm_api_key_override": "",
                "url": "",
                "username": "",
                "password": "",
            },
        )
    assert merged["llm_provider_api_keys"]["openrouter"] == "from-df"
    assert merged["llm_provider_api_keys"]["groq"] == "g2"


def test_llm_manager_accepts_api_key_override():
    """Test that LLMManager get_llm accepts api_key_override parameter"""
    from agent_ng.llm_manager import LLMManager
    import inspect

    sig = inspect.signature(LLMManager.get_llm)
    params = list(sig.parameters.keys())

    assert "api_key_override" in params, f"api_key_override not in get_llm params: {params}"
    print("✅ LLMManager.get_llm accepts api_key_override")


def test_llm_manager_get_api_key_with_override():
    """Test _get_api_key method accepts api_key_override"""
    from agent_ng.llm_manager import LLMManager
    import inspect

    sig = inspect.signature(LLMManager._get_api_key)
    params = list(sig.parameters.keys())

    assert "api_key_override" in params, f"api_key_override not in _get_api_key params: {params}"
    print("✅ LLMManager._get_api_key accepts api_key_override")


def test_session_manager_has_llm_config():
    """Test that session_manager can store/retrieve LLM override config"""
    from agent_ng import session_manager

    test_session_id = "test_llm_override_session"

    session_manager.set_session_config(
        test_session_id,
        {
            "url": "https://test.com",
            "username": "test",
            "password": "test123",
            "llm_provider_override": "groq",
            "llm_api_key_override": "sk-test-key",
        },
    )

    config = session_manager.get_session_config(test_session_id)

    assert config is not None, "Session config should not be None"
    assert config.get("llm_provider_override") == "groq", "llm_provider_override not stored"
    assert config.get("llm_api_key_override") == "sk-test-key", "llm_api_key_override not stored"

    session_manager.clear_session_config(test_session_id)
    print("✅ Session manager stores LLM override config")


def test_session_manager_clear_config():
    """Test clear_session_config function"""
    from agent_ng import session_manager

    test_session_id = "test_clear_session"

    session_manager.set_session_config(
        test_session_id,
        {"llm_provider_override": "gemini", "llm_api_key_override": "sk-key"},
    )

    config = session_manager.get_session_config(test_session_id)
    assert config is not None

    session_manager.clear_session_config(test_session_id)
    config_after = session_manager.get_session_config(test_session_id)
    assert config_after is None, "Session config should be None after clearing"

    print("✅ clear_session_config works")


def test_session_data_uses_llm_override():
    """Test that SessionData._initialize_session_agent uses LLM override"""
    from agent_ng.session_manager import SessionData
    import inspect

    source = inspect.getsource(SessionData._initialize_session_agent)

    assert "llm_provider_override" in source, "SessionData doesn't check llm_provider_override"
    assert "llm_api_key_override" in source, "SessionData doesn't check llm_api_key_override"
    assert "api_key_override" in source, "SessionData doesn't pass api_key_override"

    print("✅ SessionData uses LLM override from session config")


def main():
    print("🧪 Testing Config Tab LLM Override Feature")
    print("=" * 50)

    tests = [
        test_config_tab_browser_state_schema,
        test_config_tab_save_load_includes_llm_override,
        test_normalize_llm_provider_api_keys_from_map_and_legacy,
        test_parse_llm_keys_dataframe_pandas_and_lists,
        test_key_values_to_map,
        test_save_to_state_merges_into_llm_provider_api_keys_stub,
        test_save_to_state_accepts_legacy_dataframe_single_arg,
        test_llm_manager_accepts_api_key_override,
        test_llm_manager_get_api_key_with_override,
        test_session_manager_has_llm_config,
        test_session_manager_clear_config,
        test_session_data_uses_llm_override,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
