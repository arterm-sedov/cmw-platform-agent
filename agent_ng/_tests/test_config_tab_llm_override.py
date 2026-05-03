"""
Test Config Tab LLM API keys (BrowserState + session store)
=========================================================
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_config_tab_browser_state_schema():
    """BrowserState holds URL/creds + per-provider key map only."""
    from agent_ng.tabs.config_tab import ConfigTab

    import inspect

    source = inspect.getsource(ConfigTab._create_config_interface)

    assert "llm_provider_api_keys" in source
    assert "llm_provider_key_inputs" in source
    assert 'type="password"' in source
    assert 'llm_provider_override"] = gr.Dropdown' not in source
    assert "llm_provider_override" not in source, "legacy override key removed from schema"
    assert "llm_api_key_override" not in source, "legacy single-key field removed from schema"


def test_config_tab_save_load_uses_key_map_only():
    """Save/load persist llm_provider_api_keys (no override pair)."""
    from agent_ng.tabs.config_tab import ConfigTab

    import inspect

    save_source = inspect.getsource(ConfigTab._save_to_state)
    assert "llm_provider_api_keys" in save_source
    assert "_key_values_to_map" in save_source
    assert "llm_provider_override" not in save_source
    assert "llm_api_key_override" not in save_source

    load_source = inspect.getsource(ConfigTab._load_from_state)
    assert "llm_provider_api_keys" in load_source
    assert "llm_provider_override" not in load_source
    assert "llm_api_key_override" not in load_source


def test_normalize_llm_provider_api_keys_map_only():
    from agent_ng.tabs.config_tab import ConfigTab

    assert ConfigTab._normalize_llm_provider_api_keys(None) == {}
    mp = ConfigTab._normalize_llm_provider_api_keys(
        {"llm_provider_api_keys": {" openrouter ": " sk-aa "}}
    )
    assert mp == {"openrouter": "sk-aa"}

    stray = ConfigTab._normalize_llm_provider_api_keys(
        {
            "llm_provider_api_keys": {"openrouter": "sk-o"},
            "llm_provider_override": "groq",
            "llm_api_key_override": "sk-g",
        }
    )
    assert stray == {"openrouter": "sk-o"}


def test_parse_llm_keys_dataframe_pandas_and_lists():
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

    merged = tab._save_to_state(
        "https://example.com/",
        "",
        "",
        "sk-one",
        "sk-g-new",
        {
            "llm_provider_api_keys": {"groq": "should-be-replaced"},
            "url": "",
            "username": "",
            "password": "",
        },
    )
    assert merged["llm_provider_api_keys"]["openrouter"] == "sk-one"
    assert merged["llm_provider_api_keys"]["groq"] == "sk-g-new"
    assert "llm_api_key_override" not in merged


def test_save_to_state_accepts_dataframe_single_arg():
    from agent_ng.tabs.config_tab import ConfigTab

    tab = ConfigTab(event_handlers={}, language="en", i18n_instance=None)
    tab._config_llm_providers = ["openrouter", "groq"]

    df = pd.DataFrame(
        [["openrouter", "from-df"], ["groq", "g2"]],
        columns=["Provider", "API Key"],
    )
    merged = tab._save_to_state(
        "https://example.com/",
        "",
        "",
        df,
        {
            "llm_provider_api_keys": {},
            "url": "",
            "username": "",
            "password": "",
        },
    )
    assert merged["llm_provider_api_keys"]["openrouter"] == "from-df"
    assert merged["llm_provider_api_keys"]["groq"] == "g2"


def test_llm_manager_accepts_api_key_override():
    from agent_ng.llm_manager import LLMManager
    import inspect

    sig = inspect.signature(LLMManager.get_llm)
    assert "api_key_override" in sig.parameters


def test_llm_manager_get_api_key_with_override():
    from agent_ng.llm_manager import LLMManager
    import inspect

    sig = inspect.signature(LLMManager._get_api_key)
    assert "api_key_override" in sig.parameters


def test_session_manager_stores_api_key_map():
    from agent_ng import session_manager

    test_session_id = "test_llm_keys_session"

    session_manager.set_session_config(
        test_session_id,
        {
            "url": "https://test.com",
            "username": "test",
            "password": "test123",
            "llm_provider_api_keys": {"groq": "sk-test-key"},
        },
    )

    config = session_manager.get_session_config(test_session_id)
    assert config is not None
    assert config.get("llm_provider_api_keys", {}).get("groq") == "sk-test-key"

    session_manager.clear_session_config(test_session_id)


def test_session_manager_merge_preserves_keys():
    from agent_ng import session_manager

    sid = "test_merge_keys"
    session_manager.set_session_config(
        sid,
        {
            "url": "https://a",
            "llm_provider_api_keys": {"openrouter": "sk-o"},
        },
    )
    session_manager.set_session_config(sid, {"username": "u1"})
    cfg = session_manager.get_session_config(sid)
    assert cfg["url"] == "https://a"
    assert cfg["username"] == "u1"
    assert cfg["llm_provider_api_keys"]["openrouter"] == "sk-o"
    session_manager.clear_session_config(sid)


def test_session_manager_clear_config():
    from agent_ng import session_manager

    test_session_id = "test_clear_session"
    session_manager.set_session_config(
        test_session_id,
        {"llm_provider_api_keys": {"gemini": "sk-key"}},
    )
    assert session_manager.get_session_config(test_session_id) is not None
    session_manager.clear_session_config(test_session_id)
    assert session_manager.get_session_config(test_session_id) is None


def test_session_data_init_uses_keys_map_and_choice():
    from agent_ng.session_manager import SessionData

    import inspect

    source = inspect.getsource(SessionData._initialize_session_agent)
    assert "llm_provider_api_keys" in source
    assert "_session_llm_choice" in source
    assert "api_key_override" in source


def main():
    tests = [
        test_config_tab_browser_state_schema,
        test_config_tab_save_load_uses_key_map_only,
        test_normalize_llm_provider_api_keys_map_only,
        test_parse_llm_keys_dataframe_pandas_and_lists,
        test_key_values_to_map,
        test_save_to_state_merges_into_llm_provider_api_keys_stub,
        test_save_to_state_accepts_dataframe_single_arg,
        test_llm_manager_accepts_api_key_override,
        test_llm_manager_get_api_key_with_override,
        test_session_manager_stores_api_key_map,
        test_session_manager_merge_preserves_keys,
        test_session_manager_clear_config,
        test_session_data_init_uses_keys_map_and_choice,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"❌ {t.__name__}: {e}")
            failed += 1
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
