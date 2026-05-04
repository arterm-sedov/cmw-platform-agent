"""
Test Config Tab LLM API keys (BrowserState + session store)
=========================================================
"""

from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_config_tab_browser_state_schema():
    """BrowserState holds URL/creds + per-provider key map only."""
    import inspect

    from agent_ng.tabs.config_tab import ConfigTab

    source = inspect.getsource(ConfigTab._create_config_interface)

    assert "llm_provider_api_keys" in source
    assert "llm_provider_key_inputs" in source
    assert 'type="password"' in source
    assert 'llm_provider_override"] = gr.Dropdown' not in source
    assert "llm_provider_override" not in source, "legacy override key removed from schema"
    assert "llm_api_key_override" not in source, "legacy single-key field removed from schema"


def test_config_tab_save_load_uses_key_map_only():
    """Save/load persist llm_provider_api_keys (no override pair)."""
    import inspect

    from agent_ng.tabs.config_tab import ConfigTab

    save_source = inspect.getsource(ConfigTab._save_to_state)
    assert "llm_provider_api_keys" in save_source
    assert "_key_values_to_map" in save_source
    assert "llm_provider_override" not in save_source
    assert "llm_api_key_override" not in save_source

    load_source = inspect.getsource(ConfigTab._load_from_state)
    apply_source = inspect.getsource(ConfigTab._apply_session_config)
    assert "llm_provider_api_keys" in load_source
    assert "_apply_session_config" in save_source
    assert "_apply_session_config" in load_source
    assert "set_session_config" in apply_source
    assert "llm_provider_override" not in save_source
    assert "llm_api_key_override" not in save_source
    assert "llm_provider_override" not in load_source
    assert "llm_api_key_override" not in load_source


def test_resolve_session_id_delegates_to_session_manager_get_session_id():
    """When request is present, use SessionManager.get_session_id (same as chat)."""
    from agent_ng.tabs.config_tab import ConfigTab

    class _Req:
        session_hash = "abc"

    calls: list[object] = []

    class _SM:
        def get_session_id(self, request: object) -> str:
            calls.append(request)
            return "gradio_abc"

        def get_last_active_session_id(self) -> None:
            return None

    tab = ConfigTab(event_handlers={}, language="en")
    tab.main_app = type("MA", (), {"session_manager": _SM()})()
    req = _Req()
    assert tab._resolve_session_id(req) == "gradio_abc"
    assert calls == [req]


def test_resolve_session_id_never_calls_get_session_id_with_none():
    from agent_ng.tabs.config_tab import ConfigTab

    class _SM:
        def get_session_id(self, request: object) -> str:
            if request is None:
                msg = "get_session_id(None) must not be called"
                raise AssertionError(msg)
            return "unused"

        def get_last_active_session_id(self) -> str:
            return "fallback_sid"

    tab = ConfigTab(event_handlers={}, language="en")
    tab.main_app = type("MA", (), {"session_manager": _SM()})()
    assert tab._resolve_session_id(None) == "fallback_sid"


def test_resolve_session_id_returns_none_without_fallback():
    from agent_ng.tabs.config_tab import ConfigTab

    class _SM:
        def get_session_id(self, request: object) -> str:
            if request is None:
                raise AssertionError
            return "sid"

        def get_last_active_session_id(self) -> None:
            return None

    tab = ConfigTab(event_handlers={}, language="en")
    tab.main_app = type("MA", (), {"session_manager": _SM()})()
    assert tab._resolve_session_id(None) is None


def test_apply_session_config_calls_set_session_and_reinitialize():
    from agent_ng.tabs.config_tab import ConfigTab

    tab = ConfigTab(event_handlers={}, language="en")
    reinit_ids: list[str] = []
    tab._reinitialize_session_llm = lambda sid: reinit_ids.append(sid)

    payload = {
        "url": "https://x",
        "username": "",
        "password": "",
        "llm_provider_api_keys": {},
    }
    with patch("agent_ng.tabs.config_tab.set_session_config") as mock_set:
        tab._apply_session_config("sid123", payload)
    mock_set.assert_called_once_with("sid123", payload)
    assert reinit_ids == ["sid123"]


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


def test_save_to_state_request_before_var_positional_for_gradio_injection():
    """Gradio injects Request only for params before *args (special_args)."""
    import inspect

    from agent_ng.tabs.config_tab import ConfigTab

    params = list(inspect.signature(ConfigTab._save_to_state).parameters.values())
    var_i = next(i for i, p in enumerate(params) if p.kind == inspect.Parameter.VAR_POSITIONAL)
    req_i = next(i for i, p in enumerate(params) if p.name == "request")
    assert req_i < var_i


def test_gradio_special_args_inserts_request_into_save_inputs():
    """Harness: same injection path Gradio uses when the Save button fires."""
    from gradio.helpers import special_args

    from agent_ng.tabs.config_tab import ConfigTab

    tab = ConfigTab(event_handlers={}, language="en", i18n_instance=None)
    tab._config_llm_providers = ["openrouter", "groq"]
    browser_tail = {
        "llm_provider_api_keys": {},
        "url": "",
        "username": "",
        "password": "",
    }
    inputs = [
        "https://example.com/",
        "",
        "",
        "sk-one",
        "sk-two",
        browser_tail,
    ]
    sentinel = object()
    patched, *_ = special_args(
        tab._save_to_state,
        list(inputs),
        request=sentinel,  # type: ignore[arg-type]
        event_data=None,
    )
    assert patched[3] is sentinel
    assert patched[:3] == inputs[:3]
    assert patched[4:] == inputs[3:]


def test_save_to_state_merges_into_llm_provider_api_keys_stub():
    from agent_ng.tabs.config_tab import ConfigTab

    tab = ConfigTab(event_handlers={}, language="en", i18n_instance=None)
    tab._config_llm_providers = ["openrouter", "groq"]

    merged_t = tab._save_to_state(
        "https://example.com/",
        "",
        "",
        None,
        "sk-one",
        "sk-g-new",
        {
            "llm_provider_api_keys": {"groq": "should-be-replaced"},
            "url": "",
            "username": "",
            "password": "",
        },
    )
    merged = merged_t[0]
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
    merged_t = tab._save_to_state(
        "https://example.com/",
        "",
        "",
        None,
        df,
        {
            "llm_provider_api_keys": {},
            "url": "",
            "username": "",
            "password": "",
        },
    )
    merged = merged_t[0]
    assert merged["llm_provider_api_keys"]["openrouter"] == "from-df"
    assert merged["llm_provider_api_keys"]["groq"] == "g2"





def test_save_to_state_returns_tuple_with_passthrough_slots_for_llm_outputs():
    """Save wires optional LLM sidebar outputs — tuple is state + gr.update tails."""
    from agent_ng.tabs.config_tab import ConfigTab

    tab = ConfigTab(event_handlers={}, language="en", i18n_instance=None)
    tab._config_llm_providers = ["provider_a", "provider_b"]
    tab.sidebar_instance = type(
        "SB",
        (),
        {
            "components": {
                "provider_model_selector": object(),
                "use_fallback_model": object(),
                "fallback_model_selector": object(),
                "compression_enabled": object(),
            }
        },
    )()

    out = tab._save_to_state(
        "https://x/",
        "u",
        "p",
        None,
        "sk-g",
        "sk-o",
        None,
        None,
        None,
        None,
        {
            "llm_provider_api_keys": {},
            "url": "",
            "username": "",
            "password": "",
        },
    )
    assert isinstance(out, tuple)
    assert len(out) == 5
    assert out[0]["url"] == "https://x/"
    assert out[0]["llm_provider_api_keys"]["provider_a"] == "sk-g"
    assert out[0]["llm_provider_api_keys"]["provider_b"] == "sk-o"


def test_llm_manager_accepts_api_key_override():
    import inspect

    from agent_ng.llm_manager import LLMManager

    sig = inspect.signature(LLMManager.get_llm)
    assert "api_key_override" in sig.parameters


def test_llm_manager_get_api_key_with_override():
    import inspect

    from agent_ng.llm_manager import LLMManager

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
        {"llm_provider_api_keys": {"vendor_z": "sk-key"}},
    )
    assert session_manager.get_session_config(test_session_id) is not None
    session_manager.clear_session_config(test_session_id)
    assert session_manager.get_session_config(test_session_id) is None


def test_session_data_init_uses_keys_map_and_choice():
    import inspect

    from agent_ng.session_manager import SessionData

    source = inspect.getsource(SessionData._initialize_session_agent)
    assert "llm_provider_api_keys" in source
    assert "_session_llm_choice" in source
    assert "api_key_override" in source


def main():
    tests = [
        test_config_tab_browser_state_schema,
        test_config_tab_save_load_uses_key_map_only,
        test_save_to_state_request_before_var_positional_for_gradio_injection,
        test_gradio_special_args_inserts_request_into_save_inputs,
        test_resolve_session_id_delegates_to_session_manager_get_session_id,
        test_resolve_session_id_never_calls_get_session_id_with_none,
        test_resolve_session_id_returns_none_without_fallback,
        test_apply_session_config_calls_set_session_and_reinitialize,
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
