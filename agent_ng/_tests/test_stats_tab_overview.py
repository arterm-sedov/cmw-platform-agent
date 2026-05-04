"""Agent overview Markdown (stats_tab_overview_display) lives in Config when shown."""

import gradio as gr

from agent_ng.tabs.config_tab import ConfigTab
from agent_ng.tabs.stats_tab import StatsTab


def test_stats_tab_placeholder_only_when_config_hidden():
    tab = StatsTab(event_handlers={}, language="en")
    tab.register_overview_placeholder()
    assert "stats_tab_overview_display" in tab.components
    assert tab.components["stats_tab_overview_display"].visible is False


def test_config_tab_registers_overview_before_llm_mount():
    """Config creates overview so Sidebar LLM .change can wire to the same block."""
    with gr.Blocks():
        cfg = ConfigTab(event_handlers={}, language="en")
        cfg.register_agent_overview_display()
        assert "stats_tab_overview_display" in cfg.components
        assert cfg.components["stats_tab_overview_display"].elem_id == "stats-overview-display"


def test_update_all_ui_returns_four_outputs(monkeypatch):
    from agent_ng.app_ng_modular import NextGenApp

    app = NextGenApp(language="en")
    app.tab_instances["stats"] = StatsTab(event_handlers={}, language="en")

    monkeypatch.setattr(app, "_update_status", lambda _request=None: "o")
    monkeypatch.setattr(app, "_refresh_stats", lambda _request=None: "s")
    monkeypatch.setattr(app, "_refresh_logs", lambda _request=None: "l")

    assert app.update_all_ui_components(None) == ("o", "s", "s", "l")
