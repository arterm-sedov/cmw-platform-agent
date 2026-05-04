"""Concise agent status (stats_tab_overview_display) lives on Statistics tab."""

from agent_ng.app_ng_modular import NextGenApp
from agent_ng.tabs.stats_tab import StatsTab


def test_stats_tab_creates_overview_in_stats_card():
    tab = StatsTab(event_handlers={}, language="en")
    import gradio as gr

    with gr.Blocks():
        with gr.Tabs():
            with gr.TabItem("t", id="stats"):
                tab._create_stats_interface()
    assert "stats_tab_overview_display" in tab.components
    assert tab.components["stats_tab_overview_display"].elem_id == "stats-overview-display"


def test_update_all_ui_returns_four_outputs(monkeypatch):
    app = NextGenApp(language="en")
    app.tab_instances["stats"] = StatsTab(event_handlers={}, language="en")

    monkeypatch.setattr(app, "_update_status", lambda _request=None: "o")
    monkeypatch.setattr(app, "_refresh_stats", lambda _request=None: "s")
    monkeypatch.setattr(app, "_refresh_logs", lambda _request=None: "l")

    assert app.update_all_ui_components(None) == ("o", "s", "s", "l")
