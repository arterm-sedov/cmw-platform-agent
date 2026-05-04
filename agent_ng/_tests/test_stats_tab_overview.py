"""Stats tab overview block (former sidebar status)."""

from agent_ng.tabs.stats_tab import StatsTab


def test_stats_tab_registers_overview_placeholder_idempotently():
    tab = StatsTab(event_handlers={}, language="en")
    tab.register_overview_placeholder()
    assert "stats_tab_overview_display" in tab.components
    tab.register_overview_placeholder()
    assert len([k for k in tab.components if k == "stats_tab_overview_display"]) == 1


def test_format_stats_overview_without_agent():
    tab = StatsTab(event_handlers={}, language="en")
    assert tab.format_stats_overview(None) == "Agent not available"


def test_update_all_ui_returns_four_outputs(monkeypatch):
    from agent_ng.app_ng_modular import NextGenApp

    app = NextGenApp(language="en")
    app.tab_instances["stats"] = StatsTab(event_handlers={}, language="en")

    monkeypatch.setattr(app, "_update_status", lambda _request=None: "o")
    monkeypatch.setattr(app, "_refresh_stats", lambda _request=None: "s")
    monkeypatch.setattr(app, "_refresh_logs", lambda _request=None: "l")

    assert app.update_all_ui_components(None) == ("o", "s", "s", "l")
