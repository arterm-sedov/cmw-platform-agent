"""LLM controls: wire events after overview is registered (Config tab)."""

from unittest.mock import MagicMock, patch

import gradio as gr

from agent_ng.tabs.sidebar import Sidebar as SidebarPanel


def test_mount_llm_selection_ui_does_not_set_events_connected():
    """Wiring is deferred until ``ensure_llm_events_wired`` (after UIManager merge)."""
    panel = SidebarPanel(event_handlers={}, language="en")
    panel._llm_events_connected = False
    with gr.Blocks(), gr.Column():
        panel.mount_llm_selection_ui()
    assert panel._llm_events_connected is False


def test_apply_llm_selection_update_overview_reads_stats_not_combined_toast():
    """Overview after model change must reflect session stats, not success toast text."""
    panel = SidebarPanel(event_handlers={}, language="en")
    panel.main_app = MagicMock()
    stats_tab = MagicMock()
    stats_tab.format_stats_overview.return_value = "__overview_from_stats__"
    panel.main_app.tab_instances = {"stats": stats_tab}

    with patch.object(
        SidebarPanel,
        "_apply_llm_selection_combined",
        return_value="toast would be wrong here",
    ):
        out = panel._apply_llm_selection_update_overview_only(
            "provider_a / model_x", None
        )

    assert out == "__overview_from_stats__"
    stats_tab.format_stats_overview.assert_called_once()


def test_apply_llm_selection_update_overview_and_budget_returns_pair():
    panel = SidebarPanel(
        event_handlers={"update_token_budget": lambda _req: "__budget__"},
        language="en",
    )
    panel.main_app = MagicMock()
    stats_tab = MagicMock()
    stats_tab.format_stats_overview.return_value = "__ov__"
    panel.main_app.tab_instances = {"stats": stats_tab}

    with patch.object(SidebarPanel, "_apply_llm_selection_combined", return_value="x"):
        ov, bud = panel._apply_llm_selection_update_overview_and_budget("p / m", None)

    assert ov == "__ov__"
    assert bud == "__budget__"
