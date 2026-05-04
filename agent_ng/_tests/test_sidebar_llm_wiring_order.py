"""LLM controls: wire events after overview is registered (Config tab)."""

import gradio as gr

from agent_ng.tabs.sidebar import Sidebar as SidebarPanel


def test_mount_llm_selection_ui_does_not_set_events_connected():
    """Wiring is deferred until ``ensure_llm_events_wired`` (after UIManager merge)."""
    panel = SidebarPanel(event_handlers={}, language="en")
    panel._llm_events_connected = False
    with gr.Blocks():
        with gr.Column():
            panel.mount_llm_selection_ui()
    assert panel._llm_events_connected is False
