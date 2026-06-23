"""Sidebar layout uses native Gradio 6 collapsible ``gr.Sidebar``."""

import inspect


def test_ui_manager_wraps_panel_in_gr_sidebar():
    from agent_ng import ui_manager

    src = inspect.getsource(ui_manager.UIManager.create_interface)
    assert "with gr.Sidebar(" in src
    assert "cmw-gradio-sidebar" in src
    assert "SidebarPanel" in src or "from .tabs.sidebar import Sidebar as SidebarPanel" in inspect.getsource(
        ui_manager
    )


def test_sidebar_panel_class_alias_exists():
    from agent_ng.tabs.sidebar import Sidebar as SidebarPanel

    assert SidebarPanel.__name__ == "Sidebar"
