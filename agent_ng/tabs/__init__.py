"""
Tabs Module for App NG
=====================

Modular tab components for the Gradio interface.

This module contains individual tab implementations that can be composed
together to create the complete application interface.
"""

from .chat_tab import ChatTab
from .home_tab import HomeTab
from .logs_tab import LogsTab
from .stats_tab import StatsTab
from .config_tab import ConfigTab
from .sidebar import Sidebar

__all__ = [
    "ChatTab",
    "HomeTab",
    "LogsTab", 
    "StatsTab",
    "ConfigTab",
    "Sidebar",
]
