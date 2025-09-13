"""
Tabs Module for App NG
=====================

Modular tab components for the Gradio interface.

This module contains individual tab implementations that can be composed
together to create the complete application interface.
"""

from .chat_tab import ChatTab
from .logs_tab import LogsTab
from .stats_tab import StatsTab

__all__ = [
    "ChatTab",
    "LogsTab", 
    "StatsTab"
]
