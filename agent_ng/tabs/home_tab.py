"""
Home Tab Module for App NG
=========================

Handles the welcome/home page functionality with session-aware content.
This module provides a dedicated home tab with welcome information and quick start guidance.
Supports internationalization (i18n) with Russian and English translations.
"""

from collections.abc import Callable
import logging
from typing import Any, Optional

import gradio as gr

# Import for fallback translations
try:
    from agent_ng.i18n_translations import get_translation_key
except ImportError:
    get_translation_key = None


class HomeTab:
    """Home tab component for welcome and quick start information"""

    def __init__(
        self,
        event_handlers: dict[str, Callable],
        language: str = "en",
        i18n_instance: gr.I18n | None = None,
    ) -> None:
        self.event_handlers = event_handlers
        self.components = {}
        self.main_app = None  # Reference to main app for session management
        self.language = language
        self.i18n = i18n_instance

    def create_tab(self) -> tuple[gr.TabItem, dict[str, Any]]:
        """
        Create the home tab with welcome content.

        Returns:
            Tuple of (TabItem, components_dict)
        """
        logging.getLogger(__name__).info("✅ HomeTab: Creating home interface...")

        with gr.TabItem(self._get_translation("tab_home"), id="home") as tab:
            # Create home interface
            self._create_home_interface()

            # Connect event handlers
            self._connect_events()

        return tab, self.components

    def _create_home_interface(self):
        """Create the home interface with welcome content"""
        logging.getLogger(__name__).debug("🏠 HomeTab: Creating home interface...")

        # Main welcome section
        with (
            gr.Column(elem_classes=["home-container"]),
            gr.Row(elem_classes=["home-content"]),
            gr.Column(scale=2)
        ):
            with gr.Column(elem_classes=["welcome-block"]):
                gr.Markdown(
                    f"## {self._get_translation('welcome_title')}",
                    elem_classes=["welcome-title"]
                )
                gr.Markdown(
                    self._get_translation("welcome_description"),
                    elem_classes=["welcome-description"]
                )

            # Quick start section
            with gr.Column(elem_classes=["quick-start-block"]):
                gr.Markdown(
                    f"## {self._get_translation('quick_start_title')}",
                    elem_classes=["quick-start-title"]
                )
                gr.Markdown(
                    self._get_translation("quick_start_description"),
                    elem_classes=["quick-start-description"]
                )
    def _connect_events(self):
        """Connect event handlers for the home tab"""
        logging.getLogger(__name__).debug("🔗 HomeTab: Connecting event handlers...")

        # Add any home-specific event handlers here
        # For now, this is a static welcome page

    def set_main_app(self, main_app):
        """Set reference to main app for session management"""
        self.main_app = main_app

    def _get_translation(self, key: str, **kwargs: Any) -> str:
        """Get translation for a key from i18n system"""
        if self.i18n:
            try:
                return self.i18n.t(key, **kwargs)
            except Exception as e:
                logging.getLogger(__name__).debug(
                    "i18n translation failed for key '%s': %s", key, e
                )

        # Fallback to i18n_translations module
        if get_translation_key:
            return get_translation_key(key, self.language)
        return f"[{key}]"
