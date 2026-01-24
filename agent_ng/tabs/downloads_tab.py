"""
Downloads Tab Module for App NG
==============================

Handles file download functionality in a separate tab to isolate it from chat streaming.
This prevents freezes by keeping download button updates separate from the chat generator flow.
"""

from collections.abc import Callable
import logging
from typing import Any, Optional
import gradio as gr
from ..i18n_translations import get_translation_key


class DownloadsTab:
    """Downloads tab component for file downloads"""

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
        Create the downloads tab with download buttons.

        Returns:
            Tuple of (TabItem, components_dict)
        """
        logging.getLogger(__name__).info("✅ DownloadsTab: Creating downloads interface...")

        with gr.TabItem(self._get_translation("tab_downloads"), id="downloads") as tab:
            # Create downloads interface
            self._create_downloads_interface()

            # Connect event handlers
            self._connect_events()

        return tab, self.components

    def _create_downloads_interface(self):
        """Create the downloads interface with download buttons"""
        logging.getLogger(__name__).debug("📥 DownloadsTab: Creating downloads interface...")

        with gr.Column(elem_classes=["downloads-container"]):
            # Download buttons - initially hidden, updated after chat streaming completes
            with gr.Row():
                self.components["download_btn"] = gr.DownloadButton(
                    label=self._get_translation("download_button"),
                    variant="secondary",
                    elem_classes=["cmw-button"],
                    visible=False,  # Will be shown when files are ready
                )
                self.components["download_html_btn"] = gr.DownloadButton(
                    label=self._get_translation("download_html_button"),
                    variant="secondary",
                    elem_classes=["cmw-button"],
                    visible=False,  # Will be shown when files are ready
                )

    def _connect_events(self):
        """Connect event handlers for the downloads tab"""
        logging.getLogger(__name__).debug("🔗 DownloadsTab: Connecting event handlers...")

        # Event handlers will be connected from ui_manager after all tabs are created
        # This tab just provides the UI components

    def set_main_app(self, main_app):
        """Set reference to main app for session management"""
        self.main_app = main_app

    def _get_translation(self, key: str) -> str:
        """Get translation for a specific key"""
        return get_translation_key(key, self.language)
