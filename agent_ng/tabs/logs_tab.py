"""
Logs Tab Module for App NG
=========================

Handles logs monitoring, display, and management functionality.
This module encapsulates all logs-related UI components and functionality.
Supports internationalization (i18n) with Russian and English translations.
"""

from collections.abc import Callable
import logging
from typing import Any, Dict, Optional, Tuple

import gradio as gr


class LogsTab:
    """Logs tab component for monitoring and display"""

    def __init__(self, event_handlers: dict[str, Callable], language: str = "en", i18n_instance: gr.I18n | None = None):
        self.event_handlers = event_handlers
        self.components = {}
        self.language = language
        self.i18n = i18n_instance

    def create_tab(self) -> tuple[gr.TabItem, dict[str, Any]]:
        """
        Create the logs tab with all its components.
        
        Returns:
            Tuple of (TabItem, components_dict)
        """
        logging.getLogger(__name__).info("✅ LogsTab: Creating logs interface...")

        with gr.TabItem(self._get_translation("tab_logs"), id="logs") as tab:
            # Create logs interface
            self._create_logs_interface()

            # Connect event handlers
            self._connect_events()

        logging.getLogger(__name__).info("✅ LogsTab: Successfully created with all components and event handlers")
        return tab, self.components

    def _create_logs_interface(self):
        """Create the logs monitoring interface"""
        gr.Markdown(f"### {self._get_translation('logs_title')}")

        self.components["logs_display"] = gr.Markdown(
            self._get_translation("logs_initializing"),
            elem_classes=["status-card"]
        )

        with gr.Row():
            self.components["refresh_logs_btn"] = gr.Button(self._get_translation("refresh_logs_button"), elem_classes=["cmw-button"])
            self.components["clear_logs_btn"] = gr.Button(self._get_translation("clear_logs_button"), elem_classes=["cmw-button"])

    def _connect_events(self):
        """Connect all event handlers for the logs tab with concurrency control"""
        logging.getLogger(__name__).debug("🔗 LogsTab: Connecting event handlers with concurrency control...")

        # Get queue manager for concurrency control
        queue_manager = getattr(self, "main_app", None)
        if queue_manager:
            queue_manager = getattr(queue_manager, "queue_manager", None)

        if queue_manager:
            # Apply concurrency settings to logs events
            from agent_ng.queue_manager import apply_concurrency_to_click_event

            refresh_config = apply_concurrency_to_click_event(
                queue_manager, "logs_refresh", self.get_initialization_logs,
                [], [self.components["logs_display"]]
            )
            self.components["refresh_logs_btn"].click(**refresh_config)

            clear_config = apply_concurrency_to_click_event(
                queue_manager, "logs_refresh", self.clear_logs,
                [], [self.components["logs_display"]]
            )
            self.components["clear_logs_btn"].click(**clear_config)
        else:
            # Fallback to default behavior
            logging.getLogger(__name__).warning("⚠️ Queue manager not available - using default logs configuration")
            self.components["refresh_logs_btn"].click(
                fn=self.get_initialization_logs,
                outputs=[self.components["logs_display"]]
            )

            self.components["clear_logs_btn"].click(
                fn=self.clear_logs,
                outputs=[self.components["logs_display"]]
            )

        logging.getLogger(__name__).debug("✅ LogsTab: All event handlers connected successfully")

    def get_components(self) -> dict[str, Any]:
        """Get all components created by this tab"""
        return self.components

    def get_logs_display_component(self) -> gr.Markdown:
        """Get the logs display component for auto-refresh"""
        return self.components["logs_display"]

    # Logs handler methods
    def get_initialization_logs(self, request: gr.Request = None) -> str:
        """Get initialization logs as formatted string - now session-aware"""
        # Access the main app through event handlers context
        # This will be set by the main app when creating the tab
        if hasattr(self, "_main_app") and self._main_app:
            # Get session-specific logs
            if request and hasattr(self._main_app, "session_manager"):
                session_id = self._main_app.session_manager.get_session_id(request)

                # Get session-specific log handler
                from ..debug_streamer import get_log_handler
                session_log_handler = get_log_handler(session_id)

                # Combine static logs with real-time debug logs
                static_logs = "\n".join(self._main_app.initialization_logs)
                debug_logs = session_log_handler.get_current_logs()

                if debug_logs and debug_logs != "No logs available yet.":
                    return f"{static_logs}\n\n--- Real-time Debug Logs (Session: {session_id}) ---\n\n{debug_logs}"
                return static_logs
            else:
                # For auto-refresh, show only static logs since we can't determine the session
                static_logs = "\n".join(self._main_app.initialization_logs)
                return f"{static_logs}\n\n--- Auto-refresh mode: Session-specific logs not available ---"
        return "Logs not available - main app not connected"

    def clear_logs(self, request: gr.Request = None) -> str:
        """Clear logs and return confirmation - now session-aware"""
        if hasattr(self, "_main_app") and self._main_app:
            # Get session-specific logs
            session_id = "default"  # Default session
            if request and hasattr(self._main_app, "session_manager"):
                session_id = self._main_app.session_manager.get_session_id(request)

            # Get session-specific log handler
            from ..debug_streamer import get_log_handler
            session_log_handler = get_log_handler(session_id)
            session_log_handler.clear_logs()
            return self._get_translation("logs_cleared")
        return self._get_translation("logs_not_available")

    def set_main_app(self, main_app):
        """Set reference to main app for accessing logs functionality"""
        self._main_app = main_app

    def _get_translation(self, key: str) -> str:
        """Get a translation for a specific key"""
        # Always use direct translation for now to avoid i18n metadata issues
        from ..i18n_translations import get_translation_key
        return get_translation_key(key, self.language)
