"""
UI Manager for App NG
====================

Handles Gradio interface creation, styling, and component management.
This module orchestrates the UI creation process while maintaining all existing functionality.
Supports internationalization (i18n) with Russian and English translations.
"""

from collections.abc import Callable
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .i18n_translations import get_translation_key
from .tabs.sidebar import Sidebar
import gradio as gr

# Import configuration with fallback for direct execution
try:
    from agent_ng.agent_config import get_refresh_intervals
except ImportError:
    # Fallback for direct execution
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent))
    from agent_config import get_refresh_intervals

class UIManager:
    """Manages Gradio UI creation and configuration with i18n support"""

    def __init__(self, language: str = "en", i18n_instance: gr.I18n | None = None):
        self.css_file_path = Path(__file__).parent.parent / "resources" / "css" / "cmw_copilot_theme.css"
        self.language = language
        self.i18n = i18n_instance
        self._setup_gradio_paths()
        self.components = {}

    def _get_translation(self, key: str) -> str:
        """Get translation for a specific key"""
        return get_translation_key(key, self.language)

    def _setup_gradio_paths(self):
        """Setup Gradio static resource paths"""
        RESOURCES_DIR = Path(__file__).parent.parent / "resources"
        try:
            existing_allowed = os.environ.get("GRADIO_ALLOWED_PATHS", "")
            parts = [p for p in existing_allowed.split(os.pathsep) if p]
            if str(RESOURCES_DIR) not in parts:
                parts.append(str(RESOURCES_DIR))
            os.environ["GRADIO_ALLOWED_PATHS"] = os.pathsep.join(parts)
            logging.getLogger(__name__).debug(f"Gradio static allowed paths: {os.environ['GRADIO_ALLOWED_PATHS']}")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not set GRADIO_ALLOWED_PATHS: {e}")

    def create_interface(self, tab_modules: list[Any], event_handlers: dict[str, Callable], main_app=None) -> gr.Blocks:
        """
        Create the main Gradio interface using tab modules with i18n support.

        Args:
            tab_modules: List of tab module instances
            event_handlers: Dictionary of event handlers

        Returns:
            Gradio Blocks interface
        """
        logging.getLogger(__name__).info("🏗️ UIManager: Starting interface creation...")

        # Clear components to ensure clean state
        self.components.clear()

        # Get translated title
        app_title = self._get_translation("app_title")
        hero_title = self._get_translation("hero_title")

        with gr.Blocks(
            css_paths=[self.css_file_path],
            title=app_title,
            theme=gr.themes.Soft()
        ) as demo:

            # Header
            with gr.Row(), gr.Column():
                gr.Markdown(f"# {hero_title}", elem_classes=["hero-title"])

            # Create common sidebar using dedicated sidebar module
            sidebar_instance = Sidebar(event_handlers, language=self.language, i18n_instance=self.i18n)
            sidebar_instance.set_main_app(main_app)  # Pass main app reference

            sidebar, sidebar_components = sidebar_instance.create_sidebar()
            # Consolidate sidebar components
            self.components.update(sidebar_components)
            self.components["sidebar_instance"] = sidebar_instance

            with gr.Tabs():
                # Create tabs using provided tab modules
                for tab_module in tab_modules:
                    if tab_module:
                        tab_item, tab_components = tab_module.create_tab()
                        # Consolidate all components in one place
                        self.components.update(tab_components)
                        # Store tab reference for later use
                        self.components[f"{tab_module.__class__.__name__.lower()}_tab"] = tab_module

            # Connect quick action dropdown after all components are available
            if "sidebar_instance" in self.components:
                sidebar_instance = self.components["sidebar_instance"]
                sidebar_instance.connect_quick_action_dropdown()

            # Wire end-of-turn event-driven refresh using existing chat events
            try:
                update_all_ui_handler = event_handlers.get("update_all_ui")
                status_comp = self.components.get("status_display")
                stats_comp = self.components.get("stats_display")
                logs_comp = self.components.get("logs_display")
                token_budget_comp = self.components.get("token_budget_display")
                update_token_budget_handler = event_handlers.get("update_token_budget")

                chat_tab_instance = self.components.get("chattab_tab")
                if update_all_ui_handler and status_comp and stats_comp and logs_comp and chat_tab_instance:
                    refresh_outputs = [status_comp, stats_comp, logs_comp]

                    # After send (streaming) completes
                    if hasattr(chat_tab_instance, "streaming_event") and chat_tab_instance.streaming_event:
                        chat_tab_instance.streaming_event.then(
                            fn=update_all_ui_handler,
                            outputs=refresh_outputs
                        )

                    # After submit completes
                    if hasattr(chat_tab_instance, "submit_event") and chat_tab_instance.submit_event:
                        chat_tab_instance.submit_event.then(
                            fn=update_all_ui_handler,
                            outputs=refresh_outputs
                        )

                    logging.getLogger(__name__).debug("✅ Event-driven UI refresh wired for end-of-turn updates")

                # Token budget refresh: wire separately to avoid changing update_all_ui signature
                if (
                    update_token_budget_handler
                    and token_budget_comp
                    and chat_tab_instance
                ):
                    if hasattr(chat_tab_instance, "streaming_event") and chat_tab_instance.streaming_event:
                        chat_tab_instance.streaming_event.then(
                            fn=update_token_budget_handler,
                            outputs=[token_budget_comp],
                        )
                    if hasattr(chat_tab_instance, "submit_event") and chat_tab_instance.submit_event:
                        chat_tab_instance.submit_event.then(
                            fn=update_token_budget_handler,
                            outputs=[token_budget_comp],
                        )
                    # Wire clear button to update token budget immediately (event-driven)
                    # Chain to the existing clear button click event
                    if hasattr(chat_tab_instance, "clear_event") and chat_tab_instance.clear_event:
                        chat_tab_instance.clear_event.then(
                            fn=update_token_budget_handler,
                            outputs=[token_budget_comp],
                        )
                    # Wire stop button to update token budget immediately (event-driven)
                    # Chain to the existing stop button click event
                    if hasattr(chat_tab_instance, "stop_event") and chat_tab_instance.stop_event:
                        chat_tab_instance.stop_event.then(
                            fn=update_token_budget_handler,
                            outputs=[token_budget_comp],
                        )
                    logging.getLogger(__name__).debug("✅ Token budget event-driven refresh wired for end-of-turn updates, clear button, and stop button")
            except Exception as e:
                logging.getLogger(__name__).warning(f"Could not wire event-driven refresh: {e}")

            # Setup auto-refresh timers
            self._setup_auto_refresh(demo, event_handlers)

        logging.getLogger(__name__).info("✅ UIManager: Interface created successfully with all components and timers")
        return demo

    def _setup_auto_refresh(self, demo: gr.Blocks, event_handlers: dict[str, Callable]):
        """Setup auto-refresh timers for status and logs - matches original behavior exactly"""
        # Get handlers with validation
        update_status_handler = event_handlers.get("update_status")
        update_token_budget_handler = event_handlers.get("update_token_budget")
        refresh_logs_handler = event_handlers.get("refresh_logs")
        update_progress_handler = event_handlers.get("update_progress_display")


        # Load initial UI state once on startup
        if "status_display" in self.components and update_status_handler:
            demo.load(
                fn=update_status_handler,
                outputs=[self.components["status_display"]]
            )

        if "token_budget_display" in self.components and update_token_budget_handler:
            demo.load(
                fn=update_token_budget_handler,
                outputs=[self.components["token_budget_display"]]
            )

        # LLM selection components are initialized with static values
        # and only update when explicitly triggered by user actions

        if "logs_display" in self.components and refresh_logs_handler:
            demo.load(
                fn=refresh_logs_handler,
                outputs=[self.components["logs_display"]]
            )

        if "progress_display" in self.components and update_progress_handler:
            demo.load(
                fn=update_progress_handler,
                outputs=[self.components["progress_display"]]
            )

        refresh_stats_handler = event_handlers.get("refresh_stats")
        if "stats_display" in self.components and refresh_stats_handler:
            demo.load(
                fn=refresh_stats_handler,
                outputs=[self.components["stats_display"]]
            )

        # Setup auto-refresh timers for real-time updates
        self._setup_auto_refresh_timers(demo, event_handlers)

    def _setup_auto_refresh_timers(self, demo: gr.Blocks, event_handlers: dict[str, Callable]):
        """Setup auto-refresh timers for real-time updates (single interval)"""
        logging.getLogger(__name__).info("🔄 Setting up auto-refresh timers...")

        # Single interval from central configuration
        refresh_interval = get_refresh_intervals().interval

        # Status updates
        if "status_display" in self.components and event_handlers.get("update_status"):
            status_timer = gr.Timer(refresh_interval, active=True)
            status_timer.tick(
                fn=event_handlers["update_status"],
                outputs=[self.components["status_display"]]
            )
            logging.getLogger(__name__).debug(f"✅ Status auto-refresh timer set ({refresh_interval}s)")

        # Token budget updates
        if "token_budget_display" in self.components and event_handlers.get("update_token_budget"):
            token_budget_timer = gr.Timer(refresh_interval, active=True)
            token_budget_timer.tick(
                fn=event_handlers["update_token_budget"],
                outputs=[self.components["token_budget_display"]]
            )
            logging.getLogger(__name__).debug(f"✅ Token budget auto-refresh timer set ({refresh_interval}s)")

        # LLM selection updates - no auto-refresh (explicit only)
        logging.getLogger(__name__).debug("✅ LLM selection components will update only when explicitly triggered")

        # Logs updates
        if "logs_display" in self.components and event_handlers.get("refresh_logs"):
            logs_timer = gr.Timer(refresh_interval, active=True)
            logs_timer.tick(
                fn=event_handlers["refresh_logs"],
                outputs=[self.components["logs_display"]]
            )
            logging.getLogger(__name__).debug(f"✅ Logs auto-refresh timer set ({refresh_interval}s)")

        # Stats updates
        if "stats_display" in self.components and event_handlers.get("refresh_stats"):
            stats_timer = gr.Timer(refresh_interval, active=True)
            stats_timer.tick(
                fn=event_handlers["refresh_stats"],
                outputs=[self.components["stats_display"]]
            )
            logging.getLogger(__name__).debug(f"✅ Stats auto-refresh timer set ({refresh_interval}s)")

        # Progress/iteration updates (faster updates for turn progress)
        if "progress_display" in self.components and event_handlers.get("update_progress_display"):
            iteration_interval = get_refresh_intervals().iteration
            progress_timer = gr.Timer(iteration_interval, active=True)
            progress_timer.tick(
                fn=event_handlers["update_progress_display"],
                outputs=[self.components["progress_display"]]
            )
            logging.getLogger(__name__).debug(f"✅ Progress auto-refresh timer set ({iteration_interval}s)")

        logging.getLogger(__name__).info("🔄 Auto-refresh timers configured successfully")


    def get_components(self) -> dict[str, Any]:
        """Get all components created by the UI manager"""
        return self.components

    def get_component(self, name: str) -> Any:
        """Get a specific component by name"""
        return self.components.get(name)

    def set_agent(self, agent):
        """Set the agent reference on all tabs that support it"""
        for key, component in self.components.items():
            if hasattr(component, "set_agent"):
                component.set_agent(agent)

# Global instances for different languages
_ui_manager_en = None
_ui_manager_ru = None

def get_ui_manager(language: str = "en", i18n_instance: gr.I18n | None = None) -> UIManager:
    """
    Get the global UI manager instance for the specified language.

    Args:
        language: Language code ('en' or 'ru')
        i18n_instance: Optional Gradio I18n instance

    Returns:
        UIManager instance
    """
    global _ui_manager_en, _ui_manager_ru

    if language.lower() == "ru":
        if _ui_manager_ru is None:
            _ui_manager_ru = UIManager(language="ru", i18n_instance=i18n_instance)
        return _ui_manager_ru
    else:
        if _ui_manager_en is None:
            _ui_manager_en = UIManager(language="en", i18n_instance=i18n_instance)
        return _ui_manager_en
