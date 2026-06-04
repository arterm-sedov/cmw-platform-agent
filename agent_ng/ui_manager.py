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
from typing import Any

import gradio as gr

from .i18n_translations import get_translation_key
from .tabs.sidebar import Sidebar as SidebarPanel

# Import configuration with fallback for direct execution
try:
    from agent_ng.agent_config import (
        get_refresh_intervals,
        get_ui_download_prep_after_stream,
    )
except ImportError:
    # Fallback for direct execution
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).parent))
    from agent_config import get_refresh_intervals, get_ui_download_prep_after_stream


class UIManager:
    """Manages Gradio UI creation and configuration with i18n support"""

    def __init__(self, language: str = "en", i18n_instance: gr.I18n | None = None):
        self.css_file_path = (
            Path(__file__).parent.parent / "resources" / "css" / "cmw_copilot_theme.css"
        )
        self.language = language
        self.i18n = i18n_instance
        self._setup_gradio_paths()
        self.components = {}
        self._main_app = None  # Will be set by create_interface if main_app is provided

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
            logging.getLogger(__name__).debug(
                f"Gradio static allowed paths: {os.environ['GRADIO_ALLOWED_PATHS']}"
            )
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Could not set GRADIO_ALLOWED_PATHS: {e}"
            )

    def create_interface(
        self,
        tab_modules: list[Any],
        event_handlers: dict[str, Callable],
        main_app=None,
        *,
        sidebar_instance: SidebarPanel | None = None,
    ) -> gr.Blocks:
        # Store main_app reference for initialization completion checks
        self._main_app = main_app
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

        # In Gradio 6, app-level theme and CSS are configured on launch(),
        # so Blocks only receives structural arguments like title.
        with gr.Blocks(title=app_title) as demo:
            # Header
            with gr.Row(), gr.Column():
                gr.Markdown(f"# {hero_title}", elem_classes=["hero-title"])

            sb = sidebar_instance or SidebarPanel(
                event_handlers, language=self.language, i18n_instance=self.i18n
            )
            sb.set_main_app(main_app)

            config_tab_present = any(
                getattr(m, "__class__", type(None)).__name__ == "ConfigTab"
                for m in tab_modules
            )

            with gr.Row(equal_height=False):
                with gr.Sidebar(
                    label=self._get_translation("tab_sidebar"),
                    open=True,
                    width=370,
                    position="left",
                    elem_classes=["cmw-gradio-sidebar"],
                ):
                    sb.create_sidebar_column()

                with gr.Column(scale=1, min_width=0), gr.Tabs(selected="home"):
                    for tab_module in tab_modules:
                        if tab_module:
                            try:
                                tab_item, tab_components = tab_module.create_tab()
                                if tab_item is None:
                                    logging.getLogger(__name__).info(
                                        "⚠️ Skipping tab %s (create_tab returned None)",
                                        tab_module.__class__.__name__,
                                    )
                                    continue
                                self.components.update(tab_components)
                                self.components[
                                    f"{tab_module.__class__.__name__.lower()}_tab"
                                ] = tab_module
                                logging.getLogger(__name__).debug(
                                    "✅ Successfully created tab: %s",
                                    tab_module.__class__.__name__,
                                )
                            except Exception as e:
                                logging.getLogger(__name__).error(
                                    "❌ Error creating tab %s: %s",
                                    tab_module.__class__.__name__,
                                    e,
                                    exc_info=True,
                                )
                                raise

            self.components["sidebar_instance"] = sb
            if not config_tab_present:
                sb.mount_llm_selection_ui()
            self.components.update(sb.get_components())

            sb.ensure_llm_events_wired()

            # Connect DownloadsTab to update from chat history on demand.
            chat_tab_instance = self.components.get("chattab_tab")
            downloads_tab_instance = self.components.get("downloadstab_tab")
            if chat_tab_instance and downloads_tab_instance:
                download_btn = downloads_tab_instance.components.get("download_btn")
                download_html_btn = downloads_tab_instance.components.get(
                    "download_html_btn"
                )
                download_artifacts_zip_btn = downloads_tab_instance.components.get(
                    "download_artifacts_zip_btn"
                )
                if download_btn and download_html_btn and download_artifacts_zip_btn:
                    def _update_downloads_from_chat(
                        history: Any,
                        streaming_active: bool | None = None,
                        request: gr.Request | None = None,
                    ) -> tuple[Any, Any, Any]:
                        """Update download buttons from chat tab"""
                        if streaming_active:
                            return chat_tab_instance.get_download_cached_ui_updates()
                        return chat_tab_instance.get_download_button_updates(
                            history,
                            request,
                        )

                    downloads_tab_item = getattr(
                        downloads_tab_instance,
                        "_tab_item",
                        None,
                    )
                    if downloads_tab_item:
                        downloads_tab_item.select(
                            fn=_update_downloads_from_chat,
                            inputs=[
                                chat_tab_instance.components.get("chatbot"),
                                chat_tab_instance.components.get("streaming_active"),
                            ],
                            outputs=[
                                download_btn,
                                download_html_btn,
                                download_artifacts_zip_btn,
                            ],
                            queue=True,
                            show_progress="hidden",
                            api_visibility="private",
                        )
                    if (
                        get_ui_download_prep_after_stream()
                        and hasattr(chat_tab_instance, "streaming_event")
                        and chat_tab_instance.streaming_event
                    ):
                        chat_tab_instance.streaming_event.then(
                            fn=_update_downloads_from_chat,
                            inputs=[
                                chat_tab_instance.components.get("chatbot"),
                                chat_tab_instance.components.get("streaming_active"),
                            ],
                            outputs=[
                                download_btn,
                                download_html_btn,
                                download_artifacts_zip_btn,
                            ],
                            api_visibility="private",
                        )
                    # Also wire clear event to hide download buttons
                    if (
                        hasattr(chat_tab_instance, "clear_event")
                        and chat_tab_instance.clear_event
                    ):

                        def _hide_downloads_on_clear():
                            """Disable download buttons when chat is cleared"""
                            return (
                                gr.update(value=None, visible=True, interactive=False),
                                gr.update(value=None, visible=True, interactive=False),
                                gr.update(value=None, visible=True, interactive=False),
                            )

                        chat_tab_instance.clear_event.then(
                            fn=_hide_downloads_on_clear,
                            outputs=[
                                download_btn,
                                download_html_btn,
                                download_artifacts_zip_btn,
                            ],
                            api_visibility="private",
                        )
                    logging.getLogger(__name__).info(
                        "✅ Connected DownloadsTab automatic download preparation"
                    )

            # Wire end-of-turn event-driven refresh using existing chat events
            try:
                update_all_ui_handler = event_handlers.get("update_all_ui")
                stats_comp = self.components.get("stats_display")
                logs_comp = self.components.get("logs_display")
                token_budget_comp = self.components.get("token_budget_display")
                update_token_budget_handler = event_handlers.get("update_token_budget")

                chat_tab_instance = self.components.get("chattab_tab")
                if (
                    update_all_ui_handler
                    and stats_comp
                    and logs_comp
                    and chat_tab_instance
                ):
                    refresh_outputs = [stats_comp, stats_comp, stats_comp, logs_comp]

                    # After send (streaming) completes
                    if (
                        hasattr(chat_tab_instance, "streaming_event")
                        and chat_tab_instance.streaming_event
                    ):
                        chat_tab_instance.streaming_event.then(
                            fn=update_all_ui_handler,
                            outputs=refresh_outputs,
                            api_visibility="private",
                        )

                    # After submit completes
                    if (
                        hasattr(chat_tab_instance, "submit_event")
                        and chat_tab_instance.submit_event
                    ):
                        chat_tab_instance.submit_event.then(
                            fn=update_all_ui_handler,
                            outputs=refresh_outputs,
                            api_visibility="private",
                        )

                    # Wire clear event to update stats/progress
                    if (
                        hasattr(chat_tab_instance, "clear_event")
                        and chat_tab_instance.clear_event
                    ):
                        chat_tab_instance.clear_event.then(
                            fn=update_all_ui_handler,
                            outputs=refresh_outputs,
                            queue=False,  # Don't queue UI updates
                            api_visibility="private",
                        )

                    # Wire stop event to update stats/progress
                    if (
                        hasattr(chat_tab_instance, "stop_event")
                        and chat_tab_instance.stop_event
                    ):
                        chat_tab_instance.stop_event.then(
                            fn=update_all_ui_handler,
                            outputs=refresh_outputs,
                            queue=False,  # Don't queue UI updates
                            api_visibility="private",
                        )

                    logging.getLogger(__name__).debug(
                        "✅ Event-driven UI refresh wired for end-of-turn updates, clear, and stop"
                    )

                # Token budget refresh: wire separately to avoid changing update_all_ui signature
                if (
                    update_token_budget_handler
                    and token_budget_comp
                    and chat_tab_instance
                ):
                    if (
                        hasattr(chat_tab_instance, "streaming_event")
                        and chat_tab_instance.streaming_event
                    ):
                        chat_tab_instance.streaming_event.then(
                            fn=update_token_budget_handler,
                            outputs=[token_budget_comp],
                            api_visibility="private",
                        )
                    if (
                        hasattr(chat_tab_instance, "submit_event")
                        and chat_tab_instance.submit_event
                    ):
                        chat_tab_instance.submit_event.then(
                            fn=update_token_budget_handler,
                            outputs=[token_budget_comp],
                            api_visibility="private",
                        )
                    # Wire clear button to update token budget immediately (event-driven)
                    # Chain to the existing clear button click event
                    if (
                        hasattr(chat_tab_instance, "clear_event")
                        and chat_tab_instance.clear_event
                    ):
                        chat_tab_instance.clear_event.then(
                            fn=update_token_budget_handler,
                            outputs=[token_budget_comp],
                            api_visibility="private",
                        )
                    # Wire stop button to update token budget immediately (event-driven)
                    # Chain to the existing stop button click event
                    if (
                        hasattr(chat_tab_instance, "stop_event")
                        and chat_tab_instance.stop_event
                    ):
                        chat_tab_instance.stop_event.then(
                            fn=update_token_budget_handler,
                            outputs=[token_budget_comp],
                            api_visibility="private",
                        )
                    logging.getLogger(__name__).debug(
                        "✅ Token budget event-driven refresh wired for end-of-turn updates, clear button, and stop button"
                    )

                    provider_sel = self.components.get("provider_model_selector")
                    sync_dd = event_handlers.get("sync_llm_dropdown_from_session")
                    if provider_sel and sync_dd and main_app:
                        demo.load(
                            fn=sync_dd, outputs=[provider_sel], api_visibility="private"
                        )
                        if (
                            hasattr(chat_tab_instance, "clear_event")
                            and chat_tab_instance.clear_event
                        ):
                            chat_tab_instance.clear_event.then(
                                fn=sync_dd,
                                outputs=[provider_sel],
                                api_visibility="private",
                            )
                        logging.getLogger(__name__).debug(
                            "✅ LLM dropdown synced from session on load and after clear"
                        )
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Could not wire event-driven refresh: {e}"
                )

            # Setup auto-refresh timers
            self._setup_auto_refresh(demo, event_handlers)

        logging.getLogger(__name__).info(
            "✅ UIManager: Interface created successfully with all components and timers"
        )
        return demo

    def _setup_auto_refresh(self, demo: gr.Blocks, event_handlers: dict[str, Callable]):
        """Setup auto-refresh timers for status and logs - matches original behavior exactly"""
        # Get handlers with validation
        update_token_budget_handler = event_handlers.get("update_token_budget")
        refresh_logs_handler = event_handlers.get("refresh_logs")
        update_progress_handler = event_handlers.get("update_progress_display")

        # Single composite startup — all initial-state loads fire in one queue event
        refresh_stats_handler = event_handlers.get("refresh_stats")
        progress_comp = self.components.get("progress_display")
        _startup_fns: list[Callable] = []
        _startup_outputs: list = []

        if "token_budget_display" in self.components and update_token_budget_handler:
            _startup_fns.append(update_token_budget_handler)
            _startup_outputs.append(self.components["token_budget_display"])

        if "logs_display" in self.components and refresh_logs_handler:
            _startup_fns.append(refresh_logs_handler)
            _startup_outputs.append(self.components["logs_display"])

        if progress_comp and update_progress_handler:
            _startup_fns.append(update_progress_handler)
            _startup_outputs.append(progress_comp)

        if "stats_display" in self.components and refresh_stats_handler:
            _startup_fns.append(refresh_stats_handler)
            _startup_outputs.append(self.components["stats_display"])

        if _startup_fns:
            _fns = tuple(_startup_fns)

            def _startup_composite():
                return tuple(fn() for fn in _fns)

            demo.load(
                fn=_startup_composite,
                outputs=_startup_outputs,
                api_visibility="private",
            )

        # Setup auto-refresh timers for real-time updates
        self._setup_auto_refresh_timers(demo, event_handlers)

    def _setup_auto_refresh_timers(
        self, demo: gr.Blocks, event_handlers: dict[str, Callable]
    ):
        """Setup auto-refresh timers for real-time updates (single interval)"""
        logging.getLogger(__name__).info("🔄 Setting up auto-refresh timers...")

        # Single interval from central configuration
        refresh_interval = get_refresh_intervals().interval

        # Status / stats updates (single markdown on Statistics tab)
        stats_tick = self.components.get("stats_display")
        if stats_tick and event_handlers.get("update_status"):
            status_timer = gr.Timer(refresh_interval, active=True)
            status_timer.tick(
                fn=event_handlers["update_status"],
                outputs=[stats_tick],
                api_visibility="private",
            )
            logging.getLogger(__name__).debug(
                "✅ Stats display auto-refresh timer set (%ss)", refresh_interval
            )

        # Token budget updates - hybrid approach (immediate events + timer fallback)
        # Token budget is updated through:
        # - Immediate updates when budget_update events are emitted during streaming (pre-iteration, post-tool)
        # - End-of-turn events (streaming_event.then(), submit_event.then(), clear_event.then(), stop_event.then(), model_switch_event.then())
        # - Timer fallback (for edge cases where events might be missed)
        # This matches Gradio 5 behavior where token budget updates immediately when budget snapshots are computed
        # Budget snapshots are computed at "budget moments" (pre-iteration, post-tool), not on every chunk,
        # so immediate updates are efficient and provide real-time feedback
        if "token_budget_display" in self.components and event_handlers.get(
            "update_token_budget"
        ):
            # Timer serves as fallback for edge cases, but primary updates happen immediately via budget_update events
            token_budget_timer = gr.Timer(refresh_interval, active=True)
            token_budget_timer.tick(
                fn=event_handlers["update_token_budget"],
                outputs=[self.components["token_budget_display"]],
                api_visibility="private",
            )
            logging.getLogger(__name__).debug(
                f"✅ Token budget timer set ({refresh_interval}s) - fallback for edge cases, primary updates via budget_update events"
            )

        # LLM selection updates - no auto-refresh (explicit only)
        logging.getLogger(__name__).debug(
            "✅ LLM selection components will update only when explicitly triggered"
        )

        # Logs updates
        if "logs_display" in self.components and event_handlers.get("refresh_logs"):
            logs_timer = gr.Timer(refresh_interval, active=True)
            logs_timer.tick(
                fn=event_handlers["refresh_logs"],
                outputs=[self.components["logs_display"]],
                api_visibility="private",
            )
            logging.getLogger(__name__).debug(
                f"✅ Logs auto-refresh timer set ({refresh_interval}s)"
            )

        # Stats updates - REMOVED timer-based refresh
        # Stats is now updated through events only (preferred approach):
        # - After streaming completes (submit_event.then())
        # - After clear (clear_event.then())
        # - After stop (stop_event.then())
        # - On initial load (demo.load())
        # This ensures updates happen at appropriate events, not on a fixed timer
        logging.getLogger(__name__).debug(
            "✅ Stats uses event-driven updates only (no timer)"
        )

        # Progress/iteration updates - use timer for ticking clock and iteration display
        # Iterations are useful and informative, so we keep timer-based refresh for progress
        # Timer uses ITERATION_REFRESH_INTERVAL from .env (default 2.0s) for iteration display
        progress_comp = self.components.get("progress_display")
        update_progress_handler = event_handlers.get("update_progress_display")
        if progress_comp and update_progress_handler:
            # Use iteration interval from config (from .env ITERATION_REFRESH_INTERVAL)
            iteration_interval = get_refresh_intervals().iteration
            progress_timer = gr.Timer(iteration_interval, active=True)
            progress_timer.tick(
                fn=update_progress_handler,
                outputs=[progress_comp],
                api_visibility="private",
            )
            logging.getLogger(__name__).debug(
                f"✅ Progress/iteration timer set ({iteration_interval}s) - shows ticking clock and iterations"
            )

        logging.getLogger(__name__).info(
            "🔄 Auto-refresh timers configured successfully"
        )

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


def get_ui_manager(
    language: str = "en", i18n_instance: gr.I18n | None = None
) -> UIManager:
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
