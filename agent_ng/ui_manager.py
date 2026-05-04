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
    from agent_ng.agent_config import (
        get_refresh_intervals,
        get_ui_download_prep_after_stream,
        get_ui_export_html_after_turn,
    )
except ImportError:
    # Fallback for direct execution
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent))
    from agent_config import (  # type: ignore[no-redef]
        get_refresh_intervals,
        get_ui_download_prep_after_stream,
        get_ui_export_html_after_turn,
    )

class UIManager:
    """Manages Gradio UI creation and configuration with i18n support"""

    def __init__(self, language: str = "en", i18n_instance: gr.I18n | None = None):
        self.css_file_path = Path(__file__).parent.parent / "resources" / "css" / "cmw_copilot_theme.css"
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
            logging.getLogger(__name__).debug(f"Gradio static allowed paths: {os.environ['GRADIO_ALLOWED_PATHS']}")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not set GRADIO_ALLOWED_PATHS: {e}")

    def create_interface(
        self,
        tab_modules: list[Any],
        event_handlers: dict[str, Callable],
        main_app=None,
        *,
        include_sidebar_tab: bool = True,
        stack_home_chat: bool = False,
        disable_auto_timers: bool = False,
    ) -> gr.Blocks:
        # Store main_app reference for initialization completion checks
        self._main_app = main_app
        """
        Create the main Gradio interface using tab modules with i18n support.

        Args:
            tab_modules: List of tab module instances
            event_handlers: Dictionary of event handlers
            main_app: Application instance for tab callbacks
            include_sidebar_tab: When False, omit the settings/sidebar tab.
            stack_home_chat: When True, render ``tab_modules`` in one column via ``build_ui()`` (no ``gr.Tabs``).
            disable_auto_timers: When True, skip ``gr.Timer`` wiring (see ``CMW_UI_DISABLE_AUTO_TIMERS``).

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

            # Settings/sidebar tab (optional — omit via ``include_sidebar_tab=False`` / ``CMW_UI_TABS``)
            sidebar_instance = None
            if include_sidebar_tab:
                sidebar_instance = Sidebar(
                    event_handlers, language=self.language, i18n_instance=self.i18n
                )
                sidebar_instance.set_main_app(main_app)

            if stack_home_chat:
                logging.getLogger(__name__).info(
                    "📚 Stack layout: rendering modules without gr.Tabs (CMW_UI_STACK_HOME_CHAT)"
                )
                with gr.Column(elem_classes=["cmw-stack-layout"]):
                    for tab_module in tab_modules:
                        if not tab_module:
                            continue
                        if not hasattr(tab_module, "build_ui"):
                            msg = (
                                f"{tab_module.__class__.__name__} has no build_ui(); "
                                "narrow CMW_UI_TABS to tabs that support stack layout "
                                "or set CMW_UI_STACK_HOME_CHAT=false"
                            )
                            raise TypeError(msg)
                        try:
                            tab_module.build_ui(show_stack_heading=True)
                            self.components.update(tab_module.components)
                            key = f"{tab_module.__class__.__name__.lower()}_tab"
                            self.components[key] = tab_module
                            logging.getLogger(__name__).debug(
                                "✅ Stack section: %s", tab_module.__class__.__name__
                            )
                        except Exception as e:
                            logging.getLogger(__name__).error(
                                "❌ Stack layout failed for %s: %s",
                                tab_module.__class__.__name__,
                                e,
                                exc_info=True,
                            )
                            raise
                if sidebar_instance is not None:
                    logging.getLogger(__name__).warning(
                        "Sidebar instance exists but stack layout omits TabItem sidebar — "
                        "use include_sidebar_tab=False with CMW_UI_STACK_HOME_CHAT"
                    )
            else:
                # Do not pass ``selected=`` here: Gradio binds ``Tabs.selected`` two-way
                # (`js/tabs`). Forcing ``selected="home"`` while Home is already the first
                # registered tab correlated with hangs on the first navigation away from Home.
                with gr.Tabs():
                    # Create tabs using provided tab modules
                    for tab_module in tab_modules:
                        if tab_module:
                            try:
                                tab_item, tab_components = tab_module.create_tab()
                                # Skip if tab_item is None (e.g., ConfigTab when CMW_USE_DOTENV=true)
                                if tab_item is None:
                                    logging.getLogger(__name__).info(
                                        "⚠️ Skipping tab %s (create_tab returned None)",
                                        tab_module.__class__.__name__,
                                    )
                                    continue
                                # Consolidate all components in one place
                                self.components.update(tab_components)
                                # Store tab reference for later use
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

                    # Create sidebar as a tab (after other tabs)
                    if sidebar_instance is not None:
                        try:
                            sidebar_tab, sidebar_components = sidebar_instance.create_tab()
                            # Skip if sidebar_tab is None
                            if sidebar_tab is None:
                                logging.getLogger(__name__).warning(
                                    "⚠️ Sidebar create_tab returned None, skipping"
                                )
                            else:
                                # Consolidate sidebar components
                                self.components.update(sidebar_components)
                                self.components["sidebar_instance"] = sidebar_instance
                                logging.getLogger(__name__).debug(
                                    "✅ Successfully created sidebar tab"
                                )
                        except Exception as e:
                            logging.getLogger(__name__).error(
                                "❌ Error creating sidebar tab: %s",
                                e,
                                exc_info=True,
                            )
                            raise
                    else:
                        logging.getLogger(__name__).info(
                            "Sidebar/settings tab omitted (include_sidebar_tab=False)"
                        )

            # Connect quick action dropdown after all components are available
            if "sidebar_instance" in self.components:
                sidebar_instance = self.components["sidebar_instance"]
                sidebar_instance.connect_quick_action_dropdown()

            # Connect DownloadsTab to update from chat streaming events
            chat_tab_instance = self.components.get("chattab_tab")
            downloads_tab_instance = self.components.get("downloadstab_tab")
            if chat_tab_instance and downloads_tab_instance:
                download_btn = downloads_tab_instance.components.get("download_btn")
                download_html_btn = downloads_tab_instance.components.get("download_html_btn")
                if download_btn and download_html_btn:
                    prep_after_stream = get_ui_download_prep_after_stream()
                    chatbot_comp = chat_tab_instance.components.get("chatbot")
                    dl_tab = getattr(downloads_tab_instance, "_tab_item", None)

                    def _downloads_update(history, *, generate_html: bool):
                        """Update Markdown/HTML download buttons from chat history."""
                        return chat_tab_instance.get_download_button_updates(
                            history,
                            generate_html=generate_html,
                        )

                    def _downloads_on_tab_select(history):
                        # Match submit_tail HTML policy so tab switch does not hide HTML export.
                        return _downloads_update(
                            history,
                            generate_html=bool(get_ui_export_html_after_turn()),
                        )

                    def _downloads_after_chat_turn(history):
                        return _downloads_update(
                            history,
                            generate_html=bool(get_ui_export_html_after_turn()),
                        )

                    # After each completed turn: refresh buttons (HTML when env allows).
                    # Decoupled from CMW_UI_DOWNLOAD_PREP_AFTER_STREAM so HTML can appear without
                    # tab-select-only prep stalling other tabs.
                    if (
                        chatbot_comp is not None
                        and hasattr(chat_tab_instance, "submit_event")
                        and chat_tab_instance.submit_event
                    ):
                        chat_tab_instance.submit_event.then(
                            fn=_downloads_after_chat_turn,
                            inputs=[chatbot_comp],
                            outputs=[download_btn, download_html_btn],
                            queue=False,
                            api_visibility="private",
                        )
                    if prep_after_stream and chatbot_comp is not None:
                        if hasattr(chat_tab_instance, "streaming_event") and chat_tab_instance.streaming_event:
                            chat_tab_instance.streaming_event.then(
                                fn=_downloads_after_chat_turn,
                                inputs=[chatbot_comp],
                                outputs=[download_btn, download_html_btn],
                                queue=False,
                                api_visibility="private",
                            )
                    elif dl_tab is not None and chatbot_comp is not None:
                        dl_tab.select(
                            fn=_downloads_on_tab_select,
                            inputs=[chatbot_comp],
                            outputs=[download_btn, download_html_btn],
                            queue=False,
                        )
                    # Also wire clear event to hide download buttons
                    if hasattr(chat_tab_instance, "clear_event") and chat_tab_instance.clear_event:
                        def _hide_downloads_on_clear():
                            """Hide download buttons when chat is cleared"""
                            return gr.update(visible=False), gr.update(visible=False)
                        chat_tab_instance.clear_event.then(
                            fn=_hide_downloads_on_clear,
                            outputs=[download_btn, download_html_btn],
                        )
                    logging.getLogger(__name__).info("✅ Connected DownloadsTab to chat streaming events")

            # Wire end-of-turn event-driven refresh using existing chat events
            try:
                update_all_ui_handler = event_handlers.get("update_all_ui")
                status_comp = self.components.get("status_display")
                stats_comp = self.components.get("stats_display")
                logs_comp = self.components.get("logs_display")
                token_budget_comp = self.components.get("token_budget_display")
                update_token_budget_handler = event_handlers.get("update_token_budget")

                chat_tab_instance = self.components.get("chattab_tab")

                # Single .then per trigger so Gradio passes gr.Request once; avoids
                # queue=False tails where Request is missing and token sidebar stalls.
                if (
                    update_all_ui_handler
                    and update_token_budget_handler
                    and status_comp
                    and stats_comp
                    and logs_comp
                    and token_budget_comp
                    and chat_tab_instance
                ):
                    refresh_outputs_all = [
                        status_comp,
                        stats_comp,
                        logs_comp,
                        token_budget_comp,
                    ]

                    def _refresh_sidebar_after_turn(
                        request: gr.Request | None = None,
                    ) -> tuple[str, str, str, str]:
                        status_stats_logs = update_all_ui_handler(request)
                        tb = update_token_budget_handler(request)
                        return (*status_stats_logs, tb)

                    _ste = chat_tab_instance
                    _refresh_kw = {"queue": False, "api_visibility": "private"}
                    if hasattr(_ste, "streaming_event") and _ste.streaming_event:
                        _ste.streaming_event.then(
                            fn=_refresh_sidebar_after_turn,
                            outputs=refresh_outputs_all,
                            **_refresh_kw,
                        )
                    if hasattr(_ste, "submit_event") and _ste.submit_event:
                        _ste.submit_event.then(
                            fn=_refresh_sidebar_after_turn,
                            outputs=refresh_outputs_all,
                            **_refresh_kw,
                        )
                    if hasattr(_ste, "clear_event") and _ste.clear_event:
                        _ste.clear_event.then(
                            fn=_refresh_sidebar_after_turn,
                            outputs=refresh_outputs_all,
                            **_refresh_kw,
                        )
                    if hasattr(_ste, "stop_event") and _ste.stop_event:
                        _ste.stop_event.then(
                            fn=_refresh_sidebar_after_turn,
                            outputs=refresh_outputs_all,
                            **_refresh_kw,
                        )

                    logging.getLogger(__name__).debug(
                        "✅ Merged sidebar refresh (status/stats/logs + token budget) "
                        "wired for stream/submit/clear/stop"
                    )

                    provider_sel = self.components.get("provider_model_selector")
                    sync_dd = event_handlers.get("sync_llm_dropdown_from_session")
                    if provider_sel and sync_dd and main_app:
                        demo.load(fn=sync_dd, outputs=[provider_sel])
                        if (
                            hasattr(chat_tab_instance, "clear_event")
                            and chat_tab_instance.clear_event
                        ):
                            chat_tab_instance.clear_event.then(
                                fn=sync_dd,
                                outputs=[provider_sel],
                            )
                        logging.getLogger(__name__).debug(
                            "✅ LLM dropdown synced from session on load and after clear"
                        )
            except Exception as e:
                logging.getLogger(__name__).warning(f"Could not wire event-driven refresh: {e}")

            # Setup auto-refresh timers
            self._setup_auto_refresh(
                demo, event_handlers, disable_auto_timers=disable_auto_timers
            )

        logging.getLogger(__name__).info("✅ UIManager: Interface created successfully with all components and timers")
        return demo

    def _setup_auto_refresh(
        self,
        demo: gr.Blocks,
        event_handlers: dict[str, Callable],
        *,
        disable_auto_timers: bool = False,
    ):
        """Setup auto-refresh timers for status and logs - matches original behavior exactly"""
        # Get handlers with validation
        update_status_handler = event_handlers.get("update_status")
        update_token_budget_handler = event_handlers.get("update_token_budget")
        refresh_logs_handler = event_handlers.get("refresh_logs")
        update_progress_handler = event_handlers.get("update_progress_display")

        refresh_stats_handler = event_handlers.get("refresh_stats")

        # Single batched demo.load reduces concurrent outbound UI updates at startup
        # (Gradio 6.x: many parallel loads can saturate the browser main thread).
        boot_specs: list[tuple[str, Any, Callable[..., Any]]] = []

        def _boot_add(comp_key: str, fn: Callable[..., Any] | None) -> None:
            if fn is None:
                return
            comp = self.components.get(comp_key)
            if comp is not None:
                boot_specs.append((comp_key, comp, fn))

        _boot_add("status_display", update_status_handler)
        _boot_add("token_budget_display", update_token_budget_handler)
        _boot_add("logs_display", refresh_logs_handler)

        progress_comp = self.components.get("progress_display")
        if progress_comp is not None and update_progress_handler:
            boot_specs.append(("progress_display", progress_comp, update_progress_handler))

        _boot_add("stats_display", refresh_stats_handler)

        sidebar_inst = self.components.get("sidebar_instance")
        fb_sel = self.components.get("fallback_model_selector")
        if (
            sidebar_inst is not None
            and fb_sel is not None
            and hasattr(sidebar_inst, "hydrate_fallback_dropdown_on_load")
        ):
            boot_specs.append(
                (
                    "fallback_model_selector",
                    fb_sel,
                    sidebar_inst.hydrate_fallback_dropdown_on_load,
                )
            )

        if boot_specs:

            def _bootstrap_sidebar_batch(request: gr.Request | None = None):
                return tuple(fn(request) for _k, _c, fn in boot_specs)

            demo.load(
                fn=_bootstrap_sidebar_batch,
                outputs=[c for _k, c, _fn in boot_specs],
            )

        # Config auto-load is handled by tab.select in config_tab.py.
        # No demo.load() needed — config loads when user opens the tab.

        # Setup auto-refresh timers for real-time updates
        self._setup_auto_refresh_timers(
            demo, event_handlers, disable_auto_timers=disable_auto_timers
        )

    def _setup_auto_refresh_timers(
        self,
        demo: gr.Blocks,
        event_handlers: dict[str, Callable],
        *,
        disable_auto_timers: bool = False,
    ):
        """Setup auto-refresh timers for real-time updates (single interval)"""
        if disable_auto_timers:
            logging.getLogger(__name__).info(
                "⏹️ Auto-refresh timers skipped (CMW_UI_DISABLE_AUTO_TIMERS)"
            )
            return
        logging.getLogger(__name__).info("🔄 Setting up auto-refresh timers...")

        # Single interval from central configuration
        refresh_interval = get_refresh_intervals().interval

        # One timer tick for status + token budget + logs (same interval) cuts parallel SSE vs three timers.
        sidebar_tick_fns: list[Callable[..., Any]] = []
        sidebar_tick_outputs: list[Any] = []

        def _append_sidebar_tick(comp_key: str, fn_key: str) -> None:
            comp = self.components.get(comp_key)
            fn = event_handlers.get(fn_key)
            if comp is not None and fn:
                sidebar_tick_outputs.append(comp)
                sidebar_tick_fns.append(fn)

        _append_sidebar_tick("status_display", "update_status")
        _append_sidebar_tick("token_budget_display", "update_token_budget")
        _append_sidebar_tick("logs_display", "refresh_logs")

        if sidebar_tick_fns:

            def _sidebar_metrics_tick(request: gr.Request | None = None):
                return tuple(f(request) for f in sidebar_tick_fns)

            sidebar_metrics_timer = gr.Timer(refresh_interval, active=True)
            sidebar_metrics_timer.tick(
                fn=_sidebar_metrics_tick,
                outputs=sidebar_tick_outputs,
            )
            logging.getLogger(__name__).debug(
                "✅ Unified sidebar metrics timer (%ss): status + token_budget + logs",
                refresh_interval,
            )

        # LLM selection updates - no auto-refresh (explicit only)
        logging.getLogger(__name__).debug("✅ LLM selection components will update only when explicitly triggered")

        # Stats updates - REMOVED timer-based refresh
        # Stats is now updated through events only (preferred approach):
        # - After streaming completes (submit_event.then())
        # - After clear (clear_event.then())
        # - After stop (stop_event.then())
        # - On initial load (demo.load())
        # This ensures updates happen at appropriate events, not on a fixed timer
        logging.getLogger(__name__).debug("✅ Stats uses event-driven updates only (no timer)")

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
                outputs=[progress_comp]
            )
            logging.getLogger(__name__).debug(f"✅ Progress/iteration timer set ({iteration_interval}s) - shows ticking clock and iterations")

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
