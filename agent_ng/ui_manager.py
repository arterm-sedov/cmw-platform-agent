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

    def create_interface(self, tab_modules: list[Any], event_handlers: dict[str, Callable], main_app=None) -> gr.Blocks:
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

            # Create sidebar as a tab (converted from sidebar component)
            sidebar_instance = Sidebar(event_handlers, language=self.language, i18n_instance=self.i18n)
            sidebar_instance.set_main_app(main_app)  # Pass main app reference

            with gr.Tabs():
                # Create tabs using provided tab modules
                for tab_module in tab_modules:
                    if tab_module:
                        try:
                            tab_item, tab_components = tab_module.create_tab()
                            # Skip if tab_item is None (e.g., ConfigTab when CMW_USE_DOTENV=true)
                            if tab_item is None:
                                logging.getLogger(__name__).info(
                                    f"⚠️ Skipping tab {tab_module.__class__.__name__} (create_tab returned None)"
                                )
                                continue
                            # Consolidate all components in one place
                            self.components.update(tab_components)
                            # Store tab reference for later use
                            self.components[f"{tab_module.__class__.__name__.lower()}_tab"] = tab_module
                            logging.getLogger(__name__).debug(
                                f"✅ Successfully created tab: {tab_module.__class__.__name__}"
                            )
                        except Exception as e:
                            logging.getLogger(__name__).error(
                                f"❌ Error creating tab {tab_module.__class__.__name__}: {e}",
                                exc_info=True
                            )
                            raise

                # Create sidebar as a tab (after other tabs)
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
                        logging.getLogger(__name__).debug("✅ Successfully created sidebar tab")
                except Exception as e:
                    logging.getLogger(__name__).error(
                        f"❌ Error creating sidebar tab: {e}",
                        exc_info=True
                    )
                    raise

            # Connect quick action dropdown after all components are available
            if "sidebar_instance" in self.components:
                sidebar_instance = self.components["sidebar_instance"]
                sidebar_instance.connect_quick_action_dropdown()

            # Connect DownloadsTab to update from chat streaming events
            # #region agent log
            import json, time
            try:
                with open(r'd:\Repo\cmw-platform-agent-gradio-6\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"init","hypothesisId":"E","location":"ui_manager.py:110","message":"Looking for tab instances","data":{"component_keys":list(self.components.keys())},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
            chat_tab_instance = self.components.get("chattab_tab")
            downloads_tab_instance = self.components.get("downloadstab_tab")
            # #region agent log
            try:
                with open(r'd:\Repo\cmw-platform-agent-gradio-6\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"init","hypothesisId":"F","location":"ui_manager.py:112","message":"Tab instances found","data":{"has_chat":chat_tab_instance is not None,"has_downloads":downloads_tab_instance is not None},"timestamp":int(time.time()*1000)}) + '\n')
            except: pass
            # #endregion
            if chat_tab_instance and downloads_tab_instance:
                download_btn = downloads_tab_instance.components.get("download_btn")
                download_html_btn = downloads_tab_instance.components.get("download_html_btn")
                # #region agent log
                try:
                    with open(r'd:\Repo\cmw-platform-agent-gradio-6\.cursor\debug.log', 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"init","hypothesisId":"G","location":"ui_manager.py:115","message":"Download buttons found","data":{"has_download_btn":download_btn is not None,"has_download_html_btn":download_html_btn is not None},"timestamp":int(time.time()*1000)}) + '\n')
                except: pass
                # #endregion
                if download_btn and download_html_btn:
                    # Wire download buttons to update after streaming completes
                    def _update_downloads_from_chat(history):
                        """Update download buttons from chat tab"""
                        return chat_tab_instance.get_download_button_updates(history)

                    # #region agent log
                    try:
                        with open(r'd:\Repo\cmw-platform-agent-gradio-6\.cursor\debug.log', 'a', encoding='utf-8') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"init","hypothesisId":"H","location":"ui_manager.py:122","message":"Wiring download events","data":{"has_streaming_event":hasattr(chat_tab_instance, "streaming_event"),"has_submit_event":hasattr(chat_tab_instance, "submit_event")},"timestamp":int(time.time()*1000)}) + '\n')
                    except: pass
                    # #endregion
                    if hasattr(chat_tab_instance, "streaming_event") and chat_tab_instance.streaming_event:
                        chat_tab_instance.streaming_event.then(
                            fn=_update_downloads_from_chat,
                            inputs=[chat_tab_instance.components.get("chatbot")],
                            outputs=[download_btn, download_html_btn],
                        )
                    if hasattr(chat_tab_instance, "submit_event") and chat_tab_instance.submit_event:
                        chat_tab_instance.submit_event.then(
                            fn=_update_downloads_from_chat,
                            inputs=[chat_tab_instance.components.get("chatbot")],
                            outputs=[download_btn, download_html_btn],
                            queue=False,  # Don't queue file generation to prevent blocking
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
                    # #region agent log
                    try:
                        with open(r'd:\Repo\cmw-platform-agent-gradio-6\.cursor\debug.log', 'a', encoding='utf-8') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"init","hypothesisId":"I","location":"ui_manager.py:142","message":"Download events wired successfully","data":{},"timestamp":int(time.time()*1000)}) + '\n')
                    except: pass
                    # #endregion
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

                    # Wire clear event to update stats/progress
                    if hasattr(chat_tab_instance, "clear_event") and chat_tab_instance.clear_event:
                        chat_tab_instance.clear_event.then(
                            fn=update_all_ui_handler,
                            outputs=refresh_outputs,
                            queue=False,  # Don't queue UI updates
                        )

                    # Wire stop event to update stats/progress
                    if hasattr(chat_tab_instance, "stop_event") and chat_tab_instance.stop_event:
                        chat_tab_instance.stop_event.then(
                            fn=update_all_ui_handler,
                            outputs=refresh_outputs,
                            queue=False,  # Don't queue UI updates
                        )

                    logging.getLogger(__name__).debug("✅ Event-driven UI refresh wired for end-of-turn updates, clear, and stop")

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

        # Progress display - wire to chat events for event-driven updates
        progress_comp = self.components.get("progress_display")
        if progress_comp and update_progress_handler:
            # Load initial state
            demo.load(
                fn=update_progress_handler,
                outputs=[progress_comp]
            )
            # Note: Progress updates are wired to chat events in the main event wiring section
            # (see submit_event, clear_event, stop_event wiring above)

        refresh_stats_handler = event_handlers.get("refresh_stats")
        if "stats_display" in self.components and refresh_stats_handler:
            demo.load(
                fn=refresh_stats_handler,
                outputs=[self.components["stats_display"]]
            )

        # Wire initialization completion to update UI components
        # This is lean and timely - updates happen exactly when initialization completes
        self._wire_initialization_completion_updates(demo, event_handlers)

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

        # Token budget updates - hybrid approach (immediate events + timer fallback)
        # Token budget is updated through:
        # - Immediate updates when budget_update events are emitted during streaming (pre-iteration, post-tool)
        # - End-of-turn events (streaming_event.then(), submit_event.then(), clear_event.then(), stop_event.then(), model_switch_event.then())
        # - Timer fallback (for edge cases where events might be missed)
        # This matches Gradio 5 behavior where token budget updates immediately when budget snapshots are computed
        # Budget snapshots are computed at "budget moments" (pre-iteration, post-tool), not on every chunk,
        # so immediate updates are efficient and provide real-time feedback
        if "token_budget_display" in self.components and event_handlers.get("update_token_budget"):
            # Timer serves as fallback for edge cases, but primary updates happen immediately via budget_update events
            token_budget_timer = gr.Timer(refresh_interval, active=True)
            token_budget_timer.tick(
                fn=event_handlers["update_token_budget"],
                outputs=[self.components["token_budget_display"]]
            )
            logging.getLogger(__name__).debug(f"✅ Token budget timer set ({refresh_interval}s) - fallback for edge cases, primary updates via budget_update events")

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

    def _wire_initialization_completion_updates(self, demo: gr.Blocks, event_handlers: dict[str, Callable]):
        """Wire initialization completion to trigger UI updates - lean and timely"""
        update_all_ui_handler = event_handlers.get("update_all_ui")
        status_comp = self.components.get("status_display")
        stats_comp = self.components.get("stats_display")
        logs_comp = self.components.get("logs_display")
        token_budget_comp = self.components.get("token_budget_display")
        update_token_budget_handler = event_handlers.get("update_token_budget")
        update_progress_handler = event_handlers.get("update_progress_display")
        progress_comp = self.components.get("progress_display")

        # Create a hidden state to track initialization status
        # When initialization completes, we'll update this state which triggers UI updates
        init_state = gr.State(value=False)  # False = not ready, True = ready
        self.components["_init_state"] = init_state

        # Function to check initialization and update UI when ready
        def check_initialization_and_update(init_ready: bool, request: gr.Request = None) -> tuple[bool, str, str, str, str, str]:
            """Check if initialization is complete and update UI components - session-aware"""
            main_app = getattr(self, "_main_app", None)
            is_ready = main_app.is_ready() if main_app and hasattr(main_app, "is_ready") else False

            # If initialization just completed, update all UI components
            # Pass request for session isolation
            if is_ready and not init_ready:
                if update_all_ui_handler and status_comp and stats_comp and logs_comp:
                    # update_all_ui_components() now accepts request for session isolation
                    status, stats, logs = update_all_ui_handler(request)
                    # Other handlers also accept request for session isolation
                    token_budget = update_token_budget_handler(request) if (update_token_budget_handler and token_budget_comp) else ""
                    progress = update_progress_handler(request) if (update_progress_handler and progress_comp) else ""
                    return True, status, stats, logs, token_budget, progress

            # Return current values if not ready yet
            return (
                is_ready,
                status_comp.value if status_comp else "",
                stats_comp.value if stats_comp else "",
                logs_comp.value if logs_comp else "",
                token_budget_comp.value if token_budget_comp else "",
                progress_comp.value if progress_comp else ""
            )

        # Use a short-lived timer that checks initialization status
        # Timer stops automatically once initialization completes (no permanent timer)
        init_check_timer = gr.Timer(0.5, active=True)

        outputs = [init_state]
        if status_comp and stats_comp and logs_comp:
            outputs.extend([status_comp, stats_comp, logs_comp])
        if token_budget_comp:
            outputs.append(token_budget_comp)
        if progress_comp:
            outputs.append(progress_comp)

        def timer_check(init_ready: bool, request: gr.Request = None) -> tuple:
            """Timer function that checks initialization and stops when ready - session-aware"""
            # Pass request for session isolation
            result = check_initialization_and_update(init_ready, request)
            is_ready = result[0]
            # Stop timer when ready (return active=False)
            timer_update = gr.Timer(active=not is_ready)
            return (is_ready,) + result[1:] + (timer_update,)

        init_check_timer.tick(
            fn=timer_check,
            inputs=[init_state],
            outputs=outputs + [init_check_timer]
        )

        logging.getLogger(__name__).debug("✅ Initialization completion wired to update UI (timer stops when ready)")


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
