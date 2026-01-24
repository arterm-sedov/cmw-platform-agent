"""
Chat Tab Module for App NG
=========================

Handles the main chat interface, quick actions, and user interactions.
This module encapsulates all chat-related UI components and functionality.
Supports internationalization (i18n) with Russian and English translations.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable
from datetime import datetime
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Optional
import uuid

import gradio as gr
import markdown

from agent_ng.debug_streamer import get_debug_streamer
from agent_ng.i18n_translations import get_translation_key
from agent_ng.queue_manager import (
    apply_concurrency_to_click_event,
    apply_concurrency_to_submit_event,
)
from agent_ng.history_compression import (
    perform_compression_with_notifications,
    should_compress_on_completion,
)
from agent_ng.token_budget import (
    HISTORY_COMPRESSION_KEEP_RECENT_TURNS_MID_TURN,
    TOKEN_STATUS_CRITICAL,
    TOKEN_STATUS_CRITICAL_THRESHOLD,
    TOKEN_STATUS_MODERATE_THRESHOLD,
    TOKEN_STATUS_WARNING_THRESHOLD,
)
from tools.file_utils import FileUtils

from .sidebar import QuickActionsMixin


CHAT_DOWNLOADS_ENABLED = True  # Enable chat export/download functionality


class ChatTab(QuickActionsMixin):
    """Chat tab component with interface and quick actions"""

    def __init__(
        self,
        event_handlers: dict[str, Callable],
        language: str = "en",
        i18n_instance: gr.I18n | None = None,
    ) -> None:
        self.event_handlers = event_handlers
        self.components = {}
        self.main_app = None  # Reference to main app for progress status
        self.language = language
        self.i18n = i18n_instance

    def create_tab(self) -> tuple[gr.TabItem, dict[str, Any]]:
        """
        Create the chat tab with all its components.

        Returns:
            Tuple of (TabItem, components_dict)
        """
        logging.getLogger(__name__).info("✅ ChatTab: Creating chat interface...")

        try:
            with gr.TabItem(self._get_translation("tab_chat"), id="chat") as tab:
                # Create main chat interface (includes sidebar)
                self._create_chat_interface()

                # Connect event handlers
                self._connect_events()

                # Verify tab was created successfully
                if tab is None:
                    raise ValueError("gr.TabItem context manager returned None - this should not happen")
        except Exception as e:
            logging.getLogger(__name__).error(
                f"❌ ChatTab: Error in create_tab: {e}",
                exc_info=True
            )
            raise

        logging.getLogger(__name__).info(
            "✅ ChatTab: Successfully created with all components and event handlers"
        )
        return tab, self.components

    def _create_chat_interface(self):
        """Create the main chat interface with proper layout"""
        # Chat interface with metadata support for thinking transparency
        # In Gradio 6, Chatbot uses messages format by default, so no type parameter is needed
        self.components["chatbot"] = gr.Chatbot(
            label=self._get_translation("chat_label"),
            height=500,
            show_label=True,
            container=True,
            buttons=["copy", "copy_all"],
            elem_id="chatbot-main",
            elem_classes=["chatbot-card"],
        )

        with gr.Row():
            # Use built-in interchanging buttons in MultimodalTextbox (Gradio 6 pattern from reference repo)
            # Following reference repo: submit_btn and stop_btn interchange with nice icons
            # Start with submit_btn=True, stop_btn=False - submit button shows, stop button hidden
            self.components["msg"] = gr.MultimodalTextbox(
                label=self._get_translation("message_label"),
                placeholder=self._get_translation("message_placeholder"),
                lines=2,
                scale=4,
                max_lines=4,
                elem_id="message-input",
                elem_classes=["message-card"],
                submit_btn=True,  # Show submit button with icon (interchanges with stop)
                stop_btn=False,  # Start hidden, will be shown when streaming starts (interchanges with submit)
                file_types=[
                    ".pdf", ".csv", ".tsv", ".xlsx", ".xls",  # Documents and data
                    ".docx", ".pptx", ".vsdx", ".msg", ".eml",  # Office documents
                    ".zip", ".rar", ".tar", ".gz", ".bz2",  # Archives
                    ".dwg", ".bpmn", ".sql", ".conf", ".ico",  # Other supported formats
                    ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".xml", ".html",
                    ".css", ".md", ".ini", ".sh", ".bat", ".ps1", ".c", ".cpp", ".h",
                    ".hpp", ".java", ".go", ".rs", ".rb", ".php", ".pl", ".swift",
                    ".kt", ".scala", ".sql", ".toml", ".env",  # Common text-based code formats
                    ".wav", ".mp3",  ".aiff", ".ogg", ".flac", ".aac",  # Audio files
                    ".mp4", ".mpeg", ".mpg", ".mov", ".avi", ".flv", ".webm", ".wmv", ".3gp", ".3gpp",  # Video files
                    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg", ".tiff"  # Image files
                ],
                file_count="multiple",
            )
            with gr.Column():
                # REMOVED: Separate send button - now using built-in submit button in MultimodalTextbox
                # The built-in submit button interchanges with stop button (following reference repo pattern)
                self.components["clear_btn"] = gr.Button(
                    self._get_translation("clear_button"),
                    variant="secondary",
                    elem_classes=["cmw-button"],
                )

        # Welcome block moved to dedicated Home tab

    def _create_sidebar(self):
        """Create the status and quick actions sidebar - now handled in _create_chat_interface"""
        # This method is now empty as the sidebar is created within the chat interface

    def _connect_events(self):
        """Connect all event handlers for the chat tab with concurrency control"""
        logging.getLogger(__name__).debug(
            "🔗 ChatTab: Connecting event handlers with concurrency control..."
        )

        # Get critical event handlers
        stream_handler = self.event_handlers.get("stream_message")
        clear_handler = self.event_handlers.get("clear_chat")

        # Validate critical handlers
        stream_handler_msg = "stream_message handler not found in event_handlers"
        clear_handler_msg = "clear_chat handler not found in event_handlers"
        if not stream_handler:
            raise ValueError(stream_handler_msg)
        if not clear_handler:
            raise ValueError(clear_handler_msg)

        logging.getLogger(__name__).debug(
            "✅ ChatTab: Critical event handlers validated"
        )

        # Get queue manager for concurrency control
        queue_manager = None
        if hasattr(self, "main_app") and self.main_app:
            queue_manager = getattr(self.main_app, "queue_manager", None)
            logging.getLogger(__name__).debug(
                "ChatTab: Queue manager found: %s", queue_manager is not None
            )
            if queue_manager:
                logging.getLogger(__name__).debug(
                    "ChatTab: Queue manager has config: %s",
                    hasattr(queue_manager, "config"),
                )

        # Store original stop_btn value (True, but we start with False)
        # Following reference repo pattern: buttons interchange - submit hides when stop shows
        original_stop_btn = True

        # Following reference repo pattern: two-step process
        # Step 1: Simple function to clear textbox and save message (triggers .success())
        # Step 2: Chain streaming handler with .then()
        def clear_and_save_multimodal_textbox(multimodal_value: dict[str, Any] | None) -> tuple[gr.MultimodalTextbox, dict[str, Any] | None]:
            """Clear MultimodalTextbox and save message to state (pattern from reference repo)."""
            logging.getLogger(__name__).debug("clear_and_save_multimodal_textbox called")
            # Extract text from MultimodalValue format
            if isinstance(multimodal_value, dict):
                text = multimodal_value.get("text", "")
                files = multimodal_value.get("files", [])
                saved_value = {"text": text, "files": files}
            else:
                saved_value = {"text": str(multimodal_value) if multimodal_value else "", "files": []}

            logging.getLogger(__name__).debug(f"Saved value: text={saved_value.get('text', '')[:50]}..., files={len(saved_value.get('files', []))}")
            return (
                gr.MultimodalTextbox(value="", interactive=False, placeholder=""),
                saved_value,
            )

        # State to store saved message (pattern from reference repo)
        saved_input = gr.State()
        self.components["saved_input"] = saved_input  # Store for potential future use

        # Cancellation state - mutable dict so changes propagate to running generator
        # Following reference repo pattern: used for cooperative cancellation
        cancellation_state = gr.State(value={"cancelled": False})
        self.components["cancellation_state"] = cancellation_state

        # Step 1: Submit event - simple function that clears textbox
        # Following reference repo pattern: this triggers .success() which shows stop button
        user_submit = self.components["msg"].submit(
            fn=clear_and_save_multimodal_textbox,
            inputs=[self.components["msg"]],
            outputs=[self.components["msg"], saved_input],  # Clear textbox and save message to state
            queue=False,
        )

        # Show stop button when submit succeeds (before streaming starts)
        # Interchange: submit_btn=False, stop_btn=True (submit hides, stop shows)
        # Following reference repo pattern exactly
        def show_stop_button():
            """Show stop button and hide submit button (interchanging buttons)."""
            logging.getLogger(__name__).debug("show_stop_button: Interchanging buttons - showing stop, hiding submit")
            return gr.MultimodalTextbox(submit_btn=False, stop_btn=original_stop_btn)

        user_submit.success(
            fn=show_stop_button,
            outputs=[self.components["msg"]],
            queue=False,
        )

        # Reset cancellation state at start of new submission (following reference repo pattern)
        def reset_cancellation_state(cancel_state: dict | None) -> dict:
            """Reset cancellation state at start of new submission."""
            if cancel_state is None or not isinstance(cancel_state, dict):
                cancel_state = {"cancelled": False}
            else:
                cancel_state["cancelled"] = False
            return cancel_state

        # Step 2: Chain streaming handler from user_submit
        # Following reference repo pattern: reset cancellation state first, then chain streaming handler
        if queue_manager:
            # Apply concurrency settings to the chained streaming event
            streaming_config = apply_concurrency_to_click_event(
                queue_manager,
                "chat",
                self._stream_message_wrapper,
                [saved_input, self.components["chatbot"], cancellation_state],  # Add cancellation_state
                [
                    self.components["chatbot"],
                    self.components["msg"],
                    self._get_quick_actions_dropdown(),
                ],
            )
            # Remove 'fn' from config since we'll use it directly in .then()
            streaming_fn = streaming_config.pop("fn")
            streaming_inputs = streaming_config.pop("inputs")
            streaming_outputs = streaming_config.pop("outputs")

            # Chain streaming handler from user_submit
            # Following reference repo pattern: reset cancellation state first, then stream
            self.submit_event = user_submit.then(
                fn=reset_cancellation_state,
                inputs=[cancellation_state],
                outputs=[cancellation_state],
                queue=False,
            ).then(
                fn=streaming_fn,
                inputs=streaming_inputs,
                outputs=streaming_outputs,
                **streaming_config,
            )
        else:
            # Fallback to default behavior if queue manager not available
            logging.getLogger(__name__).warning(
                "⚠️ Queue manager not available - using default event configuration"
            )
            # Chain streaming handler from user_submit
            # Following reference repo pattern: reset cancellation state first, then stream
            self.submit_event = user_submit.then(
                fn=reset_cancellation_state,
                inputs=[cancellation_state],
                outputs=[cancellation_state],
                queue=False,
            ).then(
                fn=self._stream_message_wrapper,
                inputs=[saved_input, self.components["chatbot"], cancellation_state],  # Add cancellation_state
                outputs=[
                    self.components["chatbot"],
                    self.components["msg"],
                    self._get_quick_actions_dropdown(),
                ],
            )

        # Built-in stop button automatically cancels submit_event when stop_btn=True
        # Wire up the stop event to handle cancellation and update history
        # Following reference repo pattern: use msg.stop() instead of separate button
        # #region agent log
        import json
        try:
            with open(r"d:\Repo\cmw-platform-agent-gradio-6\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "stop-debug", "hypothesisId": "C", "location": "chat_tab.py:288", "message": "Wiring stop event", "data": {"has_submit_event": hasattr(self, "submit_event") and self.submit_event is not None}, "timestamp": __import__("time").time() * 1000}) + "\n")
        except: pass
        # #endregion
        try:
            self.stop_event = self.components["msg"].stop(
                fn=self._handle_stop_click,
                inputs=[self.components["chatbot"]],
                outputs=[
                    self.components["chatbot"],  # Only return chatbot, not msg
                ],
                cancels=[self.submit_event],  # Cancel submit event (following reference repo pattern)
            ).then(
                lambda: gr.MultimodalTextbox(submit_btn=True, stop_btn=False),  # Hide stop button after cancellation
                outputs=[self.components["msg"]],
                queue=False,
            )
            # #region agent log
            import json
            try:
                with open(r"d:\Repo\cmw-platform-agent-gradio-6\.cursor\debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "stop-debug", "hypothesisId": "C", "location": "chat_tab.py:300", "message": "Stop event wired successfully", "data": {}, "timestamp": __import__("time").time() * 1000}) + "\n")
            except: pass
            # #endregion
        except Exception as exc:
            # #region agent log
            import json
            try:
                with open(r"d:\Repo\cmw-platform-agent-gradio-6\.cursor\debug.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "stop-debug", "hypothesisId": "D", "location": "chat_tab.py:302", "message": "Stop event wiring failed", "data": {"error": str(exc), "error_type": type(exc).__name__}, "timestamp": __import__("time").time() * 1000}) + "\n")
            except: pass
            # #endregion
            logging.getLogger(__name__).error(f"Failed to wire stop event: {exc}", exc_info=True)
            raise

        # Wire UI update trigger to stop event (for token budget/status refresh)
        trigger_ui_update = self.event_handlers.get("trigger_ui_update")
        if trigger_ui_update:
            self.stop_event.then(fn=trigger_ui_update, outputs=[], queue=False)

        # Re-enable textbox and hide stop button after streaming completes
        # Following reference repo pattern: chain after handler completion
        def re_enable_textbox_and_hide_stop():
            """Re-enable textbox and hide stop button after handler completion."""
            logging.getLogger(__name__).debug("Re-enabling textbox and hiding stop button after handler completion")
            return gr.MultimodalTextbox(value="", interactive=True, submit_btn=True, stop_btn=False)

        # Chain re-enable to submit event after it completes
        # .then() fires after generator completes and all yields are processed
        # Following reference repo pattern: only chain to submit_event (no separate send button)
        self.submit_event = self.submit_event.then(
            fn=re_enable_textbox_and_hide_stop,
            outputs=[self.components["msg"]],
            queue=False,
        )

        # Handle chatbot clear event - also clear memory when chat is cleared
        # Following reference repo pattern: wire chatbot.clear() similar to regular clear button
        def handle_chatbot_clear(request: gr.Request | None = None) -> None:
            """Handle chatbot clear event - also clear memory when chat is cleared."""
            logging.getLogger(__name__).info("Chatbot clear event - clearing memory")
            # Use the same clear handler as the regular clear button
            clear_handler = self.event_handlers.get("clear_chat")
            if clear_handler:
                clear_handler(request)
            return None

        # Bind to the built-in clear button's clear event
        # Also clear memory when chat is cleared
        # Following reference repo pattern: wire chatbot.clear() similar to regular clear button
        self.components["chatbot"].clear(
            fn=handle_chatbot_clear,
            inputs=[],  # Request is automatically passed to functions that accept it
            outputs=[],  # No outputs needed
        )

        # Store clear button click event for chaining token budget updates
        # This is in addition to chatbot.clear() - both can work together
        self.clear_event = self.components["clear_btn"].click(
            fn=self._clear_chat_with_download_reset,
            outputs=[
                self.components["chatbot"],
                self.components["msg"],
            ],
        )

        # Download button uses pre-generated file - no click handler needed

        # Trigger UI updates after chat events
        self._setup_chat_event_triggers()

        # Note: Sidebar components (token_budget_display, quick_actions_dropdown, provider_model_selector, progress_display)
        # are now handled by the UI Manager and will be connected there

        logging.getLogger(__name__).debug(
            "✅ ChatTab: All event handlers connected successfully"
        )

    def _yield_ui_newline(self, history):
        """Return a UI-only assistant placeholder with a leading newline.

        This should not affect agent memory; it's purely for chat UI spacing.
        """
        ui_history = list(history) if history else []
        ui_history.append({"role": "assistant", "content": "\n"})
        return (
            ui_history,
            "",
            None,
        )

    def _setup_chat_event_triggers(self):
        """Setup event triggers to update other UI components when chat events occur"""
        # Get UI update handlers
        trigger_ui_update = self.event_handlers.get("trigger_ui_update")

        if trigger_ui_update:
            # Trigger UI update after message submit (built-in submit button)
            self.components["msg"].submit(fn=trigger_ui_update, outputs=[])

            # Trigger UI update after clear button click
            self.components["clear_btn"].click(fn=trigger_ui_update, outputs=[])

            # Trigger UI update after built-in stop button click (to refresh token budget/status)
            # Use msg.stop() to wire to built-in stop button
            if hasattr(self, "stop_event") and self.stop_event:
                self.stop_event.then(fn=trigger_ui_update, outputs=[], queue=False)

            logging.getLogger(__name__).debug(
                "✅ ChatTab: UI update triggers connected"
            )

    def get_components(self) -> dict[str, Any]:
        """Get all components created by this tab"""
        return self.components

    def get_status_component(self) -> gr.Markdown:
        """Get the status display component for auto-refresh - now handled by UI Manager"""
        # Status display is now in the UI Manager sidebar
        return None

    def get_message_component(self) -> gr.MultimodalTextbox:
        """Get the message input component for quick actions"""
        return self.components["msg"]

    def set_main_app(self, app):
        """Set reference to main app for accessing progress status"""
        self.main_app = app

    def get_progress_display(self) -> gr.Markdown:
        """Get the progress display component - now handled by UI Manager"""
        # These components are now in the UI Manager sidebar
        return None

    def get_token_budget_display(self) -> gr.Markdown:
        """Get the token budget display component - now handled by UI Manager"""
        # These components are now in the UI Manager sidebar
        return None

    def get_llm_selection_components(self) -> dict[str, Any]:
        """Get LLM selection components for UI updates - now handled by UI Manager"""
        # These components are now in the UI Manager sidebar
        return {}

    def get_stop_button(self) -> gr.MultimodalTextbox:
        """Get the message input component (stop button is now built-in)"""
        # Stop button is now built-in to MultimodalTextbox, return the textbox component
        return self.components["msg"]

    def _get_quick_actions_dropdown(self) -> gr.Dropdown:
        """Get the quick actions dropdown from the sidebar"""
        # The dropdown is now in the sidebar, so we need to get it from the main app
        if (
            hasattr(self, "main_app")
            and self.main_app
            and hasattr(self.main_app, "ui_manager")
            and self.main_app.ui_manager
        ):
            # Try to get from UI Manager components
            components = self.main_app.ui_manager.get_components()
            dropdown = components.get("quick_actions_dropdown")
            # Ensure we always return a valid component, never None
            if dropdown is not None:
                return dropdown

        # Fallback - return a dummy component that won't cause errors
        # This ensures we never return None, which would cause '_id' attribute errors
        return gr.Dropdown(visible=False)

    def _handle_stop_click(
        self, history: list[dict[str, str]], cancel_state: dict | None = None, request: gr.Request | None = None
    ) -> tuple[list[dict[str, str]], dict]:
        """Handle built-in stop button click: set cancellation flag, finalize token tracking, append stats, update UI.

        Following reference repo pattern: uses msg.stop() with built-in stop button.
        Sets cancellation flag in shared state so the running generator can check it.
        Returns history and cancellation_state - MultimodalTextbox update is handled in .then() chain.
        """
        # Set cancellation flag (following reference repo pattern)
        if cancel_state is None or not isinstance(cancel_state, dict):
            cancel_state = {"cancelled": True}
        else:
            cancel_state["cancelled"] = True
        logging.getLogger(__name__).info("Stop button clicked - setting cancellation flag")
        # #region agent log
        import json
        try:
            with open(r"d:\Repo\cmw-platform-agent-gradio-6\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "stop-debug", "hypothesisId": "A", "location": "chat_tab.py:455", "message": "_handle_stop_click called", "data": {"history_len": len(history) if history else 0, "has_request": request is not None}, "timestamp": __import__("time").time() * 1000}) + "\n")
        except: pass
        # #endregion
        try:
            # Attempt to finalize token accounting for this turn even if stream was interrupted
            if (
                hasattr(self, "main_app")
                and self.main_app
                and hasattr(self.main_app, "session_manager")
            ):
                session_id = (
                    self.main_app.session_manager.get_session_id(request)
                    if request
                    else "default"
                )
                agent = self.main_app.session_manager.get_session_agent(session_id)
                if agent and hasattr(agent, "token_tracker"):
                    # Get current request messages for estimation fallback
                    try:
                        messages = agent.get_conversation_history(session_id)
                    except Exception:
                        messages = []

                    # Update prompt tokens explicitly
                    try:
                        if messages:
                            agent.token_tracker.count_prompt_tokens(messages)
                    except Exception as exc:
                        logging.getLogger(__name__).debug(
                            "Failed to count prompt tokens on stop: %s", exc
                        )

                    # Finalize turn token usage using monotonic estimate (no API usage needed).
                    # IMPORTANT: Do NOT call track_llm_response(None, ...) as it can overwrite
                    # per-turn totals with a smaller "current request only" estimate.
                    try:
                        # Refresh snapshot to feed the per-turn estimate (best-effort).
                        try:
                            agent.token_tracker.refresh_budget_snapshot(
                                agent=agent,
                                conversation_id=session_id,
                                messages_override=messages,
                            )
                        except Exception as snap_exc:
                            logging.getLogger(__name__).debug(
                                "Failed to refresh snapshot on stop: %s", snap_exc
                            )
                        agent.token_tracker.finalize_turn_usage(None, messages)
                    except Exception as exc:
                        logging.getLogger(__name__).debug(
                            "Failed to finalize turn usage on stop: %s", exc
                        )

                    # Build a stats block and append as assistant meta message
                    try:
                        prompt_tokens = agent.token_tracker.get_last_prompt_tokens()
                        api_tokens = agent.token_tracker.get_last_api_tokens()

                        stats_lines = []
                        if prompt_tokens:
                            stats_lines.append(
                                self._get_translation("prompt_tokens").format(
                                    tokens=prompt_tokens.formatted
                                )
                            )
                        if api_tokens:
                            stats_lines.append(
                                self._get_translation("api_tokens").format(
                                    tokens=api_tokens.formatted
                                )
                            )
                        # Provider/model and execution time where possible
                        provider = "unknown"
                        model = "unknown"
                        try:
                            if getattr(agent, "llm_instance", None):
                                provider = agent.llm_instance.provider.value
                                model = agent.llm_instance.model_name
                        except Exception as exc:
                            logging.getLogger(__name__).debug(
                                "Failed to read provider/model for stats: %s", exc
                            )
                        stats_lines.append(
                            self._get_translation("provider_model").format(
                                provider=provider, model=model
                            )
                        )
                        # Execution time not tracked here; keep lean — omit if not available

                        if stats_lines:
                            token_display = "\n".join(stats_lines)
                            token_metadata_message = {
                                "role": "assistant",
                                "content": token_display,
                                "metadata": {
                                    "title": self._get_translation(
                                        "token_statistics_title"
                                        )
                                },
                            }
                            # history is a list of messages for chatbot component
                            try:
                                updated_history = list(history) if history else []
                                updated_history.append(token_metadata_message)
                                history = updated_history
                            except Exception as exc:
                                logging.getLogger(__name__).debug(
                                    "Failed to append token stats to history: %s", exc
                                )
                    except Exception as exc:
                        # Non-fatal: stats block construction may fail silently
                        logging.getLogger(__name__).debug(
                            "Stats block construction failed: %s", exc
                        )

                # Ask app to refresh sidebar/status if available
                try:
                    if hasattr(self.main_app, "trigger_ui_update"):
                        self.main_app.trigger_ui_update()
                except Exception as exc:
                    logging.getLogger(__name__).debug(
                        "Failed to trigger UI update: %s", exc
                    )
        except Exception as exc:
            # Non-fatal: UI state update should still proceed
            logging.getLogger(__name__).debug(
                "UI state update failed: %s", exc
            )

        # Return only history - MultimodalTextbox update is handled in .then() chain
        # Following reference repo pattern: return only history
        # #region agent log
        import json
        try:
            with open(r"d:\Repo\cmw-platform-agent-gradio-6\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "stop-debug", "hypothesisId": "A", "location": "chat_tab.py:590", "message": "_handle_stop_click returning", "data": {"return_type": "list", "history_len": len(history) if history else 0}, "timestamp": __import__("time").time() * 1000}) + "\n")
        except: pass
        # #endregion
        return history

    def _finalize_tokens_on_stop(self, request: gr.Request, history: list[dict[str, str]]) -> list[dict[str, str]]:
        """Finalize token tracking and append stats when streaming is stopped"""
        if not (
            hasattr(self, "main_app")
            and self.main_app
            and hasattr(self.main_app, "session_manager")
        ):
            return history

        session_id = (
            self.main_app.session_manager.get_session_id(request)
            if request
            else "default"
        )
        agent = self.main_app.session_manager.get_session_agent(session_id)
        if not agent or not hasattr(agent, "token_tracker"):
            return history

        # Get current request messages for estimation fallback
        messages = []
        try:
            messages = agent.get_conversation_history(session_id)
        except Exception:
            messages = []

        # Update prompt tokens explicitly
        try:
            if messages:
                agent.token_tracker.count_prompt_tokens(messages)
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Failed to count prompt tokens on stop: %s", exc
            )

        # Finalize turn token usage using monotonic estimate
        try:
            # Refresh snapshot to feed the per-turn estimate (best-effort)
            try:
                agent.token_tracker.refresh_budget_snapshot(
                    agent=agent,
                    conversation_id=session_id,
                    messages_override=messages,
                )
            except Exception as snap_exc:
                logging.getLogger(__name__).debug(
                    "Failed to refresh snapshot on stop: %s", snap_exc
                )
            agent.token_tracker.finalize_turn_usage(None, messages)
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Failed to finalize turn usage on stop: %s", exc
            )

        # Check for compression after stop (if critical status)
        try:
            if hasattr(agent, "token_tracker") and agent.token_tracker:
                budget_snapshot = agent.token_tracker.get_budget_snapshot()
                if (
                    budget_snapshot
                    and budget_snapshot.get("status") == TOKEN_STATUS_CRITICAL
                    and should_compress_on_completion(
                        agent, session_id, budget_snapshot.get("status")
                    )
                ):
                    language = getattr(self, "language", "en")
                    # Use asyncio.run() since this is a sync method
                    asyncio.run(
                        perform_compression_with_notifications(
                            agent=agent,
                            conversation_id=session_id,
                            language=language,
                            keep_recent_turns=HISTORY_COMPRESSION_KEEP_RECENT_TURNS_MID_TURN,
                            reason="interrupted",
                            budget_snapshot=budget_snapshot,
                            rebuild_messages=False,
                        )
                    )
        except Exception as comp_exc:
            # Non-fatal: log and continue
            logging.getLogger(__name__).debug(
                "Failed to check/perform compression on stop: %s", comp_exc
            )

        # Build a stats block and append as assistant meta message
        try:
            stats_history = self._build_token_stats_message(agent, messages)
            if stats_history:
                return stats_history
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Stats block construction failed: %s", exc
            )

        return history

    def _build_token_stats_message(self, agent, messages: list) -> list[dict[str, str]] | None:
        """Build and return token statistics message for history"""
        prompt_tokens = agent.token_tracker.get_last_prompt_tokens()
        api_tokens = agent.token_tracker.get_last_api_tokens()

        stats_lines = []
        if prompt_tokens:
            stats_lines.append(
                self._get_translation("prompt_tokens").format(
                    tokens=prompt_tokens.formatted
                )
            )
        if api_tokens:
            stats_lines.append(
                self._get_translation("api_tokens").format(
                    tokens=api_tokens.formatted
                )
            )

        # Provider/model info
        provider = "unknown"
        model = "unknown"
        try:
            if getattr(agent, "llm_instance", None):
                provider = agent.llm_instance.provider.value
                model = agent.llm_instance.model_name
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Failed to read provider/model for stats: %s", exc
            )
        stats_lines.append(
            self._get_translation("provider_model").format(
                provider=provider, model=model
            )
        )

        if not stats_lines:
            return None

        token_display = "\n".join(stats_lines)
        token_metadata_message = {
            "role": "assistant",
            "content": token_display,
            "metadata": {
                "title": self._get_translation("token_statistics_title")
            },
        }

        # Return updated history with stats message appended
        return [token_metadata_message]

    def format_token_budget_display(self, request: gr.Request = None) -> str:
        """Format and return the token budget display - now session-aware"""
        if not hasattr(self, "main_app") or not self.main_app:
            return self._get_translation("token_budget_initializing")

        # Get session-specific agent
        agent = None
        if request and hasattr(self.main_app, "session_manager"):
            session_id = self.main_app.session_manager.get_session_id(request)
            agent = self.main_app.session_manager.get_session_agent(session_id)

        # No fallback to global agent - use session-specific agents only

        if not agent:
            return self._get_translation("token_budget_initializing")

        try:
            budget_info = agent.get_token_budget_info()

            if budget_info["status"] == "unknown":
                return self._get_translation("token_budget_unknown")

            # Get cumulative stats for detailed display
            cumulative_stats = agent.token_tracker.get_cumulative_stats()

            # "Сообщение" is per-turn and must be monotonic:
            # - sums API usage across iterations when available
            # - otherwise uses a monotonic estimate (snapshot-fed), esp. for interruptions
            try:
                used_tokens = int(agent.token_tracker.get_message_display_total_tokens() or 0)
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Failed to compute message display tokens: %s", exc
                )
                used_tokens = int(budget_info.get("used_tokens", 0) or 0)

            percentage_for_display = budget_info.get("percentage", 0.0)
            if budget_info["context_window"] > 0 and used_tokens > 0:
                percentage_for_display = round(
                    (used_tokens / budget_info["context_window"]) * 100.0, 1
                )

            # Determine status icon using localized translations
            # Recalculate status based on API token percentage if available
            if budget_info["context_window"] > 0 and used_tokens > 0:
                api_percentage = (used_tokens / budget_info["context_window"]) * 100.0
                if api_percentage >= TOKEN_STATUS_CRITICAL_THRESHOLD:
                    status_icon = self._get_translation("token_status_critical")
                elif api_percentage >= TOKEN_STATUS_WARNING_THRESHOLD:
                    status_icon = self._get_translation("token_status_warning")
                elif api_percentage >= TOKEN_STATUS_MODERATE_THRESHOLD:
                    status_icon = self._get_translation("token_status_moderate")
                else:
                    status_icon = self._get_translation("token_status_good")
            else:
                status_icon = self._get_translation(
                    f"token_status_{budget_info['status']}"
                )

            # Build token usage display using separated components for better flexibility
            total = self._get_translation("token_usage_total").format(
                total_tokens=cumulative_stats["conversation_tokens"]
            )
            conversation = self._get_translation("token_usage_conversation").format(
                conversation_tokens=cumulative_stats["session_tokens"]
            )
            estimated_total = 0
            try:
                # Prefer monotonic per-turn estimate; fall back to latest snapshot total.
                estimated_total = int(
                    agent.token_tracker.get_turn_estimated_total_tokens() or 0
                )
                if estimated_total <= 0:
                    snap = agent.token_tracker.get_budget_snapshot()
                    if isinstance(snap, dict):
                        estimated_total = int(snap.get("total_tokens", 0) or 0)
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Failed to compute estimated usage for display: %s", exc
                )
                estimated_total = 0

            estimate_line = self._get_translation("token_usage_estimate").format(
                estimated_tokens=estimated_total
            )
            last_message = self._get_translation("token_usage_last_message").format(
                percentage=percentage_for_display,
                used=used_tokens,
                context_window=budget_info["context_window"],
                status_icon=status_icon,
            )
            average = self._get_translation("token_usage_average").format(
                avg_tokens=cumulative_stats["avg_tokens_per_message"]
            )

            # Add token breakdown from latest budget snapshot
            breakdown_info = ""
            try:
                snap = agent.token_tracker.get_budget_snapshot()
                if isinstance(snap, dict):
                    conv_tokens = snap.get("conversation_tokens", 0)
                    tool_tokens = snap.get("tool_tokens", 0)
                    overhead_tokens = snap.get("overhead_tokens", 0)
                    breakdown_info = (
                        "\n" + self._get_translation("token_breakdown_context").format(conv_tokens=conv_tokens) +
                        "\n" + self._get_translation("token_breakdown_tools").format(tool_tokens=tool_tokens) +
                        "\n" + self._get_translation("token_breakdown_overhead").format(overhead_tokens=overhead_tokens)
                    )
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Failed to get token breakdown: %s", exc
                )

        except Exception as e:
            print(f"Error formatting token budget: {e}")
            return self._get_translation("token_budget_unknown")
        else:
            return (
                f"- {total}\n- {conversation}\n- {estimate_line}{breakdown_info}\n"
                f"- {last_message}\n- {average}"
            )

    def _get_available_providers(self) -> list[str]:
        """Get list of available LLM providers from session manager"""
        if not hasattr(self, "main_app") or not self.main_app:
            return [
                "openrouter",
                "groq",
                "gemini",
                "mistral",
                "huggingface",
                "gigachat",
            ]

        try:
            if hasattr(self.main_app, "llm_manager") and self.main_app.llm_manager:
                return self.main_app.llm_manager.get_available_providers()
        except Exception as e:
            print(f"Error getting available providers: {e}")

        return ["openrouter", "groq", "gemini", "mistral", "huggingface", "gigachat"]

    def _get_current_provider(self) -> str:
        """Get current LLM provider"""
        return os.environ.get("AGENT_PROVIDER", "openrouter")

    def _get_available_models(self) -> list[str]:
        """Get list of available models for the current provider from session manager"""
        if not hasattr(self, "main_app") or not self.main_app:
            return [self._get_translation("no_models_available")]

        try:
            if hasattr(self.main_app, "llm_manager") and self.main_app.llm_manager:
                current_provider = self._get_current_provider()
                config = self.main_app.llm_manager.get_provider_config(current_provider)
                if config and config.models:
                    models = [model["model"] for model in config.models]
                    return (
                        models
                        if models
                        else [self._get_translation("no_models_available")]
                    )
                return [self._get_translation("no_models_available")]
        except Exception as e:
            print(f"Error getting available models: {e}")
            return [self._get_translation("error_loading_providers")]

        return [self._get_translation("no_models_available")]

    def _get_available_provider_model_combinations(self) -> list[str]:
        """Get list of available provider/model combinations in format 'Provider / Model'"""
        if not hasattr(self, "main_app") or not self.main_app:
            return [self._get_translation("no_providers_available")]

        try:
            if hasattr(self.main_app, "llm_manager") and self.main_app.llm_manager:
                combinations = []
                available_providers = (
                    self.main_app.llm_manager.get_available_providers()
                )

                if not available_providers:
                    return [self._get_translation("no_providers_available")]

                for provider in available_providers:
                    config = self.main_app.llm_manager.get_provider_config(provider)
                    if config and config.models:
                        for model in config.models:
                            model_name = model["model"]
                            # Format as "Provider / Model"
                            combination = f"{provider.title()} / {model_name}"
                            combinations.append(combination)

                if not combinations:
                    return [self._get_translation("no_models_available")]

                return combinations
        except Exception as e:
            print(f"Error getting provider/model combinations: {e}")
            return [self._get_translation("error_loading_providers")]

        # No fallback - return error message
        return [self._get_translation("no_providers_available")]

    def _get_current_model(self) -> str:
        """Get current LLM model from session manager (fallback to default)"""
        if not hasattr(self, "main_app") or not self.main_app:
            return ""

        try:
            # Try to get from session manager first
            if hasattr(self.main_app, "session_manager"):
                # Get default session for UI display
                session_data = self.main_app.session_manager.get_session_data("default")
                if (
                    session_data
                    and session_data.agent
                    and hasattr(session_data.agent, "llm_instance")
                    and session_data.agent.llm_instance
                ):
                    return session_data.agent.llm_instance.model_name
        except Exception as e:
            print(f"Error getting current model: {e}")

        return ""

    def _get_current_provider_model_combination(self) -> str:
        """Get current provider/model combination in format 'Provider / Model'"""
        if not hasattr(self, "main_app") or not self.main_app:
            # Return fallback value when main app is not available
            provider = os.environ.get("AGENT_PROVIDER", "openrouter")
            return f"{provider.title()} / {provider}/default-model"

        try:
            # Try to get from session manager first
            if hasattr(self.main_app, "session_manager"):
                # Get default session for UI display
                session_data = self.main_app.session_manager.get_session_data("default")
                if (
                    session_data
                    and session_data.agent
                    and hasattr(session_data.agent, "llm_instance")
                    and session_data.agent.llm_instance
                ):
                    provider = session_data.agent.llm_instance.provider.value
                    model = session_data.agent.llm_instance.model_name
                    return f"{provider.title()} / {model}"
        except Exception as e:
            print(f"Error getting current provider/model combination: {e}")

        # Return fallback value on error
        provider = os.environ.get("AGENT_PROVIDER", "openrouter")
        return f"{provider.title()} / {provider}/default-model"

    def _update_models_for_provider(self, provider: str) -> list[str]:
        """Update available models when provider changes from session manager"""
        try:
            if not hasattr(self, "main_app") or not self.main_app:
                return []

            if hasattr(self.main_app, "llm_manager") and self.main_app.llm_manager:
                config = self.main_app.llm_manager.get_provider_config(provider)
                if config and config.models:
                    return [model["model"] for model in config.models]
        except Exception as e:
            print(f"Error updating models for provider {provider}: {e}")

        return []

    def _apply_llm_selection(self, provider: str, model: str) -> str:
        """Apply the selected LLM provider and model (deprecated - use session-aware method)"""
        # This method is deprecated - use _apply_llm_directly instead
        return self._apply_llm_directly(provider, model)

    def _apply_llm_selection_combined(
        self, provider_model_combination: str, request: gr.Request = None
    ) -> tuple[str, list[dict[str, str]], str]:
        """Apply the selected LLM provider/model combination - now properly session-aware"""
        try:
            if (
                not provider_model_combination
                or " / " not in provider_model_combination
            ):
                return self._get_translation("llm_apply_error"), [], ""

            # Parse the combination: "Provider / Model"
            parts = provider_model_combination.split(" / ", 1)
            if len(parts) != 2:
                return self._get_translation("llm_apply_error"), [], ""

            provider = parts[0].lower()  # Convert to lowercase for environment variable
            model = parts[1]

            if not hasattr(self, "main_app") or not self.main_app:
                return self._get_translation("llm_apply_error"), [], ""

            # Check if switching to Mistral and show native Gradio warning
            if self._is_mistral_model(provider, model):
                # Check if we're switching FROM a non-Mistral provider TO Mistral
                current_provider_model = self._get_current_provider_model_combination()
                current_is_mistral = "mistral" in current_provider_model.lower()

                # Only clear chat if switching from non-Mistral to Mistral
                if not current_is_mistral:
                    # Show native Gradio warning modal
                    gr.Warning(
                        message=self._get_translation("mistral_switch_warning").format(
                            provider=provider.title(), model=model
                        ),
                        title=self._get_translation("mistral_switch_title"),
                        duration=10,
                    )
                    # Apply the LLM selection and clear chat immediately (same as clear button)
                    return self._apply_mistral_with_clear(provider, model, request)
                # Switching from Mistral to Mistral - no need to clear chat
                status = self._apply_llm_directly(provider, model, request)
                return status, gr.update(), ""

            # For non-Mistral models, apply directly and preserve current chat state
            status = self._apply_llm_directly(provider, model, request)
            # Return current chat state to preserve conversation (don't clear chat for compatible LLMs)
            return status, gr.update(), ""

        except Exception as e:
            print(f"Error applying LLM selection: {e}")
            return self._get_translation("llm_apply_error"), [], ""

    def _is_mistral_model(self, provider: str, model: str) -> bool:
        """Check if the selected model is a Mistral model"""
        return provider.lower() == "mistral" or "mistral" in model.lower()

    def _apply_llm_directly(
        self, provider: str, model: str, request: gr.Request = None
    ) -> str:
        """Apply LLM selection without confirmation dialog - now properly session-aware"""
        try:
            print(
                f"🔄 ChatTab: Applying LLM selection - Provider: {provider}, Model: {model}"
            )
            print(f"🔄 ChatTab: Request available: {request is not None}")
            print(
                f"🔄 ChatTab: Main app has session_manager: {hasattr(self.main_app, 'session_manager')}"
            )

            # Use clean session manager for session-aware LLM selection
            if request and hasattr(self.main_app, "session_manager"):
                session_id = self.main_app.session_manager.get_session_id(request)
                print(f"🔄 ChatTab: Session ID: {session_id}")
                success = self.main_app.session_manager.update_llm_provider(
                    session_id, provider, model
                )
                print(f"🔄 ChatTab: Update result: {success}")
                if success:
                    # Trigger UI update to refresh status display
                    if hasattr(self.main_app, "trigger_ui_update"):
                        self.main_app.trigger_ui_update()
                    return self._get_translation("llm_apply_success").format(
                        provider=provider.title(), model=model
                    )
                return self._get_translation("llm_apply_error")

            # No fallback to global agent - use session-specific agents only
            return self._get_translation("llm_apply_error")
        except Exception as e:
            print(f"Error applying LLM selection: {e}")
            return self._get_translation("llm_apply_error")

    def _confirm_mistral_switch(
        self, provider_model_combination: str
    ) -> tuple[str, str, str]:
        """Handle Mistral switching confirmation - returns status, chatbot, and message"""
        try:
            if (
                not provider_model_combination
                or " / " not in provider_model_combination
            ):
                return self._get_translation("llm_apply_error"), "", ""

            # Parse the combination: "Provider / Model"
            parts = provider_model_combination.split(" / ", 1)
            if len(parts) != 2:
                return self._get_translation("llm_apply_error"), "", ""

            provider = parts[0].lower()
            model = parts[1]

            # Apply the LLM selection
            status = self._apply_llm_directly(provider, model)

            # Clear the chat history for Mistral
            if "success" in status.lower():
                # Get the clear handler from event handlers
                clear_handler = self.event_handlers.get("clear_chat")
                if clear_handler:
                    # Create a mock request for session isolation
                    class MockRequest:
                        def __init__(self):
                            self.session_hash = f"mock_session_{uuid.uuid4().hex[:8]}_{int(time.time())}"
                            self.client = type(
                                "MockClient",
                                (),
                                {"id": f"client_{uuid.uuid4().hex[:8]}"},
                            )()

                    request = MockRequest()
                    chatbot, _msg = clear_handler(request)
                else:
                    # Fallback clear
                    return status, [], ""
                return status, chatbot, _msg
            return status, "", ""

        except Exception as e:
            print(f"Error confirming Mistral switch: {e}")
            return self._get_translation("llm_apply_error"), "", ""

    def _apply_mistral_with_clear(
        self, provider: str, model: str, request: gr.Request = None
    ) -> tuple[str, str, str]:
        """Apply Mistral LLM selection and clear chat history - now properly session-aware"""
        try:
            # Apply the LLM selection
            status = self._apply_llm_directly(provider, model, request)

            # If successful, clear the chat history
            if status and status != self._get_translation("llm_apply_error"):
                # Get the clear handler from event handlers
                clear_handler = self.event_handlers.get("clear_chat")
                if clear_handler:
                    # Clear the chat and get the updated state
                    chatbot, _msg = clear_handler(request)
                    status += f" {self._get_translation('mistral_chat_cleared')}"
                else:
                    # Fallback clear - return empty chat
                    status += f" {self._get_translation('mistral_chat_cleared')}"
                    return status, [], ""
                return status, chatbot, _msg
            return status, "", ""

        except Exception as e:
            print(f"Error applying Mistral with clear: {e}")
            return self._get_translation("llm_apply_error"), "", ""

    def _cancel_mistral_switch(self) -> tuple[bool, str]:
        """Cancel Mistral switching and hide confirmation dialog"""
        return False, self._get_translation("mistral_switch_cancelled")

    def _get_translation(self, key: str) -> str:
        """Get a translation for a specific key"""
        # Always use direct translation for now to avoid i18n metadata issues
        return get_translation_key(key, self.language)

    def _reset_quick_actions_dropdown(self) -> str:
        """Reset the quick actions dropdown to None"""
        return None

    def _stream_message_wrapper(
        self,
        multimodal_value: dict[str, Any] | None,
        history: list[dict[str, str]],
        cancel_state: dict | None = None,
        request: gr.Request | None = None,
    ) -> tuple[
        list[dict[str, str]],
        str,
        None,
    ]:
        """Wrapper for concurrent processing with Gradio's native queue feedback

        Handles MultimodalValue format and extracts text for processing with proper session awareness.
        With status_update_rate="auto", Gradio will show native queue status - no need for custom warnings.

        Note: Stop button visibility is now handled by built-in stop button in MultimodalTextbox,
        shown/hidden via .success() events on streaming/submit events (following reference repo pattern).
        """

        # Helper to check if cancellation was requested (following reference repo pattern)
        def is_cancelled() -> bool:
            return cancel_state is not None and cancel_state.get("cancelled", False)

        # Process message with original wrapper
        # Stop button is shown via .success() on submit_event (interchanging buttons)
        yield self._yield_ui_newline(history)

        # Check for cancellation before starting
        if is_cancelled():
            logging.getLogger(__name__).info("Streaming cancelled before start")
            yield (history, "", None)
            return

        # Process message with original wrapper
        last_result = None
        for result in self._stream_message_wrapper_internal(
            multimodal_value, history, request
        ):
            # Check for cancellation during streaming (following reference repo pattern)
            if is_cancelled():
                logging.getLogger(__name__).info("Streaming cancelled during execution")
                break
            last_result = result
            yield (
                result[0],
                result[1],
                None,
            )

        # Hide stop button at end of processing (via .then() on submit_event)
        # This is handled by re_enable_textbox_and_hide_stop function chained to submit completion
        if last_result and len(last_result) >= 2:
            yield (
                last_result[0],
                last_result[1],
                None,
            )
        else:
            yield (
                history,
                "",
                None,
            )

    def _stream_message_wrapper_internal(
        self,
        multimodal_value: dict[str, Any] | None,
        history: list[dict[str, str]],
        request: gr.Request | None = None,
    ) -> AsyncGenerator[tuple[list[dict[str, str]], str], None]:
        """Internal wrapper to handle MultimodalValue format and extract text for processing - now properly session-aware"""
        # Extract text from MultimodalValue format
        if isinstance(multimodal_value, dict):
            message = multimodal_value.get("text", "")
            files = multimodal_value.get("files", [])

            # If there are files, process them with the new lean system
            if files:
                # Session cache paths are now managed by the session manager

                # Process files with new system
                file_info = "\n\n[Files: "
                file_list = []
                current_files = []

                for i, file in enumerate(files, 1):
                    # Extract original filename and file path
                    if isinstance(file, dict):
                        original_filename = file.get("orig_name")
                        file_path = file.get("path", "")
                        if not original_filename:
                            original_filename = (
                                os.path.basename(file_path)
                                if file_path
                                else f"file_{i}"
                            )
                    else:
                        file_path = str(file)
                        original_filename = os.path.basename(file_path)

                    # Get file size
                    try:
                        file_size = (
                            os.path.getsize(file_path)
                            if os.path.exists(file_path)
                            else 0
                        )
                        if file_size > 0:
                            size_str = FileUtils.format_file_size(file_size)
                            file_list.append(f"{original_filename} ({size_str})")
                        else:
                            file_list.append(f"{original_filename} (0 bytes)")
                    except Exception:
                        file_list.append(f"{original_filename}")

                    # Register file with agent's session-isolated registry
                    if (
                        hasattr(self, "main_app")
                        and self.main_app
                        and hasattr(self.main_app, "session_manager")
                    ):
                        session_id = self.main_app.session_manager.get_session_id(
                            request
                        )
                        agent = self.main_app.session_manager.get_agent(session_id)
                        if agent and hasattr(agent, "register_file"):
                            agent.register_file(original_filename, file_path)
                            current_files.append(original_filename)
                            print(
                                f"📁 Registered file: {original_filename} -> {agent.file_registry.get((session_id, original_filename), 'NOT_FOUND')}"
                            )

                file_info += ", ".join(file_list) + "]"
                message += file_info

                # Store current files (deprecated - use session manager)
                print(f"📁 Registered {len(current_files)} files: {current_files}")
            else:
                # No files, just use the text message
                pass
        else:
            # Fallback for non-dict values
            message = str(multimodal_value) if multimodal_value else ""

        # Get the original stream handler
        stream_handler = self.event_handlers.get("stream_message")
        if not stream_handler:
            yield history, ""
            return

        # Call the original stream handler with enhanced message (text + file analysis)
        # Now properly session-aware with real Gradio request
        yield from stream_handler(message, history, request)

    def _clear_chat_with_download_reset(
        self, request: gr.Request | None = None
    ) -> tuple[
        list[dict[str, str]],
        dict[str, Any],
    ]:
        """Clear chat and reset download state - now properly session-aware"""
        # Clear download button cache
        if hasattr(self, "_last_history_str"):
            delattr(self, "_last_history_str")
        if hasattr(self, "_last_download_file"):
            delattr(self, "_last_download_file")

        # Get the clear handler from event handlers
        clear_handler = self.event_handlers.get("clear_chat")
        if clear_handler:
            # Call the original clear handler with real Gradio request
            chatbot, _msg = clear_handler(request)
            # Return empty MultimodalValue
            empty_multimodal = {"text": "", "files": []}
            return chatbot, empty_multimodal
        # Fallback if clear handler not available
        empty_multimodal = {"text": "", "files": []}
        return [], empty_multimodal

    def get_download_button_updates(self, history):
        """
        Get download button updates for DownloadsTab.
        This method can be called from DownloadsTab to update buttons.

        Args:
            history: Conversation history

        Returns:
            Tuple of (markdown_button_update, html_button_update)
        """
        return self._update_download_button_visibility(history)

    def _update_download_button_visibility(self, history):
        """Update download button visibility and file based on conversation history"""
        if not CHAT_DOWNLOADS_ENABLED:
            # Downloads are globally disabled via feature flag
            return (
                gr.update(visible=False),
                gr.update(visible=False),
            )
        if history and len(history) > 0:
            # Check if conversation has changed since last generation
            history_str = str(history)
            if (
                not hasattr(self, "_last_history_str")
                or self._last_history_str != history_str
            ):
                # Generate files with fresh timestamp when conversation changes
                # Use try/except to prevent blocking on file generation errors
                try:
                    markdown_file_path = self._download_conversation_as_markdown(history)
                    # HTML file path is now stored in _last_html_file_path by _download_conversation_as_markdown
                    html_file_path = getattr(self, "_last_html_file_path", None)
                    self._last_history_str = history_str
                    self._last_download_file = markdown_file_path
                    self._last_download_html_file = html_file_path
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        f"Failed to generate download files: {exc}", exc_info=True
                    )
                    markdown_file_path = None
                    html_file_path = None
            else:
                # Use cached files if conversation hasn't changed
                markdown_file_path = getattr(self, "_last_download_file", None)
                html_file_path = getattr(self, "_last_download_html_file", None)

            if markdown_file_path and html_file_path:
                # Show both download buttons with pre-generated files
                # Use gr.update() instead of creating new components (Gradio 6 pattern)
                return (
                    gr.update(value=markdown_file_path, visible=True),
                    gr.update(value=html_file_path, visible=True),
                )
            # Hide buttons if generation fails
            return (
                gr.update(visible=False),
                gr.update(visible=False),
            )
        # Hide download buttons when there's no conversation history
        return (
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def _download_conversation_as_markdown(
        self, history: list[dict[str, str]]
    ) -> str:
        """
        Download the conversation history as a markdown file.

        Args:
            history: List of conversation messages from Gradio chatbot component

        Returns:
            File path if successful, None if failed
        """
        logger = logging.getLogger(__name__)
        logger.debug("Download function called with history type: %s", type(history))
        logger.debug("History content: %s", str(history)[:50])

        if not history:
            logger.warning("No history provided")
            return None

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"CMW_Copilot_{timestamp}.md"

        # Create markdown content with lean frontmatter
        markdown_content = "# CMW Platform Agent - Conversation Export\n\n"
        markdown_content += (
            f"**Exported on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        markdown_content += f"**Total messages:** {len(history)}\n\n"
        # Simple conversation summary using existing agent stats (minimal, non-intrusive)
        try:
            main_app = getattr(self, "main_app", None)
            if main_app and hasattr(main_app, "session_manager"):
                # Get current session ID to maintain session isolation
                try:
                    debug_streamer = get_debug_streamer()
                    session_id = debug_streamer.get_current_session_id()
                except Exception as debug_exc:
                    # Fallback to default if debug streamer not available
                    logging.getLogger(__name__).debug(
                        "Debug streamer not available: %s", debug_exc
                    )
                    session_id = "default"

                # Use existing agent stats instead of complex turn_complete event
                agent = main_app.session_manager.get_session_agent(session_id)
                if agent:
                    stats = agent.get_stats()
                    conversation_stats = stats.get("conversation_stats", {})
                    llm_info = stats.get("llm_info", {})

                    if conversation_stats.get("message_count", 0) > 0:
                        message_count = conversation_stats.get("message_count", 0)
                        user_messages = conversation_stats.get("user_messages", 0)
                        assistant_messages = conversation_stats.get(
                            "assistant_messages", 0
                        )
                        provider = llm_info.get("provider", "unknown")
                        model = llm_info.get("model", "unknown")

                        markdown_content += f"## Сводка диалога ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n"
                        markdown_content += f"**Всего сообщений:** {message_count} ({user_messages} user, {assistant_messages} assistant)\n\n"
                        markdown_content += (
                            f"**Провайдер / модель:** {provider} / {model}\n\n"
                        )
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Failed to add conversation summary to markdown: %s", exc
            )
        markdown_content += "---\n\n"

        # Add conversation messages
        # Handle Gradio 6 format: content can be a list of dicts [{'text': '...', 'type': 'text'}]
        # or a plain string (Gradio 5 format)
        def extract_text_content(content):
            """Extract plain text from content, handling both Gradio 5 and 6 formats."""
            if content is None:
                return ""

            # If content is a string, return it directly (Gradio 5 format)
            if isinstance(content, str):
                return content

            # If content is a list (Gradio 6 format), extract text from each item
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        # Extract text from dict items (e.g., {'text': '...', 'type': 'text'})
                        text = item.get("text", "")
                        if text:
                            text_parts.append(str(text))
                    elif isinstance(item, str):
                        # Direct string in list
                        text_parts.append(item)
                return "\n".join(text_parts)

            # Fallback: convert to string
            return str(content)

        for i, message in enumerate(history, 1):
            if isinstance(message, dict):
                role = message.get("role", "unknown")
                raw_content = message.get("content", "") or ""
                content = extract_text_content(raw_content)

                if role == "user":
                    markdown_content += f"## User Message {i}\n\n"
                    markdown_content += f"{content}\n\n"
                elif role == "assistant":
                    markdown_content += f"## Assistant Response {i}\n\n"
                    markdown_content += f"{content}\n\n"
                else:
                    markdown_content += f"## {role.title()} Message {i}\n\n"
                    markdown_content += f"{content}\n\n"
            else:
                # Fallback for other formats
                markdown_content += f"## Message {i}\n\n"
                markdown_content += f"{message!s}\n\n"

        # Create file with proper filename
        try:
            # Create a temporary directory and file with the proper filename
            temp_dir = tempfile.mkdtemp()
            clean_file_path = os.path.join(temp_dir, filename)

            with open(clean_file_path, "w", encoding="utf-8") as file:
                file.write(markdown_content)

            logger.debug("Created markdown file: %s", clean_file_path)

            # Also generate HTML version and store the path
            html_file_path = self._generate_conversation_html(markdown_content, filename.replace(".md", ".html"))
            if html_file_path:
                logger.debug("Also created HTML file: %s", html_file_path)
                # Store the HTML file path for the HTML download button
                self._last_html_file_path = html_file_path
            else:
                self._last_html_file_path = None

            # Return the markdown file path for Gradio to handle the download
            return clean_file_path
        except Exception as e:
            logger.exception("Error creating markdown file: %s", e)
            return None


    def _generate_conversation_html(self, markdown_content: str, filename: str) -> str:
        """
        Generate HTML file from markdown content.

        Args:
            markdown_content: Markdown content as string
            filename: HTML filename

        Returns:
            HTML file path if successful, None if failed
        """

        logger = logging.getLogger(__name__)

        try:
            # Convert Markdown to HTML
            html_body = markdown.markdown(markdown_content, extensions=["tables", "fenced_code"])

            # Load CSS from external file
            css_content = self._load_export_css()

            # Create HTML with CSS styling and Mermaid support
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CMW Platform Agent - Conversation Export</title>
    <style>
        {css_content}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function () {{
            if (window.mermaid) {{
                try {{
                    mermaid.initialize({{ startOnLoad: false, securityLevel: 'loose' }});
                    // Transform fenced code blocks with language-mermaid into mermaid containers
                    const mermaidCodes = document.querySelectorAll('pre > code.language-mermaid');
                    mermaidCodes.forEach(function(codeEl) {{
                        const graphDefinition = codeEl.textContent || '';
                        const preEl = codeEl.parentElement;
                        const container = document.createElement('div');
                        container.className = 'mermaid';
                        container.textContent = graphDefinition;
                        if (preEl) {{
                            preEl.replaceWith(container);
                        }}
                    }});
                    // Render all mermaid diagrams
                    mermaid.run();
                }} catch (e) {{
                    // Non-fatal: if Mermaid fails, leave code blocks as-is
                    console && console.warn && console.warn('Mermaid render failed:', e);
                }}
            }}
        }});
    </script>
</head>
<body>
    <div class="content">
        {html_body}
    </div>
</body>
</html>"""

            # Create file with proper filename
            temp_dir = tempfile.mkdtemp()
            clean_file_path = os.path.join(temp_dir, filename)

            with open(clean_file_path, "w", encoding="utf-8") as file:
                file.write(html_content)

            logger.debug("Generated HTML file: %s", clean_file_path)
            return clean_file_path
        except Exception as e:
            logger.exception("Error generating HTML file: %s", e)
            return None

    def _load_export_css(self) -> str:
        """
        Load CSS content from the external CSS file.

        Returns:
            CSS content as string
        """
        # Get the path to the CSS file relative to the project root
        css_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                               "resources", "css", "html_export_theme.css")

        with open(css_path, encoding="utf-8") as css_file:
            return css_file.read()
