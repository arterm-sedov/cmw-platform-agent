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

from agent_ng.history_compression import (
    perform_compression_with_notifications,
    should_compress_on_completion,
)
from agent_ng.i18n_translations import get_translation_key
from agent_ng.queue_manager import (
    apply_concurrency_to_click_event,
    apply_concurrency_to_submit_event,
)
from agent_ng.session_manager import get_current_session_id
from agent_ng.token_budget import (
    HISTORY_COMPRESSION_KEEP_RECENT_TURNS_MID_TURN,
    TOKEN_STATUS_CRITICAL,
    TOKEN_STATUS_CRITICAL_THRESHOLD,
    TOKEN_STATUS_MODERATE_THRESHOLD,
    TOKEN_STATUS_WARNING_THRESHOLD,
)
from tools.file_utils import FileUtils

try:
    from agent_ng._file_attachment import build_file_bubbles_for_role
except ImportError:
    try:
        from .._file_attachment import (
            build_file_bubbles_for_role,  # type: ignore[no-redef]
        )
    except Exception:  # pragma: no cover
        def build_file_bubbles_for_role(_att, role="user"):  # type: ignore[no-redef]
            return []

from .sidebar import QuickActionsMixin

CHAT_DOWNLOADS_ENABLED = True  # Enable chat export/download functionality


def _chatbot_message_content_to_export_text(content: Any) -> str | None:
    """Normalize Chatbot message content for Markdown export (Gradio 5/6)."""
    if content is None:
        return None
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    chunks.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
        if not chunks:
            return None
        return "\n\n".join(chunks).strip() or None
    if isinstance(content, dict):
        if "path" in content or "file" in content:
            label = (
                content.get("alt_text")
                or content.get("display_name")
                or content.get("orig_name")
                or content.get("path")
            )
            return f"*(attachment: {label})*" if label else "*(attachment)*"
        return None
    text = str(content).strip()
    return text or None


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
            with gr.TabItem(
                self._get_translation("tab_chat"),
                id="chat",
                render_children=True,
            ) as tab:
                self.build_ui(show_stack_heading=False)

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

    def build_ui(self, *, show_stack_heading: bool = False) -> None:
        """Mount chat UI (inside ``TabItem``)."""
        if show_stack_heading:
            gr.Markdown(
                f"### {self._get_translation('tab_chat')}",
                elem_classes=["stack-section-heading"],
            )
        self._create_chat_interface()
        self._connect_events()

    def _create_chat_interface(self):
        """Create the main chat interface with proper layout"""
        # Chat interface with metadata support for thinking transparency
        # In Gradio 6, Chatbot uses messages format by default, so no type parameter is needed
        self.components["chatbot"] = gr.Chatbot(
            label=self._get_translation("chat_label"),
            value=[],
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
                    ".txt",  # Pasted text files from Gradio
                    "text",  # MIME category for all text/*
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
            # REMOVED: Separate clear button - now using Gradio's native chatbot.clear() button
            # The built-in clear button in the chatbot component handles clearing

        # In-chat exports (same as ``main``); heavy prep is deferred via ui_manager (tab select / env).
        with gr.Row(elem_classes=["chat-download-row"]):
            self.components["download_btn"] = gr.DownloadButton(
                label=self._get_translation("download_button"),
                variant="secondary",
                elem_classes=["cmw-button"],
                visible=False,
            )
            self.components["download_html_btn"] = gr.DownloadButton(
                label=self._get_translation("download_html_button"),
                variant="secondary",
                elem_classes=["cmw-button"],
                visible=False,
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

        # True while the stream generator is running (used to avoid heavy Downloads prep mid-stream).
        streaming_active = gr.State(value=False)
        self.components["streaming_active"] = streaming_active

        # Step 1: Submit event - simple function that clears textbox
        # Following reference repo pattern: this triggers .success() which shows stop button
        user_submit = self.components["msg"].submit(
            fn=clear_and_save_multimodal_textbox,
            inputs=[self.components["msg"]],
            outputs=[self.components["msg"], saved_input],  # Clear textbox and save message to state
            queue=False,
            api_visibility="private",
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
            api_visibility="private",
        )

        # Sidebar refresh once per submit (cmw-rag avoids a second .submit() on the same component)
        trigger_ui_update = self.event_handlers.get("trigger_ui_update")
        submit_chain_root = user_submit
        if trigger_ui_update:
            submit_chain_root = user_submit.then(
                fn=lambda: trigger_ui_update(),
                inputs=[],
                outputs=[],
                queue=False,
                api_visibility="private",
            )

        # Reset cancellation state at start of new submission (following reference repo pattern)
        def reset_cancellation_state(cancel_state: dict | None) -> dict:
            """Reset cancellation state at start of new submission."""
            if cancel_state is None or not isinstance(cancel_state, dict):
                cancel_state = {"cancelled": False}
            else:
                cancel_state["cancelled"] = False
            return cancel_state

        def reset_stream_start(cancel_state: dict | None) -> tuple[dict, bool]:
            """Reset cancel dict and mark streaming active before generator runs."""
            return reset_cancellation_state(cancel_state), True

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
                api_visibility="private",
            )
            # Remove 'fn' from config since we'll use it directly in .then()
            streaming_fn = streaming_config.pop("fn")
            streaming_inputs = streaming_config.pop("inputs")
            streaming_outputs = streaming_config.pop("outputs")

            # Chain streaming handler from submit chain (cmw-rag: optional step after submit)
            # Following reference repo pattern: reset cancellation state first, then stream
            streaming_pipeline = submit_chain_root.then(
                fn=reset_stream_start,
                inputs=[cancellation_state],
                outputs=[cancellation_state, streaming_active],
                queue=False,
                api_visibility="private",
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
            streaming_pipeline = submit_chain_root.then(
                fn=reset_stream_start,
                inputs=[cancellation_state],  # Request is automatically passed to functions that accept it
                outputs=[cancellation_state, streaming_active],
                queue=False,
                api_visibility="private",
            ).then(
                fn=self._stream_message_wrapper,
                inputs=[saved_input, self.components["chatbot"], cancellation_state],  # Add cancellation_state
                outputs=[
                    self.components["chatbot"],
                    self.components["msg"],
                    self._get_quick_actions_dropdown(),
                ],
                api_visibility="private",
            )

        # Re-enable textbox and hide stop button after streaming completes
        # Following reference repo pattern: chain after handler completion (cmw-rag: before msg.stop)
        def re_enable_textbox_and_hide_stop():
            """Re-enable textbox and hide stop button after handler completion."""
            logging.getLogger(__name__).debug(
                "Re-enabling textbox and hiding stop button after handler completion"
            )
            return (
                gr.MultimodalTextbox(
                    value="", interactive=True, submit_btn=True, stop_btn=False
                ),
                False,
            )

        self.submit_event = streaming_pipeline.then(
            fn=re_enable_textbox_and_hide_stop,
            outputs=[self.components["msg"], streaming_active],
            queue=False,
            api_visibility="private",
        )

        # Stop must cancel a queued dependency (Gradio validates cancels -> queue=True).
        # submit_event is the non-queued re-enable tail; streaming_pipeline ends with the stream step.
        self.stop_event = self.components["msg"].stop(
            fn=self._handle_stop_click,
            inputs=[self.components["chatbot"], cancellation_state],
            outputs=[
                self.components["chatbot"],
                cancellation_state,
            ],
            cancels=[streaming_pipeline],
            api_visibility="private",
        ).then(
            lambda: gr.MultimodalTextbox(
                value="", interactive=True, submit_btn=True, stop_btn=False
            ),
            outputs=[self.components["msg"]],
            queue=False,
            api_visibility="private",
        ).then(
            lambda: False,
            outputs=[streaming_active],
            queue=False,
            api_visibility="private",
        )

        # Handle chatbot clear event - clear memory and reset downloads
        # Following reference repo pattern: wire chatbot.clear() to handle everything
        def handle_chatbot_clear(request: gr.Request | None = None) -> tuple[list[dict[str, str]], dict[str, Any]]:
            """Handle chatbot clear event - clear memory, reset downloads, and clear UI."""
            logging.getLogger(__name__).info("Chatbot clear event - clearing memory and resetting downloads")
            # Use the same clear handler that was used for the separate clear button
            return self._clear_chat_with_download_reset(request)

        # Bind to the built-in clear button's clear event
        # This replaces the separate clear button - Gradio's native clear button handles everything
        self.clear_event = self.components["chatbot"].clear(
            fn=handle_chatbot_clear,
            inputs=[],  # Request is automatically passed to functions that accept it
            outputs=[
                self.components["chatbot"],
                self.components["msg"],
            ],
            api_visibility="private",
        ).then(
            lambda: False,
            outputs=[streaming_active],
            queue=False,
            api_visibility="private",
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
            # Submit hook is chained in _connect_events (avoids second msg.submit on same component)

            # Trigger UI update after chatbot clear (built-in clear button)
            if hasattr(self, "clear_event") and self.clear_event:
                self.clear_event.then(
                    fn=trigger_ui_update,
                    outputs=[],
                    queue=False,
                    api_visibility="private",
                )

            # Trigger UI update after built-in stop (token budget / status)
            if hasattr(self, "stop_event") and self.stop_event:
                self.stop_event.then(
                    fn=trigger_ui_update,
                    outputs=[],
                    queue=False,
                    api_visibility="private",
                )

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

        # Also update cancellation state in session manager for async streaming to access
        if (
            hasattr(self, "main_app")
            and self.main_app
            and hasattr(self.main_app, "session_manager")
            and request
        ):
            try:
                session_id = self.main_app.session_manager.get_session_id(request)
                self.main_app.session_manager.set_cancellation_state(session_id, True)
                logging.getLogger(__name__).debug(f"Updated session cancellation state for {session_id[:8]}...")
            except Exception as exc:
                logging.getLogger(__name__).debug(f"Failed to update session cancellation state: {exc}")

        logging.getLogger(__name__).info("Stop button clicked - setting cancellation flag")
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
                                    tokens=prompt_tokens.formatted_no_cost
                                )
                            )
                        if api_tokens:
                            stats_lines.append(
                                self._get_translation("api_tokens").format(
                                    tokens=api_tokens.formatted_no_cost
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

        # Return history and cancellation_state - MultimodalTextbox update is handled in .then() chain
        # Following reference repo pattern: return history and cancellation_state
        return history, cancel_state

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
        cumulative_stats = agent.token_tracker.get_cumulative_stats()

        stats_lines = []
        if prompt_tokens:
            stats_lines.append(
                self._get_translation("prompt_tokens").format(
                    tokens=prompt_tokens.formatted_no_cost
                )
            )
        if api_tokens:
            # Show API tokens with input/output breakdown (no cost in chat)
            token_line = self._get_translation("api_tokens").format(
                tokens=api_tokens.formatted_no_cost
            )
            stats_lines.append(token_line)

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

    def _is_free_model(self, agent) -> bool:
        """Heuristic: OpenRouter uses ':free' suffix for free-tier models."""
        try:
            llm_instance = getattr(agent, "llm_instance", None)
            model = (getattr(llm_instance, "model_name", "") or "").lower()
            return ":free" in model
        except Exception:
            return False

    def _format_cost_display(self, agent, cost: float | None) -> str:
        """Format cost for UI: show $0.0000 only for free models, else '—' if unknown."""
        if cost is None:
            return "—"  # Unknown pricing
        if cost == 0.0:
            # cost == 0.0: check if it's a free model or unknown
            if self._is_free_model(agent):
                return "$0.0000"  # Explicitly free model
            return "—"  # Unknown (0.0 but not a free model)
        # Use helper to format with minimum necessary decimal places
        from agent_ng.utils import format_cost
        return format_cost(cost)

    def format_token_budget_display(self, request: gr.Request = None) -> str:
        """Format and return the token budget display - now session-aware"""
        if not hasattr(self, "main_app") or not self.main_app:
            return self._get_translation("token_budget_initializing")

        # Session-specific agent: prefer gr.Request; fallback to context session when
        # tails omit Request (e.g. legacy queue=False handlers).
        agent = None
        sm = getattr(self.main_app, "session_manager", None)
        if sm:
            session_id = None
            if request:
                session_id = sm.get_session_id(request)
            else:
                session_id = sm.get_current_session_id()
            if session_id:
                agent = sm.get_session_agent(session_id)

        if not agent:
            return self._get_translation("token_budget_initializing")

        try:
            budget_info = agent.get_token_budget_info()

            if budget_info["status"] == "unknown":
                return self._get_translation("token_budget_unknown")

            # Get cumulative stats for detailed display
            try:
                cumulative_stats = agent.token_tracker.get_cumulative_stats()
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Failed to get cumulative stats: %s", exc
                )
                return self._get_translation("token_budget_initializing")

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

            # Build token usage display using hierarchical format
            # Get cost information
            conv_cost = cumulative_stats.get("conversation_cost")
            total_cost = cumulative_stats.get("total_cost")
            turn_cost = cumulative_stats.get("turn_cost")

            # Format total with cost (precision .4f)
            total_tokens = cumulative_stats.get("conversation_tokens", 0)
            cost_str = ""
            total_cost_display = self._format_cost_display(agent, total_cost)
            if total_cost_display != "—":
                cost_str = f" / {total_cost_display}"
            total = self._get_translation("token_usage_total").format(
                total_tokens=total_tokens
            ) + cost_str

            # Format conversation with cost (precision .4f)
            conv_tokens = cumulative_stats.get("session_tokens", 0)
            conv_cost_str = ""
            conv_cost_display = self._format_cost_display(agent, conv_cost)
            if conv_cost_display != "—":
                conv_cost_str = f" / {conv_cost_display}"
            conversation = self._get_translation("token_usage_conversation").format(
                conversation_tokens=conv_tokens
            ) + conv_cost_str

            # Get estimated total for forecast breakdown
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

            # Forecast breakdown (indented sub-items)
            breakdown_info = ""
            try:
                snap = agent.token_tracker.get_budget_snapshot()
                if isinstance(snap, dict):
                    conv_tokens_snap = snap.get("conversation_tokens", 0)
                    tool_tokens = snap.get("tool_tokens", 0)
                    overhead_tokens = snap.get("overhead_tokens", 0)
                    breakdown_info = (
                        "\n    - " + self._get_translation("token_breakdown_context").format(conv_tokens=conv_tokens_snap)
                        + "\n    - " + self._get_translation("token_breakdown_tools").format(tool_tokens=tool_tokens)
                        + "\n    - " + self._get_translation("token_breakdown_overhead").format(overhead_tokens=overhead_tokens)
                    )
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Failed to get token breakdown: %s", exc
                )

            estimate_line = "- " + self._get_translation("token_usage_estimate").format(
                estimated_tokens=estimated_total
            ) + breakdown_info

            # Message section with context, input/output, and cost (indented sub-items)
            message_section = "- " + self._get_translation("token_usage_last_message")
            input_tokens = cumulative_stats.get("last_input_tokens", 0)
            output_tokens = cumulative_stats.get("last_output_tokens", 0)

            # Message context (percentage)
            message_context_line = self._get_translation("token_message_context").format(
                percentage=percentage_for_display,
                used=used_tokens,
                context_window=budget_info["context_window"],
                status_icon=status_icon,
            )

            # Input/output tokens
            input_line = self._get_translation("token_message_input").format(tokens=input_tokens)
            output_line = self._get_translation("token_message_output").format(tokens=output_tokens)

            # Cache details (OpenRouter prompt_tokens_details)
            cached_tokens = cumulative_stats.get("last_cached_tokens")
            cache_write_tokens = cumulative_stats.get("last_cache_write_tokens")
            cache_lines = ""
            if cached_tokens is not None:
                cache_lines += (
                    "\n    - "
                    + self._get_translation("token_message_cached_tokens").format(
                        tokens=int(cached_tokens or 0)
                    )
                )
            if cache_write_tokens is not None:
                cache_lines += (
                    "\n    - "
                    + self._get_translation("token_message_cache_write_tokens").format(
                        tokens=int(cache_write_tokens or 0)
                    )
                )

            # Cost for current message/turn
            base_cost = turn_cost if (turn_cost is not None) else conv_cost
            message_cost_str = self._format_cost_display(agent, base_cost)
            cost_line = self._get_translation("token_message_cost").format(cost=message_cost_str)

            message_details = f"\n    - {message_context_line}\n    - {input_line}\n    - {output_line}{cache_lines}\n    - {cost_line}"

            # Average with cost (precision .4f)
            avg_tokens = cumulative_stats.get("avg_tokens_per_message", 0)
            message_count = cumulative_stats.get("message_count", 0)
            avg_cost_str = ""
            if message_count > 0:
                # Treat 0-cost as "unknown" unless we know we're on a free model.
                avg_cost = (conv_cost / message_count) if (conv_cost is not None and conv_cost > 0) else None
                avg_cost_display = self._format_cost_display(agent, avg_cost)
                avg_cost_str = f" / {avg_cost_display}"
            average = self._get_translation("token_usage_average").format(
                avg_tokens=avg_tokens
            ) + avg_cost_str

        except Exception as e:
            print(f"Error formatting token budget: {e}")
            return self._get_translation("token_budget_unknown")
        else:
            return (
                f"- {total}\n- {conversation}\n{estimate_line}\n{message_section}{message_details}\n- {average}"
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

    def _default_dropdown_combo_str(self) -> str:
        """Env/settings default as ``Provider / model_id``."""
        mgr = getattr(self.main_app, "llm_manager", None) if self.main_app else None
        if not mgr:
            p = os.environ.get("AGENT_PROVIDER", "openrouter").strip()
            return f"{p.title()} / {p}"
        pe, idx = mgr._get_configured_provider_and_model_index()
        if not pe:
            p = os.environ.get("AGENT_PROVIDER", "openrouter").strip()
            return f"{p.title()} / {p}"
        cfg = mgr.get_provider_config(pe.value)
        if cfg and 0 <= idx < len(cfg.models):
            mname = cfg.models[idx].get("model", "") or pe.value
            return f"{pe.value.title()} / {mname}"
        return f"{pe.value.title()} / {pe.value}"

    def _get_current_model(self, request: gr.Request | None = None) -> str:
        """Current model id for the active session, else empty."""
        if not self.main_app or not getattr(self.main_app, "session_manager", None):
            return ""
        if not request:
            return ""
        try:
            sid = self.main_app.session_manager.get_session_id(request)
            agent = self.main_app.session_manager.get_session_agent(sid)
            inst = getattr(agent, "llm_instance", None)
            if inst:
                return inst.model_name
        except Exception as e:
            logging.getLogger(__name__).debug("ChatTab: current model: %s", e)
        return ""

    def _get_current_provider_model_combination(
        self, request: gr.Request | None = None
    ) -> str:
        """Current ``Provider / Model`` for the active Gradio session."""
        if request and self.main_app and getattr(self.main_app, "session_manager", None):
            try:
                sid = self.main_app.session_manager.get_session_id(request)
                agent = self.main_app.session_manager.get_session_agent(sid)
                inst = getattr(agent, "llm_instance", None)
                if inst:
                    p = inst.provider.value
                    m = inst.model_name
                    return f"{p.title()} / {m}"
            except Exception as e:
                logging.getLogger(__name__).debug(
                    "ChatTab: current provider/model: %s", e
                )
        return self._default_dropdown_combo_str()

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
                current_provider_model = self._get_current_provider_model_combination(
                    request
                )
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
        # Pass cancel_state to internal wrapper so it can check cancellation during streaming
        for result in self._stream_message_wrapper_internal(
            multimodal_value, history, cancel_state, request
        ):
            # Check for cancellation during streaming (following reference repo pattern)
            # This check happens between yields, so cancellation is detected promptly
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
        cancel_state: dict | None = None,
        request: gr.Request | None = None,
    ) -> AsyncGenerator[tuple[list[dict[str, str]], str], None]:
        """Internal wrapper to handle MultimodalValue format and extract text for processing - now properly session-aware

        Args:
            cancel_state: Cancellation state dict to check for stop button clicks during streaming
        """
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
                    _main_app = getattr(self, "main_app", None)
                    _session_mgr = getattr(_main_app, "session_manager", None) if _main_app else None
                    if _main_app and _session_mgr:
                        try:
                            session_id = _session_mgr.get_session_id(request)
                            agent = _session_mgr.get_agent(session_id)
                            if agent and hasattr(agent, "register_file"):
                                agent.register_file(original_filename, file_path)
                                current_files.append(original_filename)
                            else:
                                logging.getLogger(__name__).warning(
                                    "Agent or register_file not available: agent=%s",
                                    agent is not None,
                                )
                        except Exception as e:
                            logging.getLogger(__name__).error(
                                "Error registering file %s: %s",
                                original_filename, e,
                            )
                    else:
                        logging.getLogger(__name__).warning(
                            "Cannot register file: main_app=%s, session_mgr=%s",
                            _main_app is not None, _session_mgr is not None,
                        )

                file_info += ", ".join(file_list) + "]"
                message += file_info

                # Prepend an inline preview bubble for each uploaded file so
                # it appears in the chat history visually (not just as text).
                # build_file_bubbles_for_role returns [] for missing files so
                # no guard needed — safe to call unconditionally.
                for file in files:
                    fpath = (
                        file.get("path", "") if isinstance(file, dict) else str(file)
                    )
                    fname = (
                        file.get("orig_name") or os.path.basename(fpath)
                        if isinstance(file, dict)
                        else os.path.basename(fpath)
                    )
                    if fpath and os.path.isfile(fpath):
                        att = {
                            "path": str(Path(fpath).resolve()),
                            "display_name": fname or os.path.basename(fpath),
                            "size_bytes": (
                                os.path.getsize(fpath)
                                if os.path.exists(fpath)
                                else 0
                            ),
                        }
                        history = list(history or [])
                        history.extend(build_file_bubbles_for_role(att, role="user"))

                logging.getLogger(__name__).debug(
                    "Registered %d files: %s", len(current_files), current_files,
                )
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

        # Helper to check if cancellation was requested (following reference repo pattern)
        def is_cancelled() -> bool:
            return cancel_state is not None and cancel_state.get("cancelled", False)

        # Call the original stream handler with enhanced message (text + file analysis)
        # Now properly session-aware with real Gradio request
        # Check for cancellation between yields from stream_handler
        # Note: This check happens between yields, so cancellation is detected as soon as the generator yields
        for result in stream_handler(message, history, request):
            # Check for cancellation during streaming (following reference repo pattern)
            # This check happens between yields, so if the generator is yielding frequently, cancellation is detected promptly
            if is_cancelled():
                logging.getLogger(__name__).info("Streaming cancelled during execution (in internal wrapper)")
                break
            yield result

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
        if hasattr(self, "_last_download_html_file"):
            delattr(self, "_last_download_html_file")
        if hasattr(self, "_last_export_include_html"):
            delattr(self, "_last_export_include_html")
        if hasattr(self, "_last_html_file_path"):
            delattr(self, "_last_html_file_path")

        # Get the clear handler from event handlers
        clear_handler = self.event_handlers.get("clear_chat")
        if clear_handler:
            # Call the original clear handler with real Gradio request
            chatbot, _msg = clear_handler(request)
            # Return empty MultimodalValue
            empty_multimodal = {"text": "", "files": []}
            return chatbot if chatbot is not None else [], empty_multimodal
        # Fallback if clear handler not available
        empty_multimodal = {"text": "", "files": []}
        return [], empty_multimodal

    def get_download_cached_ui_updates(self) -> tuple[Any, Any]:
        """Reuse last export paths without regenerating files (avoids stalls mid-stream)."""
        if not CHAT_DOWNLOADS_ENABLED:
            return gr.update(visible=False), gr.update(visible=False)
        markdown_file_path = getattr(self, "_last_download_file", None)
        html_file_path = getattr(self, "_last_download_html_file", None)
        if markdown_file_path and html_file_path:
            return (
                gr.update(value=markdown_file_path, visible=True),
                gr.update(value=html_file_path, visible=True),
            )
        if markdown_file_path:
            return (
                gr.update(value=markdown_file_path, visible=True),
                gr.update(visible=False),
            )
        return gr.update(visible=False), gr.update(visible=False)

    def get_download_button_updates(
        self,
        history,
        *,
        generate_html: bool | None = None,
    ):
        """
        Get download button updates for DownloadsTab.
        This method can be called from DownloadsTab to update buttons.

        Args:
            history: Conversation history
            generate_html: When ``None``, include HTML export. When ``False``, Markdown only.

        Returns:
            Tuple of (markdown_button_update, html_button_update)
        """
        return self._update_download_button_visibility(
            history,
            generate_html=generate_html,
        )

    def _update_download_button_visibility(
        self,
        history,
        *,
        generate_html: bool | None = None,
    ):
        """Update download button visibility and file based on conversation history"""
        if not CHAT_DOWNLOADS_ENABLED:
            # Downloads are globally disabled via feature flag
            return (
                gr.update(visible=False),
                gr.update(visible=False),
            )
        if history and len(history) > 0:
            effective_html = True if generate_html is None else bool(generate_html)
            # Check if conversation has changed since last generation
            history_str = str(history)
            regen = (
                not hasattr(self, "_last_history_str")
                or self._last_history_str != history_str
                or getattr(self, "_last_export_include_html", None) != effective_html
            )
            if regen:
                # Generate files with fresh timestamp when conversation changes
                # Use try/except to prevent blocking on file generation errors
                try:
                    markdown_file_path = self._download_conversation_as_markdown(
                        history,
                        generate_html=effective_html,
                    )
                    # HTML file path is now stored in _last_html_file_path by _download_conversation_as_markdown
                    html_file_path = getattr(self, "_last_html_file_path", None)
                    self._last_history_str = history_str
                    self._last_download_file = markdown_file_path
                    self._last_download_html_file = html_file_path
                    self._last_export_include_html = effective_html
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
            if markdown_file_path:
                # MD-only when HTML export is disabled or HTML build failed
                return (
                    gr.update(value=markdown_file_path, visible=True),
                    gr.update(visible=False),
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
        self,
        history: list[dict[str, str]],
        *,
        generate_html: bool | None = None,
    ) -> str | None:
        """
        Download the conversation history as a markdown file.

        Args:
            history: List of conversation messages from Gradio chatbot component
            generate_html: When ``None``, generate HTML companion export.

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
        # Simple conversation summary using existing agent stats (minimal, non-intrusive).
        try:
            main_app = getattr(self, "main_app", None)
            if main_app and hasattr(main_app, "session_manager"):
                session_id = (get_current_session_id() or "").strip() or "default"

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

                        markdown_content += (
                            f"## Сводка диалога ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n"
                        )
                        markdown_content += (
                            "**Всего сообщений:** "
                            f"{message_count} ({user_messages} user, {assistant_messages} assistant)\n\n"
                        )
                        markdown_content += (
                            f"**Провайдер / модель:** {provider} / {model}\n\n"
                        )
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "Failed to add conversation summary to markdown: %s", exc
            )
        markdown_content += "---\n\n"

        # Conversation bodies (Gradio 6 uses list/dict multimodal ``content``, not only str).
        for i, message in enumerate(history, 1):
            if isinstance(message, dict):
                body = _chatbot_message_content_to_export_text(message.get("content"))
                if not body:
                    continue
                role = message.get("role", "unknown")
                if role == "user":
                    markdown_content += f"## User Message {i}\n\n{body}\n\n"
                elif role == "assistant":
                    markdown_content += f"## Assistant Response {i}\n\n{body}\n\n"
                else:
                    markdown_content += f"## {role.title()} Message {i}\n\n{body}\n\n"
            else:
                markdown_content += f"## Message {i}\n\n{message!s}\n\n"

        # Create file with proper filename
        try:
            # Create a temporary directory and file with the proper filename
            temp_dir = tempfile.mkdtemp()
            clean_file_path = os.path.join(temp_dir, filename)

            with open(clean_file_path, "w", encoding="utf-8") as file:
                file.write(markdown_content)

            logger.debug("Created markdown file: %s", clean_file_path)

            _want_html = True if generate_html is None else bool(generate_html)
            # HTML is optional (can stall Gradio when run on tab.select).
            html_file_path = None
            if _want_html:
                html_file_path = self._generate_conversation_html(
                    markdown_content, filename.replace(".md", ".html")
                )
                if html_file_path:
                    logger.debug("Also created HTML file: %s", html_file_path)
                    self._last_html_file_path = html_file_path
                else:
                    self._last_html_file_path = None
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
