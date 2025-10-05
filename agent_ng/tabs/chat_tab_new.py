"""
Chat New Tab Module for App NG
=============================

Handles the new chat interface using Gradio's native ChatInterface.
This module provides a modern, streamlined chat experience while preserving
all existing functionality: memory, streaming, tools, sessions, etc.

Based on Gradio's ChatInterface documentation and best practices.
"""

from collections.abc import AsyncGenerator
import logging
from typing import Any

import gradio as gr

from agent_ng.debug_streamer import get_debug_streamer
from agent_ng.i18n_translations import get_translation_key
from agent_ng.session_manager import set_current_session_id
from agent_ng.tabs.sidebar import QuickActionsMixin


class ChatNewTab(QuickActionsMixin):
    """New Chat tab component with native Gradio ChatInterface"""

    def __init__(
        self,
        event_handlers: dict[str, Any],
        language: str = "en",
        i18n_instance: gr.I18n | None = None,
    ) -> None:
        self.event_handlers = event_handlers
        self.components = {}
        self.main_app = None  # Reference to main app for progress status
        self.language = language
        self.i18n = i18n_instance
        self.debug_streamer = get_debug_streamer("chat_new_tab")

    def create_tab(self) -> tuple[gr.TabItem, dict[str, Any]]:
        """
        Create the new chat tab with all its components.

        Returns:
            Tuple of (TabItem, components_dict)
        """
        logging.getLogger(__name__).info(
            "✅ ChatNewTab: Creating native ChatInterface..."
        )

        with gr.TabItem(self._get_translation("tab_chat_new"), id="chat_new") as tab:
            self._create_native_chat_interface()

        logging.getLogger(__name__).info(
            "✅ ChatNewTab: Successfully created with native ChatInterface"
        )
        return tab, self.components

    def _create_native_chat_interface(self):
        """Create the native ChatInterface with agent integration"""
        # Create the native ChatInterface with basic configuration
        self.components["chat_interface"] = gr.ChatInterface(
            fn=self._chat_function,
            title=self._get_translation("chat_new_title"),
            description=self._get_translation("chat_new_description"),
            examples=self._get_examples(),
            cache_examples=False,
            type="messages",  # Use messages format for better compatibility
            # Enable chat history persistence
            save_history=True,
            # Enable user feedback
            flagging_mode="manual",
            flagging_options=["Like", "Dislike", "Inappropriate", "Other"],
        )

    def _chat_function(
        self,
        message: str,
        history: list[dict[str, str]],
        request: gr.Request = None,
    ) -> AsyncGenerator[gr.ChatMessage, None]:
        """
        Main chat function that integrates with the agent.

        Args:
            message: User message
            history: Chat history in OpenAI format
            request: Gradio request object for session management

        Yields:
            ChatMessage objects for streaming
        """
        if not message.strip():
            return

        # Get the stream handler from event handlers
        stream_handler = self.event_handlers.get("stream_message")
        if not stream_handler:
            yield gr.ChatMessage(
                content=self._get_translation("error_no_stream_handler"),
                role="assistant",
            )
            return

        # Set session context for logging and request config resolution
        session_id = self.main_app.session_manager.get_session_id(request)
        set_current_session_id(session_id)
        snippet = message[:50]
        log_msg = (
            f"ChatNewTab: Streaming message for session {session_id}: {snippet}..."
        )
        self.debug_streamer.info(log_msg)

        # Convert history to the format expected by the stream handler
        # The stream handler expects a list of dicts with role/content
        formatted_history = []
        for msg in history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                formatted_history.append(msg)
            elif isinstance(msg, tuple) and len(msg) == 2:
                # Handle tuple format (user, assistant)
                user_msg, assistant_msg = msg
                if user_msg:
                    formatted_history.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    formatted_history.append(
                        {
                            "role": "assistant",
                            "content": assistant_msg,
                        }
                    )

        # Stream the response using the existing agent
        try:
            # Track the accumulated response content
            last_yielded_content = ""
            last_history_signature: str | None = None

            for result in stream_handler(message, formatted_history, request):
                if isinstance(result, tuple) and len(result) >= 2:
                    # Extract history and message from result
                    updated_history, _ = result[0], result[1]

                    if isinstance(updated_history, list) and updated_history:
                        # Find the latest assistant message with content
                        latest_assistant: dict[str, Any] | None = None
                        for msg in reversed(updated_history):
                            if (
                                isinstance(msg, dict)
                                and msg.get("role") == "assistant"
                                and (msg.get("content") or "").strip()
                            ):
                                latest_assistant = msg
                                break

                        if latest_assistant:
                            # Pull latest fields; used when rebuilding history
                            _ = (latest_assistant.get("content") or "").strip()
                            _ = latest_assistant.get("metadata") or None

                            # Build full message list so UI appends chronologically
                            gr_messages: list[gr.ChatMessage] = []
                            for hmsg in updated_history:
                                if not isinstance(hmsg, dict):
                                    continue
                                role = hmsg.get("role")
                                if role not in ("user", "assistant"):
                                    continue
                                hcontent = (hmsg.get("content") or "").strip()
                                hmeta = hmsg.get("metadata") or None
                                if hcontent:
                                    gr_messages.append(
                                        gr.ChatMessage(
                                            content=hcontent,
                                            role=role,
                                            metadata=hmeta,
                                        )
                                    )

                            # Generate a lightweight signature
                            signature_parts = [
                                f"{m.role}:{m.content}" for m in gr_messages
                            ]
                            history_signature = "\n".join(signature_parts)

                            if (
                                history_signature
                                and history_signature != last_history_signature
                            ):
                                last_history_signature = history_signature
                                yield gr_messages
                elif isinstance(result, str):
                    # Direct string response
                    if result and result != last_yielded_content:
                        last_yielded_content = result
                        # Append as a new assistant message
                        yield [
                            gr.ChatMessage(
                                content=result,
                                role="assistant",
                            )
                        ]
        except Exception as e:
            self.debug_streamer.error(f"Error in _chat_function: {e}")
            yield gr.ChatMessage(
                content=f"Error: {e!s}",
                role="assistant",
            )
        finally:
            # Clear session context after turn
            set_current_session_id(None)

    def _get_examples(self) -> list[str]:
        """Get examples for the chat interface"""
        return [
            self._get_translation("example_1"),
            self._get_translation("example_2"),
            self._get_translation("example_3"),
        ]


    def set_main_app(self, app):
        """Set reference to main app for accessing session manager"""
        self.main_app = app

    def _get_translation(self, key: str) -> str:
        """Get a translation for a specific key"""
        return get_translation_key(key, self.language)
