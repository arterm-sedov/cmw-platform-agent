"""
Chat Tab Module for App NG
=========================

Handles the main chat interface, quick actions, and user interactions.
This module encapsulates all chat-related UI components and functionality.
Supports internationalization (i18n) with Russian and English translations.
"""

import asyncio
from collections.abc import Callable
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from .sidebar import QuickActionsMixin


class ChatTab(QuickActionsMixin):
    """Chat tab component with interface and quick actions"""

    def __init__(
        self,
        event_handlers: dict[str, Callable],
        language: str = "en",
        i18n_instance: gr.I18n | None = None,
    ):
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

        with gr.TabItem(self._get_translation("tab_chat"), id="chat") as tab:
            # Create main chat interface (includes sidebar)
            self._create_chat_interface()

            # Connect event handlers
            self._connect_events()

        logging.getLogger(__name__).info(
            "✅ ChatTab: Successfully created with all components and event handlers"
        )
        return tab, self.components

    def _create_chat_interface(self):
        """Create the main chat interface with proper layout"""
        # Chat interface with metadata support for thinking transparency
        self.components["chatbot"] = gr.Chatbot(
            label=self._get_translation("chat_label"),
            height=500,
            show_label=True,
            container=True,
            show_copy_button=True,
            type="messages",
            elem_id="chatbot-main",
            elem_classes=["chatbot-card"],
        )

        with gr.Row():
            self.components["msg"] = gr.MultimodalTextbox(
                label=self._get_translation("message_label"),
                placeholder=self._get_translation("message_placeholder"),
                lines=2,
                scale=4,
                max_lines=4,
                elem_id="message-input",
                elem_classes=["message-card"],
                file_types=[
                    ".pdf", ".csv", ".tsv", ".xlsx", ".xls",  # Documents and data
                    ".docx", ".pptx", ".vsdx", ".msg", ".eml",  # Office documents
                    ".zip", ".rar", ".tar", ".gz", ".bz2",  # Archives
                    ".dwg", ".bpmn", ".sql", ".conf", ".ico",  # Other supported formats
                    ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".xml", ".html",
                    ".css", ".md", ".ini", ".sh", ".bat", ".ps1", ".c", ".cpp", ".h",
                    ".hpp", ".java", ".go", ".rs", ".rb", ".php", ".pl", ".swift",
                    ".kt", ".scala", ".sql", ".toml", ".env"  # Common text-based code formats
                ],
                file_count="multiple",
            )
            with gr.Column():
                self.components["send_btn"] = gr.Button(
                    self._get_translation("send_button"),
                    variant="primary",
                    scale=1,
                    elem_classes=["cmw-button"],
                )
                self.components["stop_btn"] = gr.Button(
                    self._get_translation("stop_button"),
                    variant="stop",
                    scale=1,
                    elem_classes=["cmw-button"],
                    visible=False,
                )
                self.components["clear_btn"] = gr.Button(
                    self._get_translation("clear_button"),
                    variant="secondary",
                    elem_classes=["cmw-button"],
                )
                self.components["download_btn"] = gr.DownloadButton(
                    label=self._get_translation("download_button"),
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
        if not stream_handler:
            raise ValueError("stream_message handler not found in event_handlers")
        if not clear_handler:
            raise ValueError("clear_chat handler not found in event_handlers")

        logging.getLogger(__name__).debug(
            "✅ ChatTab: Critical event handlers validated"
        )

        # Get queue manager for concurrency control
        queue_manager = getattr(self, "main_app", None)
        if queue_manager:
            queue_manager = getattr(queue_manager, "queue_manager", None)

        # Main chat events with concurrency control and queue status
        if queue_manager:
            # Apply concurrency settings to chat events
            from agent_ng.queue_manager import (
                apply_concurrency_to_click_event,
                apply_concurrency_to_submit_event,
            )

            # Send button click with concurrency and queue status
            send_config = apply_concurrency_to_click_event(
                queue_manager,
                "chat",
                self._stream_message_with_queue_status,
                [self.components["msg"], self.components["chatbot"]],
                [
                    self.components["chatbot"],
                    self.components["msg"],
                    self.components["stop_btn"],
                    self.components["download_btn"],
                    self._get_quick_actions_dropdown(),
                ],
            )
            self.streaming_event = self.components["send_btn"].click(**send_config)

            # Message submit with concurrency and queue status
            submit_config = apply_concurrency_to_submit_event(
                queue_manager,
                "chat",
                self._stream_message_with_queue_status,
                [self.components["msg"], self.components["chatbot"]],
                [
                    self.components["chatbot"],
                    self.components["msg"],
                    self.components["stop_btn"],
                    self.components["download_btn"],
                    self._get_quick_actions_dropdown(),
                ],
            )
            self.submit_event = self.components["msg"].submit(**submit_config)
        else:
            # Fallback to default behavior if queue manager not available
            logging.getLogger(__name__).warning(
                "⚠️ Queue manager not available - using default event configuration"
            )
            self.streaming_event = self.components["send_btn"].click(
                fn=self._stream_message_wrapper,
                inputs=[self.components["msg"], self.components["chatbot"]],
                outputs=[
                    self.components["chatbot"],
                    self.components["msg"],
                    self.components["stop_btn"],
                    self.components["download_btn"],
                    self._get_quick_actions_dropdown(),
                ],
            )

            self.submit_event = self.components["msg"].submit(
                fn=self._stream_message_wrapper,
                inputs=[self.components["msg"], self.components["chatbot"]],
                outputs=[
                    self.components["chatbot"],
                    self.components["msg"],
                    self.components["stop_btn"],
                    self.components["download_btn"],
                    self._get_quick_actions_dropdown(),
                ],
            )

        # Stop button - cancel both send and submit events and hide itself
        self.components["stop_btn"].click(
            fn=self._handle_stop_click,
            inputs=[self.components["chatbot"]],
            outputs=[self.components["stop_btn"]],
            cancels=[self.streaming_event, self.submit_event],
        )

        self.components["clear_btn"].click(
            fn=self._clear_chat_with_download_reset,
            outputs=[
                self.components["chatbot"],
                self.components["msg"],
                self.components["download_btn"],
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

    def _setup_chat_event_triggers(self):
        """Setup event triggers to update other UI components when chat events occur"""
        # Get UI update handlers
        trigger_ui_update = self.event_handlers.get("trigger_ui_update")

        if trigger_ui_update:
            # Trigger UI update after send button click
            self.components["send_btn"].click(
                fn=trigger_ui_update,
                outputs=[],  # No specific outputs, just triggers the update
            )

            # Trigger UI update after message submit
            self.components["msg"].submit(fn=trigger_ui_update, outputs=[])

            # Trigger UI update after clear button click
            self.components["clear_btn"].click(fn=trigger_ui_update, outputs=[])

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

    def get_stop_button(self) -> gr.Button:
        """Get the stop button component for visibility control"""
        return self.components["stop_btn"]

    def _get_quick_actions_dropdown(self) -> gr.Dropdown:
        """Get the quick actions dropdown from the sidebar"""
        # The dropdown is now in the sidebar, so we need to get it from the main app
        if hasattr(self, "main_app") and self.main_app:
            # Try to get from UI Manager components
            if hasattr(self.main_app, "ui_manager") and self.main_app.ui_manager:
                components = self.main_app.ui_manager.get_components()
                return components.get("quick_actions_dropdown")

        # Fallback - return a dummy component that won't cause errors
        return gr.Dropdown(visible=False)

    def _handle_stop_click(self, history):
        """Handle stop button click - hide the button immediately"""
        return gr.Button(visible=False)

    def format_token_budget_display(self, request: gr.Request = None) -> str:
        """Format and return the token budget display - now session-aware"""
        if not hasattr(self, "main_app") or not self.main_app:
            return self._get_translation("token_budget_initializing")

        # Get session-specific agent
        agent = None
        if request and hasattr(self.main_app, "session_manager"):
            session_id = self.main_app.session_manager.get_session_id(request)
            agent = self.main_app.session_manager.get_session_agent(session_id)

        # Fallback to global agent if no session context
        if not agent:
            agent = self.main_app.agent

        if not agent:
            return self._get_translation("token_budget_initializing")

        try:
            budget_info = agent.get_token_budget_info()

            if budget_info["status"] == "unknown":
                return self._get_translation("token_budget_unknown")

            # Get cumulative stats for detailed display
            cumulative_stats = agent.token_tracker.get_cumulative_stats()

            # Determine status icon using localized translations
            status_icon = self._get_translation(f"token_status_{budget_info['status']}")

            # Build token usage display using separated components for better flexibility
            total = self._get_translation("token_usage_total").format(
                total_tokens=cumulative_stats["conversation_tokens"]
            )
            conversation = self._get_translation("token_usage_conversation").format(
                conversation_tokens=cumulative_stats["session_tokens"]
            )
            last_message = self._get_translation("token_usage_last_message").format(
                percentage=budget_info["percentage"],
                used=budget_info["used_tokens"],
                context_window=budget_info["context_window"],
                status_icon=status_icon,
            )
            average = self._get_translation("token_usage_average").format(
                avg_tokens=cumulative_stats["avg_tokens_per_message"]
            )

            return f"- {total}\n- {conversation}\n- {last_message}\n- {average}"
        except Exception as e:
            print(f"Error formatting token budget: {e}")
            return self._get_translation("token_budget_unknown")

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
        import os

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
                else:
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
            import os

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
        import os

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
                current_provider = (
                    current_provider_model.split(" / ")[0].lower()
                    if " / " in current_provider_model
                    else ""
                )

                # Only clear chat if switching from non-Mistral to Mistral
                if current_provider and current_provider != "mistral":
                    # Show native Gradio warning modal
                    gr.Warning(
                        message=self._get_translation("mistral_switch_warning").format(
                            provider=provider.title(), model=model
                        ),
                        title=self._get_translation("mistral_switch_title"),
                        duration=10,
                    )
                    # Apply the LLM selection and clear chat
                    return self._apply_mistral_with_clear(provider, model, request)
                else:
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
                else:
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
                    import time
                    import uuid

                    class MockRequest:
                        def __init__(self):
                            self.session_hash = f"mock_session_{uuid.uuid4().hex[:8]}_{int(time.time())}"
                            self.client = type(
                                "MockClient",
                                (),
                                {"id": f"client_{uuid.uuid4().hex[:8]}"},
                            )()

                    request = MockRequest()
                    chatbot, msg = clear_handler(request)
                    return status, chatbot, msg
                else:
                    # Fallback clear
                    return status, [], ""

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
            if "success" in status.lower():
                # Get the clear handler from event handlers
                clear_handler = self.event_handlers.get("clear_chat")
                if clear_handler:
                    # Clear the chat and get the updated state
                    chatbot, msg = clear_handler(request)
                    status += f" {self._get_translation('mistral_chat_cleared')}"
                    return status, chatbot, msg
                else:
                    # Fallback clear - return empty chat
                    status += f" {self._get_translation('mistral_chat_cleared')}"
                    return status, [], ""

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
        from ..i18n_translations import get_translation_key

        return get_translation_key(key, self.language)

    def _reset_quick_actions_dropdown(self) -> str:
        """Reset the quick actions dropdown to None"""
        return None

    def _stream_message_with_queue_status(
        self, multimodal_value, history, request: gr.Request = None
    ):
        """Wrapper for concurrent processing - relies on Gradio's native queue feedback"""
        # With status_update_rate="auto", Gradio will show native queue status
        # No need for custom warnings - Gradio handles this natively

        # Show stop button at start of processing
        yield (
            history,
            "",
            gr.Button(visible=True),
            self._update_download_button_visibility(history),
            None,
        )  # Show stop button, update download, reset dropdown

        # Process message with original wrapper
        last_result = None
        for result in self._stream_message_wrapper_internal(
            multimodal_value, history, request
        ):
            last_result = result
            if len(result) >= 2:
                yield (
                    result[0],
                    result[1],
                    gr.Button(visible=True),
                    self._update_download_button_visibility(result[0]),
                    None,
                )  # Keep stop button visible, update download, reset dropdown
            else:
                yield (
                    result[0],
                    result[1],
                    gr.Button(visible=True),
                    self._update_download_button_visibility(result[0]),
                    None,
                )  # Keep stop button visible, update download, reset dropdown

        # Hide stop button at end of processing and reset dropdown
        if last_result and len(last_result) >= 2:
            yield (
                last_result[0],
                last_result[1],
                gr.Button(visible=False),
                self._update_download_button_visibility(last_result[0]),
                None,
            )  # Reset dropdown
        else:
            yield (
                history,
                "",
                gr.Button(visible=False),
                self._update_download_button_visibility(history),
                None,
            )  # Reset dropdown

    def _stream_message_wrapper(
        self, multimodal_value, history, request: gr.Request = None
    ):
        """Wrapper to handle MultimodalValue format and extract text for processing - now properly session-aware"""
        # Fallback mode without queue status

        # Show stop button at start of processing
        yield (
            history,
            "",
            gr.Button(visible=True),
            self._update_download_button_visibility(history),
            None,
        )  # Show stop button, update download, reset dropdown

        # Process message with original wrapper
        last_result = None
        for result in self._stream_message_wrapper_internal(
            multimodal_value, history, request
        ):
            last_result = result
            if len(result) >= 2:
                yield (
                    result[0],
                    result[1],
                    gr.Button(visible=True),
                    self._update_download_button_visibility(result[0]),
                    None,
                )  # Keep stop button visible, update download, reset dropdown
            else:
                yield (
                    result[0],
                    result[1],
                    gr.Button(visible=True),
                    self._update_download_button_visibility(result[0]),
                    None,
                )  # Keep stop button visible, update download, reset dropdown

        # Hide stop button at end of processing and reset dropdown
        if last_result and len(last_result) >= 2:
            yield (
                last_result[0],
                last_result[1],
                gr.Button(visible=False),
                self._update_download_button_visibility(last_result[0]),
                None,
            )  # Reset dropdown
        else:
            yield (
                history,
                "",
                gr.Button(visible=False),
                self._update_download_button_visibility(history),
                None,
            )  # Reset dropdown

    def _stream_message_wrapper_internal(
        self, multimodal_value, history, request: gr.Request = None
    ):
        """Internal wrapper to handle MultimodalValue format and extract text for processing - now properly session-aware"""
        # Extract text from MultimodalValue format
        if isinstance(multimodal_value, dict):
            message = multimodal_value.get("text", "")
            files = multimodal_value.get("files", [])

            # If there are files, process them with the new lean system
            if files:
                from tools.file_utils import FileUtils

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
        for result in stream_handler(message, history, request):
            yield result


    def _clear_chat_with_download_reset(self, request: gr.Request = None):
        """Clear chat and reset download state - now properly session-aware"""
        # Get the clear handler from event handlers
        clear_handler = self.event_handlers.get("clear_chat")
        if clear_handler:
            # Call the original clear handler with real Gradio request
            chatbot, msg = clear_handler(request)
            # Reset download button (hide it) and return empty MultimodalValue
            empty_multimodal = {"text": "", "files": []}
            return chatbot, empty_multimodal, gr.DownloadButton(visible=False)
        else:
            # Fallback if clear handler not available
            empty_multimodal = {"text": "", "files": []}
            return [], empty_multimodal, gr.DownloadButton(visible=False)

    def _update_download_button_visibility(self, history):
        """Update download button visibility and file based on conversation history"""
        if history and len(history) > 0:
            # Generate file with fresh timestamp when conversation changes
            file_path = self._download_conversation_as_markdown(history)
            if file_path:
                # Show download button with pre-generated file
                return gr.DownloadButton(
                    label=self._get_translation("download_button"),
                    value=file_path,
                    variant="secondary",
                    elem_classes=["cmw-button"],
                    visible=True,
                )
            else:
                # Show button without file if generation fails
                return gr.DownloadButton(
                    label=self._get_translation("download_button"),
                    variant="secondary",
                    elem_classes=["cmw-button"],
                    visible=True,
                )
        else:
            # Hide download button when there's no conversation history
            return gr.DownloadButton(visible=False)

    def _download_conversation_as_markdown(self, history) -> str:
        """
        Download the conversation history as a markdown file.

        Args:
            history: List of conversation messages from Gradio chatbot component

        Returns:
            File path if successful, None if failed
        """
        from datetime import datetime
        import os
        import tempfile

        print(f"DEBUG: Download function called with history type: {type(history)}")
        print(f"DEBUG: History content: {history}")

        if not history:
            print("DEBUG: No history provided")
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
                    from ..debug_streamer import get_debug_streamer

                    debug_streamer = get_debug_streamer()
                    session_id = debug_streamer.get_current_session_id()
                except Exception:
                    # Fallback to default if debug streamer not available
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
        except Exception:
            pass
        markdown_content += "---\n\n"

        # Add conversation messages
        # Handle the actual format from the debug output
        for i, message in enumerate(history, 1):
            if isinstance(message, dict):
                role = message.get("role", "unknown")
                content = message.get("content", "")

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

            print(f"DEBUG: Created file: {clean_file_path}")
            # Return the clean file path for Gradio to handle the download
            return clean_file_path
        except Exception as e:
            print(f"Error creating markdown file: {e}")
            return None
