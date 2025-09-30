"""
Sidebar Module for App NG
=========================

Handles the common sidebar components that are shared across all tabs.
This module provides a unified sidebar with LLM selection, quick actions,
status, and monitoring components.
Supports internationalization (i18n) with Russian and English translations.
"""

import logging
import os
from typing import Any, ClassVar, Optional

import gradio as gr

from agent_ng.i18n_translations import get_translation_key


class QuickActionsMixin:
    """Mixin class providing quick action methods for UI components"""

    # Configuration for all quick actions - eliminates repetitive code
    QUICK_ACTIONS_CONFIG: ClassVar[dict[str, str]] = {
        "quick_what_can_do": "quick_what_can_do_message",
        "quick_what_cannot_do": "quick_what_cannot_do_message",
        "quick_list_apps": "quick_list_apps_message",
        "quick_math": "quick_math_message",
        "quick_code": "quick_code_message",
        "quick_explain": "quick_explain_message",
        "quick_full_audit": "quick_full_audit_message",
        "quick_templates_erp": "quick_templates_erp_message",
        "quick_attributes_contractors": "quick_attributes_contractors_message",
        "quick_edit_date_time": "quick_edit_date_time_message",
        "quick_create_comment_attr": "quick_create_comment_attr_message",
        "quick_create_id_attr": "quick_create_id_attr_message",
        "quick_edit_phone_mask": "quick_edit_phone_mask_message",
        "quick_edit_enum": "quick_edit_enum_message",
        "quick_get_comment_attr": "quick_get_comment_attr_message",
        "quick_create_attr": "quick_create_attr_message",
        "quick_edit_mask": "quick_edit_mask_message",
        "quick_archive_attr": "quick_archive_attr_message",

    }

    def _get_translation(self, key: str) -> str:
        """Get a translation for a specific key - must be implemented by the class using this mixin"""
        error_msg = (
            "Classes using QuickActionsMixin must implement _get_translation method"
        )
        raise NotImplementedError(error_msg)

    def _get_quick_action_choices(self) -> list[tuple[str, str]]:
        """Get list of quick action choices for the dropdown"""
        choices = [("", "")]  # Empty first option to allow proper dropdown behavior

        choices.extend(
            (self._get_translation(action_key), action_key)
            for action_key in self.QUICK_ACTIONS_CONFIG
        )

        return choices

    def _handle_quick_action_dropdown(self, selected_action: str) -> str:
        """Handle quick action dropdown selection and return appropriate message text"""
        if not selected_action or selected_action not in self.QUICK_ACTIONS_CONFIG:
            return ""

        message_key = self.QUICK_ACTIONS_CONFIG[selected_action]
        return self._get_translation(message_key)

    def _handle_quick_action_dropdown_multimodal(
        self, selected_action: str
    ) -> dict[str, Any]:
        """Handle quick action dropdown selection and return appropriate message in MultimodalValue format"""
        if not selected_action or selected_action not in self.QUICK_ACTIONS_CONFIG:
            return {"text": "", "files": []}

        message_key = self.QUICK_ACTIONS_CONFIG[selected_action]
        return {"text": self._get_translation(message_key), "files": []}


class Sidebar(QuickActionsMixin):
    """Common sidebar component for all tabs"""

    def __init__(
        self,
        event_handlers: dict[str, Any],
        language: str = "en",
        i18n_instance: gr.I18n | None = None,
    ) -> None:
        self.event_handlers = event_handlers
        self.components = {}
        self.main_app = None  # Reference to main app for accessing session manager
        self.language = language
        self.i18n = i18n_instance

    def create_sidebar(self) -> tuple[gr.Sidebar, dict[str, Any]]:
        """
        Create the common sidebar with all its components.

        Returns:
            Tuple of (Sidebar, components_dict)
        """
        logging.getLogger(__name__).info(
            "✅ Sidebar: Creating common sidebar interface..."
        )

        with gr.Sidebar(open=True, width=420) as sidebar:
            # LLM Selection section
            with gr.Column(elem_classes=["model-card"]):
                gr.Markdown(
                    f"### {self._get_translation('llm_selection_title')}",
                    elem_classes=["llm-selection-title"],
                )

                # Combined Provider/Model selector
                self.components["provider_model_selector"] = gr.Dropdown(
                    choices=self._get_available_provider_model_combinations(),
                    value=self._get_current_provider_model_combination(),
                    show_label=False,
                    interactive=True,
                    allow_custom_value=True,
                    elem_classes=["provider-model-selector"],
                )

            # Quick actions section - styled like LLM selection
            with gr.Column(elem_classes=["model-card"]):
                gr.Markdown(
                    f"### {self._get_translation('quick_actions_title')}",
                    elem_classes=["llm-selection-title"],
                )

                # Single dropdown for all quick actions - styled like LLM selector
                self.components["quick_actions_dropdown"] = gr.Dropdown(
                    choices=self._get_quick_action_choices(),
                    value=None,
                    show_label=False,
                    interactive=True,
                    allow_custom_value=False,
                    elem_classes=["provider-model-selector"],
                )

            # Status section
            with gr.Column(elem_classes=["model-card"]):
                # Progress indicator
                gr.Markdown(
                    f"### {self._get_translation('progress_title')}",
                    elem_classes=["progress-title"],
                )
                self.components["progress_display"] = gr.Markdown(
                    self._get_translation("progress_ready")
                )
                # Token budget indicator
                gr.Markdown(
                    f"### {self._get_translation('token_budget_title')}",
                    elem_classes=["token-budget-title"],
                )
                self.components["token_budget_display"] = gr.Markdown(
                    self._get_translation("token_budget_initializing")
                )
                # Status indicator
                gr.Markdown(
                    f"### {self._get_translation('status_title')}",
                    elem_classes=["status-title"],
                )
                self.components["status_display"] = gr.Markdown(
                    self._get_translation("status_initializing")
                )

        # Connect sidebar event handlers
        self._connect_sidebar_events()

        logging.getLogger(__name__).info(
            "✅ Sidebar: Successfully created with all components and event handlers"
        )
        return sidebar, self.components

    def _connect_sidebar_events(self):
        """Connect all event handlers for the sidebar components"""
        logging.getLogger(__name__).debug("🔗 Sidebar: Connecting event handlers...")

        # Quick action dropdown event will be connected later after all tabs are created
        # This is handled in the UI Manager after all components are available

        # LLM selection events - now applies immediately on dropdown change
        if (
            "provider_model_selector" in self.components
            and "status_display" in self.components
        ):
            self.components["provider_model_selector"].change(
                fn=self._apply_llm_selection_combined,
                inputs=[self.components["provider_model_selector"]],
                outputs=[self.components["status_display"]],
            )

        # Token budget display change event for download button visibility
        if "token_budget_display" in self.components:
            self.components["token_budget_display"].change(
                fn=self._update_download_button_visibility,
                inputs=[],
                outputs=[],  # Output will be handled by the main app
            )

        logging.getLogger(__name__).debug(
            "✅ Sidebar: All event handlers connected successfully"
        )

    def set_main_app(self, app):
        """Set reference to main app for accessing session manager and other services"""
        self.main_app = app

    def get_components(self) -> dict[str, Any]:
        """Get all components created by this sidebar"""
        return self.components

    def get_status_component(self) -> gr.Markdown:
        """Get the status display component for auto-refresh"""
        return self.components.get("status_display")

    def get_progress_display(self) -> gr.Markdown:
        """Get the progress display component"""
        return self.components.get("progress_display")

    def get_token_budget_display(self) -> gr.Markdown:
        """Get the token budget display component"""
        return self.components.get("token_budget_display")

    def get_llm_selection_components(self) -> dict[str, Any]:
        """Get LLM selection components for UI updates"""
        return {
            "provider_model_selector": self.components.get("provider_model_selector"),
            "quick_actions_dropdown": self.components.get("quick_actions_dropdown"),
        }

    def _get_translation(self, key: str) -> str:
        """Get a translation for a specific key"""
        return get_translation_key(key, self.language)

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

    def _apply_llm_selection_combined(
        self, provider_model_combination: str, request: gr.Request = None
    ) -> str:
        """Apply the selected LLM provider/model combination - now properly session-aware"""
        try:
            if (
                not provider_model_combination
                or " / " not in provider_model_combination
            ):
                return self._get_translation("llm_apply_error")

            # Parse the combination: "Provider / Model"
            parts = provider_model_combination.split(" / ", 1)
            if len(parts) != 2:
                return self._get_translation("llm_apply_error")

            provider = parts[0].lower()  # Convert to lowercase for environment variable
            model = parts[1]

            if not hasattr(self, "main_app") or not self.main_app:
                return self._get_translation("llm_apply_error")

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
                    # Apply the LLM selection and clear chat
                    return self._apply_mistral_with_clear(provider, model, request)
                # Switching from Mistral to Mistral - no need to clear chat
                return self._apply_llm_directly(provider, model, request)

            # For non-Mistral models, apply directly and preserve current chat state
            return self._apply_llm_directly(provider, model, request)

        except Exception as e:
            print(f"Error applying LLM selection: {e}")
            return self._get_translation("llm_apply_error")

    def _is_mistral_model(self, provider: str, model: str) -> bool:
        """Check if the selected model is a Mistral model"""
        return provider.lower() == "mistral" or "mistral" in model.lower()

    def _apply_llm_directly(
        self, provider: str, model: str, request: gr.Request = None
    ) -> str:
        """Apply LLM selection without confirmation dialog - now properly session-aware"""
        try:
            print(
                f"🔄 Sidebar: Applying LLM selection - Provider: {provider}, Model: {model}"
            )
            print(f"🔄 Sidebar: Request available: {request is not None}")
            print(
                f"🔄 Sidebar: Main app has session_manager: {hasattr(self.main_app, 'session_manager')}"
            )

            # Use clean session manager for session-aware LLM selection
            if request and hasattr(self.main_app, "session_manager"):
                session_id = self.main_app.session_manager.get_session_id(request)
                print(f"🔄 Sidebar: Session ID: {session_id}")
                success = self.main_app.session_manager.update_llm_provider(
                    session_id, provider, model
                )
                print(f"🔄 Sidebar: Update result: {success}")
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

    def _apply_mistral_with_clear(
        self, provider: str, model: str, request: gr.Request = None
    ) -> str:
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
                    _chatbot, _msg = clear_handler(request)
                    status += f" {self._get_translation('mistral_chat_cleared')}"
                    return status
                # Fallback clear - return empty chat
                status += f" {self._get_translation('mistral_chat_cleared')}"
                return status

            return status

        except Exception as e:
            print(f"Error applying Mistral with clear: {e}")
            return self._get_translation("llm_apply_error")

    def _get_message_input_component(self) -> gr.Textbox:
        """Get the message input component from the main app"""
        if hasattr(self, "main_app") and self.main_app:
            # Try to get from UI Manager components
            if hasattr(self.main_app, "ui_manager") and self.main_app.ui_manager:
                components = self.main_app.ui_manager.get_components()
                return components.get("msg")
        return None

    def connect_quick_action_dropdown(self):
        """Connect the quick action dropdown after all components are available"""
        if "quick_actions_dropdown" in self.components:
            # Get the message input component from the main app
            msg_component = self._get_message_input_component()
            if msg_component:
                self.components["quick_actions_dropdown"].change(
                    fn=self._handle_quick_action_dropdown,
                    inputs=[self.components["quick_actions_dropdown"]],
                    outputs=[msg_component],  # Output to message input
                )
                logging.getLogger(__name__).debug(
                    "✅ Sidebar: Quick action dropdown connected to message input"
                )
            else:
                logging.getLogger(__name__).warning(
                    "⚠️ Sidebar: Message input component not found for quick actions"
                )

    def _update_download_button_visibility(self):
        """Update download button visibility - handled by main app"""
        # This will be handled by the main app's event system
