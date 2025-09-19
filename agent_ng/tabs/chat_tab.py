"""
Chat Tab Module for App NG
=========================

Handles the main chat interface, quick actions, and user interactions.
This module encapsulates all chat-related UI components and functionality.
Supports internationalization (i18n) with Russian and English translations.
"""

import gradio as gr
from typing import Dict, Any, Callable, List, Tuple, Optional
import asyncio

class ChatTab:
    """Chat tab component with interface and quick actions"""
    
    def __init__(self, event_handlers: Dict[str, Callable], language: str = "en", i18n_instance: Optional[gr.I18n] = None):
        self.event_handlers = event_handlers
        self.components = {}
        self.main_app = None  # Reference to main app for progress status
        self.language = language
        self.i18n = i18n_instance
    
    def create_tab(self) -> Tuple[gr.TabItem, Dict[str, Any]]:
        """
        Create the chat tab with all its components.
        
        Returns:
            Tuple of (TabItem, components_dict)
        """
        print("✅ ChatTab: Creating chat interface...")
        
        with gr.TabItem(self._get_translation("tab_chat"), id="chat") as tab:
            # Create main chat interface (includes sidebar)
            self._create_chat_interface()
            
            # Connect event handlers
            self._connect_events()
        
        print("✅ ChatTab: Successfully created with all components and event handlers")
        return tab, self.components
    
    def _create_chat_interface(self):
        """Create the main chat interface with proper layout"""
        with gr.Row():
            with gr.Column(elem_classes=["chat-hints"]):
                gr.Markdown(f"## {self._get_translation('welcome_title')}", elem_classes=["chat-hints-title"])

                gr.Markdown(self._get_translation("welcome_description")) 
            
            # Prompt examples section (quick action buttons)
            with gr.Column(elem_classes=["quick-actions-card"]):
                gr.Markdown(f"## {self._get_translation('quick_actions_title')}", elem_classes=["chat-hints-title"])
                
                self.components["quick_list_apps_btn"] = gr.Button(self._get_translation("quick_list_apps"), elem_classes=["cmw-button"])
                self.components["quick_templates_erp_btn"] = gr.Button(self._get_translation("quick_templates_erp"), elem_classes=["cmw-button"])
                self.components["quick_attributes_contractors_btn"] = gr.Button(self._get_translation("quick_attributes_contractors"), elem_classes=["cmw-button"])
                self.components["quick_edit_date_time_btn"] = gr.Button(self._get_translation("quick_edit_date_time"), elem_classes=["cmw-button"])
                self.components["quick_create_comment_attr_btn"] = gr.Button(self._get_translation("quick_create_comment_attr"), elem_classes=["cmw-button"])
                self.components["quick_create_id_attr_btn"] = gr.Button(self._get_translation("quick_create_id_attr"), elem_classes=["cmw-button"])
                self.components["quick_edit_phone_mask_btn"] = gr.Button(self._get_translation("quick_edit_phone_mask"), elem_classes=["cmw-button"])
            
            # Quick actions section
            with gr.Column(elem_classes=["quick-actions-card"]):
            
                self.components["quick_edit_enum_btn"] = gr.Button(self._get_translation("quick_edit_enum"), elem_classes=["cmw-button"])
                self.components["quick_get_comment_attr_btn"] = gr.Button(self._get_translation("quick_get_comment_attr"), elem_classes=["cmw-button"])
                self.components["quick_create_attr_btn"] = gr.Button(self._get_translation("quick_create_attr"), elem_classes=["cmw-button"]) 
                self.components["quick_edit_mask_btn"] = gr.Button(self._get_translation("quick_edit_mask"), elem_classes=["cmw-button"]) 
                self.components["quick_archive_attr_btn"] = gr.Button(self._get_translation("quick_archive_attr"), elem_classes=["cmw-button"])
                self.components["quick_math_btn"] = gr.Button(self._get_translation("quick_math"), elem_classes=["cmw-button"]) 
                self.components["quick_code_btn"] = gr.Button(self._get_translation("quick_code"), elem_classes=["cmw-button"]) 
                self.components["quick_explain_btn"] = gr.Button(self._get_translation("quick_explain"), elem_classes=["cmw-button"])
            
        with gr.Row():
            with gr.Column(scale=3):
                # Chat interface with metadata support for thinking transparency
                self.components["chatbot"] = gr.Chatbot(
                    label=self._get_translation("chat_label"),
                    height=500,
                    show_label=True,
                    container=True,
                    show_copy_button=True,
                    type="messages",
                    elem_id="chatbot-main",
                    elem_classes=["chatbot-card"]
                )
                
                with gr.Row():
                    self.components["msg"] = gr.Textbox(
                        label=self._get_translation("message_label"),
                        placeholder=self._get_translation("message_placeholder"),
                        lines=2,
                        scale=4,
                        max_lines=4,
                        elem_id="message-input",
                        elem_classes=["message-card"]
                    )
                    with gr.Column():
                        self.components["send_btn"] = gr.Button(self._get_translation("send_button"), variant="primary", scale=1, elem_classes=["cmw-button"])
                        self.components["clear_btn"] = gr.Button(self._get_translation("clear_button"), variant="secondary", elem_classes=["cmw-button"])
                        self.components["download_btn"] = gr.Button(self._get_translation("download_button"), variant="secondary", elem_classes=["cmw-button"])
                        
                        # State variables for file download
                        self.components["file_ready"] = gr.State(False)
                        self.components["file_path"] = gr.State(None)
                        
                        # Static download file component (no render decorator to avoid duplicate IDs)
                        self.components["download_file"] = self._create_download_file_component()
                        
                
            
            # Status and Quick Actions sidebar (moved here to be on the right)
            with gr.Column(scale=1):
                # LLM Selection section
                with gr.Column(elem_classes=["model-card"]):
                    gr.Markdown(f"### {self._get_translation('llm_selection_title')}", elem_classes=["llm-selection-title"])
                    
                    # Combined Provider/Model selector
                    self.components["provider_model_selector"] = gr.Dropdown(
                        choices=self._get_available_provider_model_combinations(),
                        value=self._get_current_provider_model_combination(),
                        label=self._get_translation("provider_model_label"),
                        interactive=True,
                        allow_custom_value=True,
                        elem_classes=["provider-model-selector"]
                    )
                    
                    # Apply button
                    self.components["apply_llm_btn"] = gr.Button(
                        self._get_translation("apply_llm_button"),
                        variant="primary",
                        elem_classes=["cmw-button"]
                    )
                
                # Status section
                with gr.Column(elem_classes=["model-card"]):
                    gr.Markdown(f"### {self._get_translation('status_title')}", elem_classes=["status-title"])
                    self.components["status_display"] = gr.Markdown(self._get_translation("status_initializing"))
                    
                    # Token budget indicator
                    gr.Markdown(f"### {self._get_translation('token_budget_title')}", elem_classes=["token-budget-title"])
                    self.components["token_budget_display"] = gr.Markdown(
                        self._get_translation("token_budget_initializing")
                    )
                    
                    # Progress indicator
                    gr.Markdown(f"### {self._get_translation('progress_title')}", elem_classes=["progress-title"])
                    self.components["progress_display"] = gr.Markdown(
                        self._get_translation("progress_ready")
                    )
    
    def _create_sidebar(self):
        """Create the status and quick actions sidebar - now handled in _create_chat_interface"""
        # This method is now empty as the sidebar is created within the chat interface
        pass
    
    def _connect_events(self):
        """Connect all event handlers for the chat tab"""
        print("🔗 ChatTab: Connecting event handlers...")
        
        # Get critical event handlers
        stream_handler = self.event_handlers.get("stream_message")
        clear_handler = self.event_handlers.get("clear_chat")
        
        # Validate critical handlers
        if not stream_handler:
            raise ValueError("stream_message handler not found in event_handlers")
        if not clear_handler:
            raise ValueError("clear_chat handler not found in event_handlers")
        
        print("✅ ChatTab: Critical event handlers validated")
        
        # Main chat events
        self.components["send_btn"].click(
            fn=stream_handler,
            inputs=[self.components["msg"], self.components["chatbot"]],
            outputs=[self.components["chatbot"], self.components["msg"]]
        )
        
        self.components["msg"].submit(
            fn=stream_handler,
            inputs=[self.components["msg"], self.components["chatbot"]],
            outputs=[self.components["chatbot"], self.components["msg"]]
        )
        
        self.components["clear_btn"].click(
            fn=self._clear_chat_with_download_reset,
            outputs=[self.components["chatbot"], self.components["msg"], self.components["file_ready"], self.components["file_path"], self.components["download_file"]]
        )
        
        # Download button event - updates both state variables and file component
        self.components["download_btn"].click(
            fn=self._download_conversation_wrapper,
            inputs=[self.components["chatbot"]],
            outputs=[self.components["file_ready"], self.components["file_path"], self.components["download_file"]]
        )
        
        # Trigger UI updates after chat events
        self._setup_chat_event_triggers()
        
        # Quick action events (using local methods)
        self.components["quick_math_btn"].click(
            fn=self._quick_math,
            outputs=[self.components["msg"]]
        )
        
        self.components["quick_code_btn"].click(
            fn=self._quick_code,
            outputs=[self.components["msg"]]
        )
        
        self.components["quick_explain_btn"].click(
            fn=self._quick_explain,
            outputs=[self.components["msg"]]
        )
        
        self.components["quick_create_attr_btn"].click(
            fn=self._quick_create_attr,
            outputs=[self.components["msg"]]
        )
        
        self.components["quick_edit_mask_btn"].click(
            fn=self._quick_edit_mask,
            outputs=[self.components["msg"]]
        )
        
        self.components["quick_list_apps_btn"].click(
            fn=self._quick_list_apps,
            outputs=[self.components["msg"]]
        )
        
        # Query example button events
        self.components["quick_edit_enum_btn"].click(
            fn=self._quick_edit_enum,
            outputs=[self.components["msg"]]
        )
        
        self.components["quick_templates_erp_btn"].click(
            fn=self._quick_templates_erp,
            outputs=[self.components["msg"]]
        )
        
        self.components["quick_attributes_contractors_btn"].click(
            fn=self._quick_attributes_contractors,
            outputs=[self.components["msg"]]
        )
        
        self.components["quick_create_comment_attr_btn"].click(
            fn=self._quick_create_comment_attr,
            outputs=[self.components["msg"]]
        )
        
        self.components["quick_create_id_attr_btn"].click(
            fn=self._quick_create_id_attr,
            outputs=[self.components["msg"]]
        )
        
        self.components["quick_edit_phone_mask_btn"].click(
            fn=self._quick_edit_phone_mask,
            outputs=[self.components["msg"]]
        )
        
        self.components["quick_get_comment_attr_btn"].click(
            fn=self._quick_get_comment_attr,
            outputs=[self.components["msg"]]
        )
        
        self.components["quick_edit_date_time_btn"].click(
            fn=self._quick_edit_date_time,
            outputs=[self.components["msg"]]
        )
        
        self.components["quick_archive_attr_btn"].click(
            fn=self._quick_archive_attr,
            outputs=[self.components["msg"]]
        )
        
        # LLM selection events
        if "apply_llm_btn" in self.components and "provider_model_selector" in self.components and "status_display" in self.components:
            self.components["apply_llm_btn"].click(
                fn=self._apply_llm_selection_combined,
                inputs=[self.components["provider_model_selector"]],
                outputs=[self.components["status_display"]]
            )
        
        print("✅ ChatTab: All event handlers connected successfully")
    
    def _setup_chat_event_triggers(self):
        """Setup event triggers to update other UI components when chat events occur"""
        # Get UI update handlers
        trigger_ui_update = self.event_handlers.get("trigger_ui_update")
        
        if trigger_ui_update:
            # Trigger UI update after send button click
            self.components["send_btn"].click(
                fn=trigger_ui_update,
                outputs=[]  # No specific outputs, just triggers the update
            )
            
            # Trigger UI update after message submit
            self.components["msg"].submit(
                fn=trigger_ui_update,
                outputs=[]
            )
            
            # Trigger UI update after clear button click
            self.components["clear_btn"].click(
                fn=trigger_ui_update,
                outputs=[]
            )
            
            print("✅ ChatTab: UI update triggers connected")
    
    def get_components(self) -> Dict[str, Any]:
        """Get all components created by this tab"""
        return self.components
    
    def get_status_component(self) -> gr.Markdown:
        """Get the status display component for auto-refresh"""
        return self.components["status_display"]
    
    def get_message_component(self) -> gr.Textbox:
        """Get the message input component for quick actions"""
        return self.components["msg"]
    
    def set_main_app(self, app):
        """Set reference to main app for accessing progress status"""
        self.main_app = app
    
    def get_progress_display(self) -> gr.Markdown:
        """Get the progress display component"""
        return self.components["progress_display"]
    
    def get_token_budget_display(self) -> gr.Markdown:
        """Get the token budget display component"""
        return self.components["token_budget_display"]
    
    def get_llm_selection_components(self) -> Dict[str, Any]:
        """Get LLM selection components for UI updates"""
        return {
            "provider_selector": self.components.get("provider_selector"),
            "model_selector": self.components.get("model_selector"),
            "apply_llm_btn": self.components.get("apply_llm_btn")
        }
    
    def format_token_budget_display(self) -> str:
        """Format and return the token budget display"""
        if not hasattr(self, 'main_app') or not self.main_app or not self.main_app.agent:
            return self._get_translation("token_budget_initializing")
        
        try:
            budget_info = self.main_app.agent.get_token_budget_info()
            
            if budget_info["status"] == "unknown":
                return self._get_translation("token_budget_unknown")
            
            # Get cumulative stats for detailed display
            cumulative_stats = self.main_app.agent.token_tracker.get_cumulative_stats()
            
            # Determine status icon using localized translations
            status_icon = self._get_translation(f"token_status_{budget_info['status']}")
            
            # Build token usage display using separated components for better flexibility
            total = self._get_translation("token_usage_total").format(
                total_tokens=cumulative_stats["conversation_tokens"]
            )
            conversation = self._get_translation("token_usage_conversation").format(
                conversation_tokens=cumulative_stats["conversation_tokens"]
            )
            last_message = self._get_translation("token_usage_last_message").format(
                percentage=budget_info["percentage"],
                used=budget_info["used_tokens"],
                context_window=budget_info["context_window"],
                status_icon=status_icon
            )
            average = self._get_translation("token_usage_average").format(
                avg_tokens=cumulative_stats["avg_tokens_per_message"]
            )
            
            return f"- {total}\n- {conversation}\n- {last_message}\n- {average}"
        except Exception as e:
            print(f"Error formatting token budget: {e}")
            return self._get_translation("token_budget_unknown")
    
    def _get_available_providers(self) -> List[str]:
        """Get list of available LLM providers"""
        if not hasattr(self, 'main_app') or not self.main_app or not self.main_app.agent:
            return ["openrouter", "groq", "gemini", "mistral", "huggingface", "gigachat"]
        
        try:
            if hasattr(self.main_app.agent, 'llm_manager') and self.main_app.agent.llm_manager:
                return self.main_app.agent.llm_manager.get_available_providers()
        except Exception as e:
            print(f"Error getting available providers: {e}")
        
        return ["openrouter", "groq", "gemini", "mistral", "huggingface", "gigachat"]
    
    def _get_current_provider(self) -> str:
        """Get current LLM provider"""
        import os
        return os.environ.get("AGENT_PROVIDER", "openrouter")
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models for the current provider"""
        if not hasattr(self, 'main_app') or not self.main_app or not self.main_app.agent:
            return []
        
        try:
            if hasattr(self.main_app.agent, 'llm_manager') and self.main_app.agent.llm_manager:
                current_provider = self._get_current_provider()
                config = self.main_app.agent.llm_manager.get_provider_config(current_provider)
                if config and config.models:
                    return [model["model"] for model in config.models]
        except Exception as e:
            print(f"Error getting available models: {e}")
        
        return []
    
    def _get_available_provider_model_combinations(self) -> List[str]:
        """Get list of available provider/model combinations in format 'Provider / Model'"""
        if not hasattr(self, 'main_app') or not self.main_app or not self.main_app.agent:
            return [""]
        
        try:
            if hasattr(self.main_app.agent, 'llm_manager') and self.main_app.agent.llm_manager:
                combinations = []
                available_providers = self.main_app.agent.llm_manager.get_available_providers()
                
                for provider in available_providers:
                    config = self.main_app.agent.llm_manager.get_provider_config(provider)
                    if config and config.models:
                        for model in config.models:
                            model_name = model["model"]
                            # Format as "Provider / Model"
                            combination = f"{provider.title()} / {model_name}"
                            combinations.append(combination)
                
                return combinations
        except Exception as e:
            print(f"Error getting provider/model combinations: {e}")
        
        # Return fallback combinations on error
        return [
            "Openrouter / openrouter/anthropic/claude-3.5-sonnet",
            "Groq / groq/compound",
            "Gemini / gemini-2.5-pro",
            "Mistral / mistral-large-latest",
            "Huggingface / microsoft/DialoGPT-medium",
            "Gigachat / gigachat"
        ]
    
    def _get_current_model(self) -> str:
        """Get current LLM model"""
        if not hasattr(self, 'main_app') or not self.main_app or not self.main_app.agent:
            return ""
        
        try:
            if hasattr(self.main_app.agent, 'llm_instance') and self.main_app.agent.llm_instance:
                return self.main_app.agent.llm_instance.model_name
        except Exception as e:
            print(f"Error getting current model: {e}")
        
        return ""
    
    def _get_current_provider_model_combination(self) -> str:
        """Get current provider/model combination in format 'Provider / Model'"""
        if not hasattr(self, 'main_app') or not self.main_app or not self.main_app.agent:
            # Return fallback value when main app is not available
            import os
            provider = os.environ.get("AGENT_PROVIDER", "openrouter")
            return f"{provider.title()} / {provider}/default-model"
        
        try:
            if hasattr(self.main_app.agent, 'llm_instance') and self.main_app.agent.llm_instance:
                provider = self.main_app.agent.llm_instance.provider.value
                model = self.main_app.agent.llm_instance.model_name
                return f"{provider.title()} / {model}"
        except Exception as e:
            print(f"Error getting current provider/model combination: {e}")
        
        # Return fallback value on error
        import os
        provider = os.environ.get("AGENT_PROVIDER", "openrouter")
        return f"{provider.title()} / {provider}/default-model"
    
    def _update_models_for_provider(self, provider: str) -> List[str]:
        """Update available models when provider changes"""
        try:
            if not hasattr(self, 'main_app') or not self.main_app or not self.main_app.agent:
                return []
            
            if hasattr(self.main_app.agent, 'llm_manager') and self.main_app.agent.llm_manager:
                config = self.main_app.agent.llm_manager.get_provider_config(provider)
                if config and config.models:
                    return [model["model"] for model in config.models]
        except Exception as e:
            print(f"Error updating models for provider {provider}: {e}")
        
        return []
    
    def _apply_llm_selection(self, provider: str, model: str) -> str:
        """Apply the selected LLM provider and model"""
        try:
            if not hasattr(self, 'main_app') or not self.main_app:
                return self._get_translation("llm_apply_error")
            
            # Update environment variable
            import os
            os.environ["AGENT_PROVIDER"] = provider
            
            # Reinitialize the agent with new LLM
            if hasattr(self.main_app, 'agent') and self.main_app.agent:
                # Find the model index
                if hasattr(self.main_app.agent, 'llm_manager') and self.main_app.agent.llm_manager:
                    config = self.main_app.agent.llm_manager.get_provider_config(provider)
                    if config and config.models:
                        model_index = 0
                        for i, model_config in enumerate(config.models):
                            if model_config["model"] == model:
                                model_index = i
                                break
                        
                        # Get new LLM instance
                        new_llm_instance = self.main_app.agent.llm_manager.get_llm(provider, model_index=model_index)
                        if new_llm_instance:
                            self.main_app.agent.llm_instance = new_llm_instance
                            
                            # Reset token budget for the new model
                            if hasattr(self.main_app.agent, 'token_tracker') and self.main_app.agent.token_tracker:
                                self.main_app.agent.token_tracker.reset_current_conversation_budget()
                            
                            return self._get_translation("llm_apply_success").format(provider=provider, model=model)
                        else:
                            return self._get_translation("llm_apply_error")
            
            return self._get_translation("llm_apply_error")
        except Exception as e:
            print(f"Error applying LLM selection: {e}")
            return self._get_translation("llm_apply_error")
    
    def _apply_llm_selection_combined(self, provider_model_combination: str) -> str:
        """Apply the selected LLM provider/model combination"""
        try:
            if not provider_model_combination or " / " not in provider_model_combination:
                return self._get_translation("llm_apply_error")
            
            # Parse the combination: "Provider / Model"
            parts = provider_model_combination.split(" / ", 1)
            if len(parts) != 2:
                return self._get_translation("llm_apply_error")
            
            provider = parts[0].lower()  # Convert to lowercase for environment variable
            model = parts[1]
            
            if not hasattr(self, 'main_app') or not self.main_app:
                return self._get_translation("llm_apply_error")
            
            # Update environment variable
            import os
            os.environ["AGENT_PROVIDER"] = provider
            
            # Reinitialize the agent with new LLM
            if hasattr(self.main_app, 'agent') and self.main_app.agent:
                # Find the model index
                if hasattr(self.main_app.agent, 'llm_manager') and self.main_app.agent.llm_manager:
                    config = self.main_app.agent.llm_manager.get_provider_config(provider)
                    if config and config.models:
                        model_index = 0
                        for i, model_config in enumerate(config.models):
                            if model_config["model"] == model:
                                model_index = i
                                break
                        
                        # Get new LLM instance
                        new_llm_instance = self.main_app.agent.llm_manager.get_llm(provider, model_index=model_index)
                        if new_llm_instance:
                            self.main_app.agent.llm_instance = new_llm_instance
                            
                            # Reset token budget for the new model
                            if hasattr(self.main_app.agent, 'token_tracker') and self.main_app.agent.token_tracker:
                                self.main_app.agent.token_tracker.reset_current_conversation_budget()
                            
                            return self._get_translation("llm_apply_success").format(provider=provider.title(), model=model)
                        else:
                            return self._get_translation("llm_apply_error")
            
            return self._get_translation("llm_apply_error")
        except Exception as e:
            print(f"Error applying LLM selection: {e}")
            return self._get_translation("llm_apply_error")
    
    def _get_translation(self, key: str) -> str:
        """Get a translation for a specific key"""
        # Always use direct translation for now to avoid i18n metadata issues
        from ..i18n_translations import get_translation_key
        return get_translation_key(key, self.language)
    
    def _create_download_file_component(self, value=None, visible=False):
        """Create a download file component with consistent parameters"""
        return gr.File(
            value=value,
            label=self._get_translation("download_file_label"),
            visible=visible,
            interactive=False,
            elem_classes=["download-file-pane"]
        )
    
    # Quick action methods
    def _quick_math(self) -> str:
        """Generate math quick action message"""
        return self._get_translation("quick_math_message")
    
    def _quick_code(self) -> str:
        """Generate code quick action message"""
        return self._get_translation("quick_code_message")
    
    def _quick_explain(self) -> str:
        """Generate explain quick action message"""
        return self._get_translation("quick_explain_message")
    
    def _quick_create_attr(self) -> str:
        """Generate create attribute quick action message"""
        return self._get_translation("quick_create_attr_message")
    
    def _quick_edit_mask(self) -> str:
        """Generate edit mask quick action message"""
        return self._get_translation("quick_edit_mask_message")
    
    def _quick_list_apps(self) -> str:
        """Generate list apps quick action message"""
        return self._get_translation("quick_list_apps_message")
    
    # Query example methods
    def _quick_edit_enum(self) -> str:
        """Generate query list apps message"""
        return self._get_translation("quick_edit_enum_message")
    
    def _quick_templates_erp(self) -> str:
        """Generate query templates ERP message"""
        return self._get_translation("quick_templates_erp_message")
    
    def _quick_attributes_contractors(self) -> str:
        """Generate query attributes contractors message"""
        return self._get_translation("quick_attributes_contractors_message")
    
    def _quick_create_comment_attr(self) -> str:
        """Generate query create comment attribute message"""
        return self._get_translation("quick_create_comment_attr_message")
    
    def _quick_create_id_attr(self) -> str:
        """Generate query create ID attribute message"""
        return self._get_translation("quick_create_id_attr_message")
    
    def _quick_edit_phone_mask(self) -> str:
        """Generate query edit phone mask message"""
        return self._get_translation("quick_edit_phone_mask_message")
    
    def _quick_get_comment_attr(self) -> str:
        """Generate query get comment attribute message"""
        return self._get_translation("quick_get_comment_attr_message")
    
    def _quick_edit_date_time(self) -> str:
        """Generate query enum add value message"""
        return self._get_translation("quick_edit_date_time_message")
    
    def _quick_archive_attr(self) -> str:
        """Generate query archive attribute message"""
        return self._get_translation("quick_archive_attr_message")
    
    def _clear_chat_with_download_reset(self):
        """Clear chat and reset download state"""
        # Get the clear handler from event handlers
        clear_handler = self.event_handlers.get("clear_chat")
        if clear_handler:
            # Call the original clear handler
            chatbot, msg = clear_handler()
            # Reset download state and file component
            return chatbot, msg, False, None, self._create_download_file_component()
        else:
            # Fallback if clear handler not available
            return [], "", False, None, self._create_download_file_component()
    
    def _download_conversation_wrapper(self, history):
        """Wrapper to handle the download and update state variables and file component"""
        file_path = self._download_conversation_as_markdown(history)
        if file_path:
            # Return state variables and updated file component
            return True, file_path, self._create_download_file_component(value=file_path, visible=True)
        else:
            # Return state variables and hidden file component
            return False, None, self._create_download_file_component()
    
    def _download_conversation_as_markdown(self, history) -> str:
        """
        Download the conversation history as a markdown file.
        
        Args:
            history: List of conversation messages from Gradio chatbot component
            
        Returns:
            File path if successful, None if failed
        """
        import os
        import tempfile
        from datetime import datetime
        
        print(f"DEBUG: Download function called with history type: {type(history)}")
        print(f"DEBUG: History content: {history}")
        
        if not history:
            print("DEBUG: No history provided")
            return None
        
        # Create timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"CMW_Copilot_{timestamp}.md"
        
        # Create markdown content
        markdown_content = f"# CMW Platform Agent - Conversation Export\n\n"
        markdown_content += f"**Exported on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        markdown_content += f"**Total messages:** {len(history)}\n\n"
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
                markdown_content += f"{str(message)}\n\n"
        
        # Create file with proper filename
        try:
            # Create a temporary directory and file with the proper filename
            temp_dir = tempfile.mkdtemp()
            clean_file_path = os.path.join(temp_dir, filename)
            
            with open(clean_file_path, 'w', encoding='utf-8') as file:
                file.write(markdown_content)
            
            print(f"DEBUG: Created file: {clean_file_path}")
            # Return the clean file path for Gradio to handle the download
            return clean_file_path
        except Exception as e:
            print(f"Error creating markdown file: {e}")
            return None