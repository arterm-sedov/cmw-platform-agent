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
                gr.Markdown(f"## {self._get_translation("welcome_title")}")

                gr.Markdown(self._get_translation("welcome_description")) 
            
            # Prompt examples section (quick action buttons)
            with gr.Column(elem_classes=["quick-actions-card"]):
                gr.Markdown(f"## {self._get_translation('quick_actions_title')}")
                
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
            
            # Status and Quick Actions sidebar (moved here to be on the right)
            with gr.Column(scale=1):
                # Status section
                with gr.Column(elem_classes=["model-card"]):
                    gr.Markdown(f"### {self._get_translation('status_title')}")
                    self.components["status_display"] = gr.Markdown(self._get_translation("status_initializing"))
                    
                    # Progress indicator
                    gr.Markdown(f"### {self._get_translation('progress_title')}")
                    self.components["progress_display"] = gr.Markdown(
                        self._get_translation("progress_ready"), 
                        elem_classes=["progress-status"]
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
            fn=clear_handler,
            outputs=[self.components["chatbot"], self.components["msg"]]
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
    
    def _get_translation(self, key: str) -> str:
        """Get a translation for a specific key"""
        # Always use direct translation for now to avoid i18n metadata issues
        from ..i18n_translations import get_translation_key
        return get_translation_key(key, self.language)
    
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