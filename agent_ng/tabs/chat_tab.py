"""
Chat Tab Module for App NG
=========================

Handles the main chat interface, quick actions, and user interactions.
This module encapsulates all chat-related UI components and functionality.
"""

import gradio as gr
from typing import Dict, Any, Callable, List, Tuple
import asyncio

class ChatTab:
    """Chat tab component with interface and quick actions"""
    
    def __init__(self, event_handlers: Dict[str, Callable]):
        self.event_handlers = event_handlers
        self.components = {}
    
    def create_tab(self) -> Tuple[gr.TabItem, Dict[str, Any]]:
        """
        Create the chat tab with all its components.
        
        Returns:
            Tuple of (TabItem, components_dict)
        """
        print("✅ ChatTab: Creating chat interface...")
        
        with gr.TabItem("💬 Chat", id="chat") as tab:
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
                gr.Markdown("""
                ## 💬 Welcome!
                                                      
                The Comindware Analyst Copilot focuses on the **Comindware Platform** entity operations (applications, templates, attributes) and uses deterministic tools to execute precise changes.

                - **Platform Operations First**: Validates your intent and executes tools for entity changes (e.g., create/edit attributes)
                - **Multi-Model Orchestration**: Tries multiple LLM providers with intelligent fallback
                - **Compact Structured Output**: Intent → Plan → Validate → Execute → Result
                """) 
            with gr.Column(elem_classes=["chat-hints"]):
                gr.Markdown("""
                ## ❓ Try asking:
                
                - List all applications in the Platform
                - List all record templates in app 'ERP'
                - List all attributes in template 'Counterparties', app 'ERP'
                - Create plain text attribute 'Comment', app 'HR', template 'Candidates'
                - Create 'Customer ID' text attribute, app 'ERP', template 'Counterparties', custom input mask: ([0-9]{10}|[0-9]{12})
                - For attribute 'Contact Phone' in app 'CRM', template 'Leads', change display format to Russian phone
                - Fetch attribute: system name 'Comment', app 'HR', template 'Candidates'
                - Archive/unarchive attribute, system name 'Comment', app 'HR', template 'Candidates'
                """) 
                
        with gr.Row():
            with gr.Column(scale=3):
                # Chat interface with metadata support for thinking transparency
                self.components["chatbot"] = gr.Chatbot(
                    label="Chat with the Agent",
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
                        label="Your Message",
                        placeholder="Type your message here...",
                        lines=2,
                        scale=4,
                        max_lines=4,
                        elem_id="message-input",
                        elem_classes=["message-card"]
                    )
                    with gr.Column():
                        self.components["send_btn"] = gr.Button("Send", variant="primary", scale=1, elem_classes=["cmw-button"])
                        self.components["clear_btn"] = gr.Button("Clear Chat", variant="secondary", elem_classes=["cmw-button"])
            
            # Status and Quick Actions sidebar (moved here to be on the right)
            with gr.Column(scale=1):
                # Status section
                with gr.Column(elem_classes=["model-card"]):
                    gr.Markdown("### 🤖 Status")
                    self.components["status_display"] = gr.Markdown("🟡 Initializing...")
                    
                # Quick actions section
                with gr.Column(elem_classes=["quick-actions-card"]):
                    gr.Markdown("### ⚡ Quick Actions")
                    with gr.Column():
                        self.components["quick_list_apps_btn"] = gr.Button("🔎 List all apps", elem_classes=["cmw-button"])
                        self.components["quick_create_attr_btn"] = gr.Button("🧩 Create text attribute", elem_classes=["cmw-button"]) 
                        self.components["quick_edit_mask_btn"] = gr.Button("🛠️ Edit phone mask", elem_classes=["cmw-button"]) 
                        self.components["quick_math_btn"] = gr.Button("🧮 15 * 23 + 7 = ?", elem_classes=["cmw-button"]) 
                        self.components["quick_code_btn"] = gr.Button("💻 Python prime check function", elem_classes=["cmw-button"]) 
                        self.components["quick_explain_btn"] = gr.Button("💭 Explain ML briefly", elem_classes=["cmw-button"])
    
    def _create_sidebar(self):
        """Create the status and quick actions sidebar - now handled in _create_chat_interface"""
        # This method is now empty as the sidebar is created within the chat interface
        pass
    
    def _connect_events(self):
        """Connect all event handlers for the chat tab"""
        print("🔗 ChatTab: Connecting event handlers...")
        
        # Get event handlers with fallbacks
        stream_handler = self.event_handlers.get("stream_message")
        clear_handler = self.event_handlers.get("clear_chat")
        quick_math_handler = self.event_handlers.get("quick_math")
        quick_code_handler = self.event_handlers.get("quick_code")
        quick_explain_handler = self.event_handlers.get("quick_explain")
        quick_create_attr_handler = self.event_handlers.get("quick_create_attr")
        quick_edit_mask_handler = self.event_handlers.get("quick_edit_mask")
        quick_list_apps_handler = self.event_handlers.get("quick_list_apps")
        
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
        
        # Quick action events (with fallbacks)
        if quick_math_handler:
            self.components["quick_math_btn"].click(
                fn=quick_math_handler,
                outputs=[self.components["msg"]]
            )
        
        if quick_code_handler:
            self.components["quick_code_btn"].click(
                fn=quick_code_handler,
                outputs=[self.components["msg"]]
            )
        
        if quick_explain_handler:
            self.components["quick_explain_btn"].click(
                fn=quick_explain_handler,
                outputs=[self.components["msg"]]
            )
        
        if quick_create_attr_handler:
            self.components["quick_create_attr_btn"].click(
                fn=quick_create_attr_handler,
                outputs=[self.components["msg"]]
            )
        
        if quick_edit_mask_handler:
            self.components["quick_edit_mask_btn"].click(
                fn=quick_edit_mask_handler,
                outputs=[self.components["msg"]]
            )
        
        if quick_list_apps_handler:
            self.components["quick_list_apps_btn"].click(
                fn=quick_list_apps_handler,
                outputs=[self.components["msg"]]
            )
        
        print("✅ ChatTab: All event handlers connected successfully")
    
    def get_components(self) -> Dict[str, Any]:
        """Get all components created by this tab"""
        return self.components
    
    def get_status_component(self) -> gr.Markdown:
        """Get the status display component for auto-refresh"""
        return self.components["status_display"]
    
    def get_message_component(self) -> gr.Textbox:
        """Get the message input component for quick actions"""
        return self.components["msg"]
