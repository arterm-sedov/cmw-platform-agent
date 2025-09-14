"""
Logs Tab Module for App NG
=========================

Handles logs monitoring, display, and management functionality.
This module encapsulates all logs-related UI components and functionality.
"""

import gradio as gr
from typing import Dict, Any, Callable, Tuple

class LogsTab:
    """Logs tab component for monitoring and display"""
    
    def __init__(self, event_handlers: Dict[str, Callable]):
        self.event_handlers = event_handlers
        self.components = {}
    
    def create_tab(self) -> Tuple[gr.TabItem, Dict[str, Any]]:
        """
        Create the logs tab with all its components.
        
        Returns:
            Tuple of (TabItem, components_dict)
        """
        print("✅ LogsTab: Creating logs interface...")
        
        with gr.TabItem("📜 Logs", id="logs") as tab:
            # Create logs interface
            self._create_logs_interface()
            
            # Connect event handlers
            self._connect_events()
        
        print("✅ LogsTab: Successfully created with all components and event handlers")
        return tab, self.components
    
    def _create_logs_interface(self):
        """Create the logs monitoring interface"""
        gr.Markdown("### Initialization Logs")
        
        self.components["logs_display"] = gr.Markdown(
            "🟡 Starting initialization...",
            elem_classes=["status-card"]
        )
        
        with gr.Row():
            self.components["refresh_logs_btn"] = gr.Button("🔄 Refresh Logs", elem_classes=["cmw-button"])
            self.components["clear_logs_btn"] = gr.Button("🗑️ Clear Logs", elem_classes=["cmw-button"])
    
    def _connect_events(self):
        """Connect all event handlers for the logs tab"""
        print("🔗 LogsTab: Connecting event handlers...")
        
        # Get event handlers with validation
        refresh_handler = self.event_handlers.get("refresh_logs")
        clear_handler = self.event_handlers.get("clear_logs")
        
        # Validate critical handlers
        if not refresh_handler:
            raise ValueError("refresh_logs handler not found in event_handlers")
        if not clear_handler:
            raise ValueError("clear_logs handler not found in event_handlers")
        
        print("✅ LogsTab: Critical event handlers validated")
        
        self.components["refresh_logs_btn"].click(
            fn=refresh_handler,
            outputs=[self.components["logs_display"]]
        )
        
        self.components["clear_logs_btn"].click(
            fn=clear_handler,
            outputs=[self.components["logs_display"]]
        )
        
        print("✅ LogsTab: All event handlers connected successfully")
    
    def get_components(self) -> Dict[str, Any]:
        """Get all components created by this tab"""
        return self.components
    
    def get_logs_display_component(self) -> gr.Markdown:
        """Get the logs display component for auto-refresh"""
        return self.components["logs_display"]
