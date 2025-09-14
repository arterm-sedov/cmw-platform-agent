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
        
        # Use local methods for logs functionality
        self.components["refresh_logs_btn"].click(
            fn=self.get_initialization_logs,
            outputs=[self.components["logs_display"]]
        )
        
        self.components["clear_logs_btn"].click(
            fn=self.clear_logs,
            outputs=[self.components["logs_display"]]
        )
        
        print("✅ LogsTab: All event handlers connected successfully")
    
    def get_components(self) -> Dict[str, Any]:
        """Get all components created by this tab"""
        return self.components
    
    def get_logs_display_component(self) -> gr.Markdown:
        """Get the logs display component for auto-refresh"""
        return self.components["logs_display"]
    
    # Logs handler methods
    def get_initialization_logs(self) -> str:
        """Get initialization logs as formatted string"""
        # Access the main app through event handlers context
        # This will be set by the main app when creating the tab
        if hasattr(self, '_main_app') and self._main_app:
            # Combine static logs with real-time debug logs
            static_logs = "\n".join(self._main_app.initialization_logs)
            debug_logs = self._main_app.log_handler.get_current_logs()
            
            if debug_logs and debug_logs != "No logs available yet.":
                return f"{static_logs}\n\n--- Real-time Debug Logs ---\n\n{debug_logs}"
            return static_logs
        return "Logs not available - main app not connected"
    
    def clear_logs(self) -> str:
        """Clear logs and return confirmation"""
        if hasattr(self, '_main_app') and self._main_app:
            self._main_app.log_handler.clear_logs()
            return "Logs cleared."
        return "Cannot clear logs - main app not connected"
    
    def set_main_app(self, main_app):
        """Set reference to main app for accessing logs functionality"""
        self._main_app = main_app