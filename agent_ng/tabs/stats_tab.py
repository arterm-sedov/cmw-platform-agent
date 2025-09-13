"""
Stats Tab Module for App NG
==========================

Handles statistics monitoring, display, and management functionality.
This module encapsulates all statistics-related UI components and functionality.
"""

import gradio as gr
from typing import Dict, Any, Callable, Tuple

class StatsTab:
    """Stats tab component for statistics and monitoring"""
    
    def __init__(self, event_handlers: Dict[str, Callable]):
        self.event_handlers = event_handlers
        self.components = {}
    
    def create_tab(self) -> Tuple[gr.TabItem, Dict[str, Any]]:
        """
        Create the stats tab with all its components.
        
        Returns:
            Tuple of (TabItem, components_dict)
        """
        with gr.TabItem("📊 Statistics", id="stats") as tab:
            # Create stats interface
            self._create_stats_interface()
            
            # Connect event handlers
            self._connect_events()
        
        return tab, self.components
    
    def _create_stats_interface(self):
        """Create the statistics monitoring interface"""
        gr.Markdown("### Agent Statistics")
        
        self.components["stats_display"] = gr.Markdown(
            "Loading statistics...",
            elem_classes=["status-card"]
        )
        
        with gr.Row():
            self.components["refresh_stats_btn"] = gr.Button("🔄 Refresh Stats", elem_classes=["cmw-button"])
    
    def _connect_events(self):
        """Connect all event handlers for the stats tab"""
        self.components["refresh_stats_btn"].click(
            fn=self.event_handlers.get("refresh_stats"),
            outputs=[self.components["stats_display"]]
        )
    
    def get_components(self) -> Dict[str, Any]:
        """Get all components created by this tab"""
        return self.components
    
    def get_stats_display_component(self) -> gr.Markdown:
        """Get the stats display component for auto-refresh"""
        return self.components["stats_display"]
