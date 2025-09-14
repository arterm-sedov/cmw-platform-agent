"""
UI Manager for App NG
====================

Handles Gradio interface creation, styling, and component management.
This module orchestrates the UI creation process while maintaining all existing functionality.
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any, Callable, List, Tuple
import os

class UIManager:
    """Manages Gradio UI creation and configuration"""
    
    def __init__(self):
        self.css_file_path = Path(__file__).parent.parent / "resources" / "css" / "cmw_copilot_theme.css"
        self._setup_gradio_paths()
        self.components = {}
    
    def _setup_gradio_paths(self):
        """Setup Gradio static resource paths"""
        RESOURCES_DIR = Path(__file__).parent.parent / "resources"
        try:
            existing_allowed = os.environ.get("GRADIO_ALLOWED_PATHS", "")
            parts = [p for p in existing_allowed.split(os.pathsep) if p]
            if str(RESOURCES_DIR) not in parts:
                parts.append(str(RESOURCES_DIR))
            os.environ["GRADIO_ALLOWED_PATHS"] = os.pathsep.join(parts)
            print(f"Gradio static allowed paths: {os.environ['GRADIO_ALLOWED_PATHS']}")
        except Exception as e:
            print(f"Warning: could not set GRADIO_ALLOWED_PATHS: {e}")
    
    def create_interface(self, tab_modules: List[Any], event_handlers: Dict[str, Callable]) -> gr.Blocks:
        """
        Create the main Gradio interface using tab modules.
        
        Args:
            tab_modules: List of tab module instances
            event_handlers: Dictionary of event handlers
            
        Returns:
            Gradio Blocks interface
        """
        print("🏗️ UIManager: Starting interface creation...")
        
        # Clear components to ensure clean state
        self.components.clear()
        
        with gr.Blocks(
            css_paths=[self.css_file_path],
            title="Comindware Analyst Copilot",
            theme=gr.themes.Soft()
        ) as demo:
            
            # Header
            gr.Markdown("# Analyst Copilot", elem_classes=["hero-title"])
            
            with gr.Tabs():
                # Create tabs using provided tab modules
                for tab_module in tab_modules:
                    if tab_module:
                        tab_item, tab_components = tab_module.create_tab()
                        # Consolidate all components in one place
                        self.components.update(tab_components)
                        # Store tab reference for later use
                        self.components[f"{tab_module.__class__.__name__.lower()}_tab"] = tab_module
            
            # Setup auto-refresh timers
            self._setup_auto_refresh(demo, event_handlers)
        
        print("✅ UIManager: Interface created successfully with all components and timers")
        return demo
    
    def _setup_auto_refresh(self, demo: gr.Blocks, event_handlers: Dict[str, Callable]):
        """Setup auto-refresh timers for status and logs - matches original behavior exactly"""
        # Get handlers with validation
        update_status_handler = event_handlers.get("update_status")
        refresh_logs_handler = event_handlers.get("refresh_logs")
        
        # Working hybrid approach - minimal timers that actually work
        # Load initial UI state once on startup
        if "status_display" in self.components and update_status_handler:
            demo.load(
                fn=update_status_handler,
                outputs=[self.components["status_display"]]
            )
            
            # Add minimal timer for status updates during initialization
            status_timer = gr.Timer(2.0, active=True)
            status_timer.tick(
                fn=update_status_handler,
                outputs=[self.components["status_display"]]
            )
        
        if "logs_display" in self.components and refresh_logs_handler:
            demo.load(
                fn=refresh_logs_handler,
                outputs=[self.components["logs_display"]]
            )
            
            # Add minimal timer for logs updates
            logs_timer = gr.Timer(4.0, active=True)
            logs_timer.tick(
                fn=refresh_logs_handler,
                outputs=[self.components["logs_display"]]
            )
        
        refresh_stats_handler = event_handlers.get("refresh_stats")
        if "stats_display" in self.components and refresh_stats_handler:
            demo.load(
                fn=refresh_stats_handler,
                outputs=[self.components["stats_display"]]
            )
            
            # Add minimal timer for stats updates
            stats_timer = gr.Timer(3.0, active=True)
            stats_timer.tick(
                fn=refresh_stats_handler,
                outputs=[self.components["stats_display"]]
            )
    
    def get_components(self) -> Dict[str, Any]:
        """Get all components created by the UI manager"""
        return self.components
    
    def get_component(self, name: str) -> Any:
        """Get a specific component by name"""
        return self.components.get(name)
    
    def set_agent(self, agent):
        """Set the agent reference on all tabs that support it"""
        for key, component in self.components.items():
            if hasattr(component, 'set_agent'):
                component.set_agent(agent)

# Global instance
_ui_manager = None

def get_ui_manager() -> UIManager:
    """Get the global UI manager instance"""
    global _ui_manager
    if _ui_manager is None:
        _ui_manager = UIManager()
    return _ui_manager
