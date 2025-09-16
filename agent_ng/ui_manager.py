"""
UI Manager for App NG
====================

Handles Gradio interface creation, styling, and component management.
This module orchestrates the UI creation process while maintaining all existing functionality.
Supports internationalization (i18n) with Russian and English translations.
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any, Callable, List, Tuple, Optional
import os

class UIManager:
    """Manages Gradio UI creation and configuration with i18n support"""
    
    def __init__(self, language: str = "en", i18n_instance: Optional[gr.I18n] = None):
        self.css_file_path = Path(__file__).parent.parent / "resources" / "css" / "cmw_copilot_theme.css"
        self.language = language
        self.i18n = i18n_instance
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
        Create the main Gradio interface using tab modules with i18n support.
        
        Args:
            tab_modules: List of tab module instances
            event_handlers: Dictionary of event handlers
            
        Returns:
            Gradio Blocks interface
        """
        print("🏗️ UIManager: Starting interface creation...")
        
        # Clear components to ensure clean state
        self.components.clear()
        
        # Get translated title
        app_title = self._get_translation("app_title")
        hero_title = self._get_translation("hero_title")
        
        with gr.Blocks(
            css_paths=[self.css_file_path],
            title=app_title,
            theme=gr.themes.Soft()
        ) as demo:
            
            # Header
            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"# {hero_title}", elem_classes=["hero-title"])
            
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
        update_progress_handler = event_handlers.get("update_progress_display")
        
        
        # Load initial UI state once on startup - no timers to avoid queue issues
        if "status_display" in self.components and update_status_handler:
            demo.load(
                fn=update_status_handler,
                outputs=[self.components["status_display"]]
            )
        
        if "logs_display" in self.components and refresh_logs_handler:
            demo.load(
                fn=refresh_logs_handler,
                outputs=[self.components["logs_display"]]
            )
        
        if "progress_display" in self.components and update_progress_handler:
            demo.load(
                fn=update_progress_handler,
                outputs=[self.components["progress_display"]]
            )
        
        refresh_stats_handler = event_handlers.get("refresh_stats")
        if "stats_display" in self.components and refresh_stats_handler:
            demo.load(
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
    
    def _get_translation(self, key: str) -> str:
        """Get a translation for a specific key"""
        # Always use direct translation to avoid i18n metadata issues
        from .i18n_translations import get_translation_key
        return get_translation_key(key, self.language)

# Global instances for different languages
_ui_manager_en = None
_ui_manager_ru = None

def get_ui_manager(language: str = "en", i18n_instance: Optional[gr.I18n] = None) -> UIManager:
    """
    Get the global UI manager instance for the specified language.
    
    Args:
        language: Language code ('en' or 'ru')
        i18n_instance: Optional Gradio I18n instance
        
    Returns:
        UIManager instance
    """
    global _ui_manager_en, _ui_manager_ru
    
    if language.lower() == "ru":
        if _ui_manager_ru is None:
            _ui_manager_ru = UIManager(language="ru", i18n_instance=i18n_instance)
        return _ui_manager_ru
    else:
        if _ui_manager_en is None:
            _ui_manager_en = UIManager(language="en", i18n_instance=i18n_instance)
        return _ui_manager_en
