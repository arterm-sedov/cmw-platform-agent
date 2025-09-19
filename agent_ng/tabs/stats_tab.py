"""
Stats Tab Module for App NG
==========================

Handles statistics monitoring, display, and management functionality.
This module encapsulates all statistics-related UI components and functionality.
Supports internationalization (i18n) with Russian and English translations.
"""

import gradio as gr
import time
from typing import Dict, Any, Callable, Tuple, Optional

class StatsTab:
    """Stats tab component for statistics and monitoring"""
    
    def __init__(self, event_handlers: Dict[str, Callable], language: str = "en", i18n_instance: Optional[gr.I18n] = None):
        self.event_handlers = event_handlers
        self.components = {}
        self.agent = None  # Will be set by the app
        self._last_conversation_stats = None  # Track last stats for change detection
        self.language = language
        self.i18n = i18n_instance
    
    def create_tab(self) -> Tuple[gr.TabItem, Dict[str, Any]]:
        """
        Create the stats tab with all its components.
        
        Returns:
            Tuple of (TabItem, components_dict)
        """
        print("✅ StatsTab: Creating stats interface...")
        
        with gr.TabItem(self._get_translation("tab_stats"), id="stats") as tab:
            # Create stats interface
            self._create_stats_interface()
            
            # Connect event handlers
            self._connect_events()
        
        print("✅ StatsTab: Successfully created with all components and event handlers")
        return tab, self.components
    
    def _create_stats_interface(self):
        """Create the statistics monitoring interface"""
        self.components["stats_title"] = gr.Markdown(f"### {self._get_translation('stats_title')}")
        
        self.components["stats_display"] = gr.Markdown(
            self._get_translation("stats_loading"),
            elem_classes=["status-card"]
        )
        
        with gr.Row():
            self.components["refresh_stats_btn"] = gr.Button(self._get_translation("refresh_stats_button"), elem_classes=["cmw-button"])
    
    def _connect_events(self):
        """Connect all event handlers for the stats tab"""
        print("🔗 StatsTab: Connecting event handlers...")
        
        # Use local methods for stats functionality
        self.components["refresh_stats_btn"].click(
            fn=self.refresh_stats,
            outputs=[self.components["stats_display"]]
        )
        
        print("✅ StatsTab: All event handlers connected successfully")
    
    def get_components(self) -> Dict[str, Any]:
        """Get all components created by this tab"""
        return self.components
    
    def get_stats_display_component(self) -> gr.Markdown:
        """Get the stats display component for auto-refresh"""
        return self.components["stats_display"]
    
    def set_agent(self, agent):
        """Set the agent reference for stats access"""
        self.agent = agent
    
    def _get_translation(self, key: str) -> str:
        """Get a translation for a specific key"""
        # Always use direct translation for now to avoid i18n metadata issues
        from ..i18n_translations import get_translation_key
        return get_translation_key(key, self.language)
    
    def format_stats_display(self) -> str:
        """Format and return the complete stats display"""
        if not self.agent:
            return self._get_translation("agent_not_available")
        
        try:
            # Get basic agent statistics
            stats = self.agent.get_stats()
            
            # Format complete display (token stats removed to avoid duplication with chat tab)
            return f"""
{self._get_translation('agent_status_section')}
- {self._get_translation('status_ready_true' if stats['agent_status']['is_ready'] else 'status_ready_false')}
- LLM: {stats['llm_info'].get('model_name', 'Unknown')}
- {self._get_translation('provider_info').format(provider=stats['llm_info'].get('provider', 'Unknown'))}

{self._get_translation('conversation_section')}
- {self._get_translation('messages_label')}: {stats['conversation_stats']['message_count']}
- {self._get_translation('user_messages_label')}: {stats['conversation_stats']['user_messages']}
- {self._get_translation('assistant_messages_label')}: {stats['conversation_stats']['assistant_messages']}
- {self._get_translation('total_messages_label')}: {stats['conversation_stats']['message_count']}

{self._get_translation('tools_section')}
- {self._get_translation('available_label')}: {stats['agent_status']['tools_count']}{self._format_tools_stats()}
            """
        except Exception as e:
            return f"{self._get_translation('error_loading_stats')}: {str(e)}"
    
    
    def _format_tools_stats(self) -> str:
        """Format tools usage statistics"""
        if not self.agent:
            return ""
        
        try:
            # Get tool usage from memory manager
            if hasattr(self.agent, 'memory_manager') and self.agent.memory_manager:
                total_tool_calls = 0
                unique_tools = set()
                
                for conversation_id, conversation in self.agent.memory_manager.memories.items():
                    for message in conversation:
                        if hasattr(message, 'type') and message.type == "tool":
                            total_tool_calls += 1
                            # Try to get tool name from message
                            if hasattr(message, 'name'):
                                unique_tools.add(message.name)
                
                if total_tool_calls > 0:
                    return f"\n- {self._get_translation('used_label')}: {len(unique_tools)} {self._get_translation('unique_tools_label')}\n- {self._get_translation('total_calls_label')}: {total_tool_calls}"
                else:
                    return f"\n- {self._get_translation('used_label')}: 0 {self._get_translation('unique_tools_label')}"
            else:
                return ""
        except Exception:
            return ""
    
    # Stats handler methods
    def refresh_stats(self) -> str:
        """Refresh and return agent statistics"""
        return self.format_stats_display()
    
    def get_agent_status(self) -> str:
        """Get current agent status with comprehensive details"""
        if not self.agent:
            return self._get_translation("agent_initializing")
        
        status = self.agent.get_status()
        if status["is_ready"]:
            llm_info = self.agent.get_llm_info()
            return f"""{self._get_translation('agent_status_ready')}

{self._get_translation('provider_info').format(provider=llm_info.get('provider', 'Unknown'))}

{self._get_translation('model_info').format(model=llm_info.get('model_name', 'Unknown'))}

{self._get_translation('status_label')} {self._get_translation('healthy_status') if llm_info.get('is_healthy', False) else self._get_translation('unhealthy_status')}

{self._get_translation('tools_label')} {status['tools_count']} {self._get_translation('available_label')}

{self._get_translation('last_used_label')} {time.ctime(llm_info.get('last_used', 0))}"""
        else:
            return f"{self._get_translation('agent_status_initializing')}\n\n{self._get_translation('status_label')} {status.get('status', 'Unknown')}\n{self._get_translation('tools_label')} {status.get('tools_count', 0)} {self._get_translation('available_label')}"