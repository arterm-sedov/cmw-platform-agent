"""
Stats Tab Module for App NG
==========================

Handles statistics monitoring, display, and management functionality.
This module encapsulates all statistics-related UI components and functionality.
"""

import gradio as gr
import time
from typing import Dict, Any, Callable, Tuple, Optional

class StatsTab:
    """Stats tab component for statistics and monitoring"""
    
    def __init__(self, event_handlers: Dict[str, Callable]):
        self.event_handlers = event_handlers
        self.components = {}
        self.agent = None  # Will be set by the app
        self._last_conversation_stats = None  # Track last stats for change detection
    
    def create_tab(self) -> Tuple[gr.TabItem, Dict[str, Any]]:
        """
        Create the stats tab with all its components.
        
        Returns:
            Tuple of (TabItem, components_dict)
        """
        print("✅ StatsTab: Creating stats interface...")
        
        with gr.TabItem("📊 Statistics", id="stats") as tab:
            # Create stats interface
            self._create_stats_interface()
            
            # Connect event handlers
            self._connect_events()
        
        print("✅ StatsTab: Successfully created with all components and event handlers")
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
    
    def format_stats_display(self) -> str:
        """Format and return the complete stats display"""
        if not self.agent:
            return "Agent not available"
        
        try:
            # Get basic agent statistics
            stats = self.agent.get_stats()
            
            # Get token statistics
            token_stats = self._format_token_stats()
            
            # Format complete display
            return f"""
**Agent Status:**
- Ready: {stats['agent_status']['is_ready']}
- LLM: {stats['llm_info'].get('model_name', 'Unknown')}
- Provider: {stats['llm_info'].get('provider', 'Unknown')}

**Conversation:**
- Messages: {stats['conversation_stats']['message_count']}
- User: {stats['conversation_stats']['user_messages']}
- Assistant: {stats['conversation_stats']['assistant_messages']}
- Total Messages: {stats['conversation_stats']['message_count']}

**Tools:**
- Available: {stats['agent_status']['tools_count']}{self._format_tools_stats()}{token_stats}
            """
        except Exception as e:
            return f"Error loading statistics: {str(e)}"
    
    def _format_token_stats(self) -> str:
        """Format token usage statistics"""
        if not self.agent:
            return ""
        
        try:
            token_info = self.agent.get_token_display_info()
            cumulative_stats = token_info.get("cumulative_stats", {})
            
            if not cumulative_stats:
                return ""
            
            # Get conversation stats for message count
            # Only show debug messages if stats have changed
            conversation_stats = self.agent._get_conversation_stats(debug=False)
            debug_stats = self._last_conversation_stats != conversation_stats
            if debug_stats:
                # Re-get with debug enabled to show the change
                conversation_stats = self.agent._get_conversation_stats(debug=True)
            self._last_conversation_stats = conversation_stats.copy() if conversation_stats else None
            total_messages = conversation_stats.get('message_count', 0)
            
            return f"""           
**Token Usage:**
- Total (Persistent): {cumulative_stats['conversation_tokens']:,} tokens
- Current Conversation: {cumulative_stats['session_tokens']:,} tokens
- Average per Message: {cumulative_stats['avg_tokens_per_message']} tokens
            """
        except Exception:
            return ""
    
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
                    return f"\n- Used: {len(unique_tools)} unique tools\n- Total Calls: {total_tool_calls}"
                else:
                    return "\n- Used: 0 tools"
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
            return "🟡 Initializing agent..."
        
        status = self.agent.get_status()
        if status["is_ready"]:
            llm_info = self.agent.get_llm_info()
            return f"""✅ **Agent Ready**

**Provider:** {llm_info.get('provider', 'Unknown')}

**Model:** {llm_info.get('model_name', 'Unknown')}

**Status:** {'✅ Healthy' if llm_info.get('is_healthy', False) else '❌ Unhealthy'}

**Tools:** {status['tools_count']} available

**Last Used:** {time.ctime(llm_info.get('last_used', 0))}"""
        else:
            return f"🟡 **Agent Initializing**\n\n**Status:** {status.get('status', 'Unknown')}\n**Tools:** {status.get('tools_count', 0)} available"