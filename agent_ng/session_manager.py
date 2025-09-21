#!/usr/bin/env python3
"""
Session Manager
==============

A clean, modular session manager for handling user session isolation in Gradio applications.
This module provides a lean interface for managing per-user sessions, agents, and state.

Key Features:
- Per-user session isolation
- Agent instance management per session
- Session-aware status and LLM provider tracking
- Clean integration with existing app architecture
- Proper Gradio request handling
"""

from typing import Dict, Optional, Any, Tuple
import gradio as gr
import uuid
import time

# Handle both relative and absolute imports
try:
    from .langchain_agent import CmwAgent
    from .i18n_translations import get_translation_key
except ImportError:
    # Fallback for when running as script
    from agent_ng.langchain_agent import CmwAgent
    from agent_ng.i18n_translations import get_translation_key


class SessionManager:
    """Clean, modular session manager for user isolation"""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.sessions: Dict[str, SessionData] = {}
    
    def get_session_id(self, request: gr.Request = None) -> str:
        """Get or create session ID from Gradio request"""
        if request and hasattr(request, 'session_hash') and request.session_hash:
            return f"gradio_{request.session_hash}"
        elif request and hasattr(request, 'client'):
            return f"client_{id(request.client)}"
        else:
            # Fallback for testing or when no request available
            return f"session_{uuid.uuid4().hex[:16]}_{int(time.time())}"
    
    def get_session_data(self, session_id: str) -> 'SessionData':
        """Get or create session data for the given session ID"""
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionData(session_id, self.language)
        return self.sessions[session_id]
    
    def get_agent(self, session_id: str) -> CmwAgent:
        """Get or create agent instance for the session"""
        session_data = self.get_session_data(session_id)
        return session_data.agent
    
    def get_status(self, session_id: str) -> str:
        """Get session-specific status"""
        session_data = self.get_session_data(session_id)
        return session_data.status
    
    def set_status(self, session_id: str, status: str) -> None:
        """Set session-specific status"""
        session_data = self.get_session_data(session_id)
        session_data.status = status
    
    def get_llm_provider(self, session_id: str) -> str:
        """Get session-specific LLM provider"""
        session_data = self.get_session_data(session_id)
        return session_data.llm_provider
    
    def set_llm_provider(self, session_id: str, provider: str) -> None:
        """Set session-specific LLM provider"""
        session_data = self.get_session_data(session_id)
        session_data.llm_provider = provider
    
    def clear_conversation(self, session_id: str) -> None:
        """Clear conversation for the session"""
        session_data = self.get_session_data(session_id)
        if hasattr(session_data.agent, 'clear_conversation'):
            session_data.agent.clear_conversation(session_id)
        if hasattr(session_data.agent, 'token_tracker') and session_data.agent.token_tracker:
            session_data.agent.token_tracker.start_new_conversation()
        session_data.status = get_translation_key("progress_ready", self.language)
    
    def update_llm_provider(self, session_id: str, provider: str, model: str) -> bool:
        """Update LLM provider for the session"""
        try:
            session_data = self.get_session_data(session_id)
            agent = session_data.agent
            
            if hasattr(agent, 'llm_manager') and agent.llm_manager:
                config = agent.llm_manager.get_provider_config(provider)
                if config and config.models:
                    # Find model index
                    model_index = 0
                    for i, model_config in enumerate(config.models):
                        if model_config["model"] == model:
                            model_index = i
                            break
                    
                    # Get new LLM instance
                    new_llm_instance = agent.llm_manager.get_llm(provider, model_index=model_index)
                    if new_llm_instance:
                        agent.llm_instance = new_llm_instance
                        session_data.llm_provider = provider
                        
                        # Reset token budget
                        if hasattr(agent, 'token_tracker') and agent.token_tracker:
                            agent.token_tracker.reset_current_conversation_budget()
                        
                        return True
            return False
        except Exception as e:
            print(f"Error updating session LLM provider: {e}")
            return False
    
    def format_session_stats(self, session_id: str) -> str:
        """Format session-specific stats display"""
        try:
            session_data = self.get_session_data(session_id)
            agent = session_data.agent
            
            if not agent:
                return get_translation_key("agent_not_available", self.language)
            
            # Get agent statistics
            stats = agent.get_stats()
            llm_info = agent.get_llm_info() if hasattr(agent, 'get_llm_info') else {}
            
            # Build display parts
            display_parts = []
            
            # Agent status
            display_parts.append(f"**🤖 {get_translation_key('agent_status_section', self.language)}**")
            display_parts.append(f"- {get_translation_key('status_ready_true' if stats['agent_status']['is_ready'] else 'status_ready_false', self.language)}")
            display_parts.append(f"- LLM: {llm_info.get('model_name', 'Unknown')}")
            display_parts.append(f"- {get_translation_key('provider_info', self.language).format(provider=llm_info.get('provider', 'Unknown'))}")
            display_parts.append("")
            
            # Conversation stats
            display_parts.append(f"**💬 {get_translation_key('conversation_section', self.language)}:**")
            display_parts.append(f"- {get_translation_key('messages_label', self.language)}: {stats['conversation_stats']['message_count']}")
            display_parts.append(f"- {get_translation_key('user_messages_label', self.language)}: {stats['conversation_stats']['user_messages']}")
            display_parts.append(f"- {get_translation_key('assistant_messages_label', self.language)}: {stats['conversation_stats']['assistant_messages']}")
            display_parts.append(f"- {get_translation_key('total_messages_label', self.language)}: {stats['conversation_stats']['message_count']}")
            display_parts.append("")
            
            # Tools
            display_parts.append(f"**🛠️ {get_translation_key('tools_section', self.language)}:**")
            display_parts.append(f"- {get_translation_key('available_label', self.language)}: {stats['agent_status']['tools_count']}")
            display_parts.append("")
            
            # Token budget - handle gracefully
            display_parts.append(f"**💰 {get_translation_key('token_budget_title', self.language)}**")
            if 'token_stats' in stats and stats['token_stats']:
                token_stats = stats['token_stats']
                display_parts.append(f"{get_translation_key('total_tokens_label', self.language)}: {token_stats.get('total_tokens', 0):,}")
                display_parts.append(f"{get_translation_key('dialog_tokens_label', self.language)}: {token_stats.get('total_tokens', 0):,}")
                display_parts.append(f"{get_translation_key('last_message_tokens_label', self.language)} {token_stats.get('last_message_percentage', 0):.1f}% ({token_stats.get('last_message_tokens', 0):,}/{token_stats.get('context_limit', 0):,}) 🟢")
                display_parts.append(f"{get_translation_key('avg_tokens_per_message_label', self.language)}: {token_stats.get('avg_tokens_per_message', 0):,}")
            else:
                display_parts.append(get_translation_key('token_stats_unavailable', self.language))
            display_parts.append("")
            
            # Progress
            display_parts.append(f"**📊 {get_translation_key('progress_section', self.language)}**")
            display_parts.append(session_data.status)
            
            return "\n".join(display_parts)
            
        except Exception as e:
            return f"{get_translation_key('error_loading_stats', self.language)}: {str(e)}"
    
    def get_session_count(self) -> int:
        """Get total number of active sessions"""
        return len(self.sessions)
    
    def cleanup_inactive_sessions(self, max_age_seconds: int = 3600) -> int:
        """Clean up inactive sessions (placeholder for future implementation)"""
        # TODO: Implement session cleanup based on last activity
        return 0


class SessionData:
    """Data container for individual session state"""
    
    def __init__(self, session_id: str, language: str = "en"):
        self.session_id = session_id
        self.language = language
        self.agent = CmwAgent()
        self.status = get_translation_key("progress_ready", language)
        self.llm_provider = "openrouter"  # Default provider
        self.created_at = time.time()
        self.last_activity = time.time()
    
    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity = time.time()
