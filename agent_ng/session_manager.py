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
                    
                    # Create a NEW LLM instance for this session (not shared)
                    new_llm_instance = agent.llm_manager.create_new_llm_instance(provider, model_index)
                    if new_llm_instance:
                        # Update the agent's LLM instance
                        agent.llm_instance = new_llm_instance
                        session_data.llm_provider = provider
                        
                        # Reset token budget for this session
                        if hasattr(agent, 'token_tracker') and agent.token_tracker:
                            agent.token_tracker.reset_current_conversation_budget()
                        
                        print(f"✅ Updated session {session_id} to use {provider}/{model}")
                        return True
            return False
        except Exception as e:
            print(f"Error updating session LLM provider: {e}")
            return False
    
    def get_session_agent(self, session_id: str) -> CmwAgent:
        """Get the agent for a specific session - for UI modules to use"""
        session_data = self.get_session_data(session_id)
        return session_data.agent
    
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
        self.agent = CmwAgent(session_id=session_id)  # Pass session ID to agent
        self.status = get_translation_key("progress_ready", language)
        self.llm_provider = "openrouter"  # Default provider
        self.created_at = time.time()
        self.last_activity = time.time()
        
        # Initialize the agent with a default LLM instance for this session
        self._initialize_session_agent()
    
    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity = time.time()
    
    def _initialize_session_agent(self) -> None:
        """Initialize the session agent with a unique LLM instance"""
        try:
            if hasattr(self.agent, 'llm_manager') and self.agent.llm_manager:
                # Create a unique LLM instance for this session
                llm_instance = self.agent.llm_manager.create_new_llm_instance(self.llm_provider)
                if llm_instance:
                    self.agent.llm_instance = llm_instance
                    print(f"✅ Initialized session {self.session_id} with {self.llm_provider}")
                else:
                    print(f"⚠️ Failed to initialize LLM for session {self.session_id}")
        except Exception as e:
            print(f"❌ Error initializing session agent: {e}")
