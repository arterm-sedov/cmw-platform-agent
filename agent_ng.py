"""
Next-Generation Agent Module
===========================

A clean, modern agent implementation that orchestrates all modular components
for LLM interactions, tool calling, and streaming responses.

Key Features:
- Clean separation of concerns
- Modern async/streaming patterns
- Modular architecture integration
- Real-time streaming with metadata
- Error handling and fallback logic
- Conversation management

Usage:
    agent = NextGenAgent()
    async for chunk in agent.stream_chat("Hello", history):
        print(chunk)
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from dataclasses import dataclass
from pathlib import Path

# LangChain imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler

# Local modular imports
from core_agent import CoreAgent, get_agent
from llm_manager import get_llm_manager, LLMInstance
from error_handler import get_error_handler
from streaming_manager import get_streaming_manager, StreamingEvent
from message_processor import get_message_processor, MessageContext
from response_processor import get_response_processor, ProcessedResponse
from stats_manager import get_stats_manager
from trace_manager import get_trace_manager
from utils import ensure_valid_answer


@dataclass
class ChatMessage:
    """Structured chat message for Gradio compatibility"""
    role: str  # "user", "assistant", "system"
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentConfig:
    """Configuration for the next-gen agent"""
    enable_vector_similarity: bool = True
    max_conversation_history: int = 50
    max_tool_calls: int = 10
    similarity_threshold: float = 0.95
    streaming_chunk_size: int = 100
    enable_tool_calling: bool = True
    enable_streaming: bool = True


class StreamingCallbackHandler(BaseCallbackHandler):
    """Modern streaming callback handler for real-time updates"""
    
    def __init__(self, event_callback=None):
        self.event_callback = event_callback
        self.current_tool = None
        self.event_count = 0
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts processing"""
        if self.event_callback:
            self.event_callback({
                "type": "llm_start",
                "content": "🤖 **LLM is thinking...**",
                "metadata": {"llm_type": serialized.get("name", "unknown")}
            })
    
    def on_llm_stream(self, chunk, **kwargs):
        """Called for each streaming chunk"""
        if hasattr(chunk, 'content') and chunk.content and self.event_callback:
            self.event_callback({
                "type": "llm_chunk",
                "content": chunk.content,
                "metadata": {"chunk_type": "llm_response"}
            })
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Called when a tool starts executing"""
        self.current_tool = serialized.get("name", "unknown_tool")
        if self.event_callback:
            self.event_callback({
                "type": "tool_start",
                "content": f"🔧 **Using tool: {self.current_tool}**",
                "metadata": {
                    "tool_name": self.current_tool,
                    "tool_args": input_str
                }
            })
    
    def on_tool_end(self, output, **kwargs):
        """Called when a tool finishes executing"""
        if self.event_callback:
            self.event_callback({
                "type": "tool_end",
                "content": f"✅ **Tool completed: {self.current_tool}**",
                "metadata": {"tool_name": self.current_tool}
            })
        self.current_tool = None
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM finishes processing"""
        if self.event_callback:
            self.event_callback({
                "type": "llm_end",
                "content": "✅ **LLM processing completed**",
                "metadata": {}
            })


class NextGenAgent:
    """
    Next-generation agent that orchestrates all modular components.
    
    This agent provides a clean, modern interface for LLM interactions
    with streaming, tool calling, and conversation management.
    """
    
    def __init__(self, config: AgentConfig = None):
        """
        Initialize the next-generation agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config or AgentConfig()
        
        # Initialize modular components
        self.core_agent = get_agent()
        self.llm_manager = get_llm_manager()
        self.error_handler = get_error_handler()
        self.streaming_manager = get_streaming_manager()
        self.message_processor = get_message_processor()
        self.response_processor = get_response_processor()
        self.stats_manager = get_stats_manager()
        self.trace_manager = get_trace_manager()
        
        # Agent state
        self.is_initialized = False
        self.current_llm_instance = None
        self.conversation_history = []
        self.active_streams = {}
        
        # Initialize in background
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize the agent asynchronously"""
        try:
            # Get single LLM instance from AGENT_PROVIDER
            llm_instance = self.llm_manager.get_agent_llm()
            
            if not llm_instance:
                raise Exception("No LLM provider available. Check AGENT_PROVIDER environment variable.")
            
            # Use the single LLM instance
            self.current_llm_instance = llm_instance
            self.is_initialized = True
            print(f"✅ NextGen Agent initialized with {llm_instance.provider} ({llm_instance.model_name})")
                
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            self.is_initialized = False
    
    def is_ready(self) -> bool:
        """Check if the agent is ready to process requests"""
        return self.is_initialized and self.current_llm_instance is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "is_initialized": self.is_initialized,
            "is_ready": self.is_ready(),
            "current_llm": self.current_llm_instance.model_name if self.current_llm_instance else None,
            "current_provider": self.current_llm_instance.provider.value if self.current_llm_instance else None,
            "tools_count": len(self.core_agent.tools) if self.core_agent else 0,
            "conversation_length": len(self.conversation_history)
        }
    
    async def stream_chat(self, message: str, history: List[ChatMessage] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat response with real-time updates.
        
        Args:
            message: User message
            history: Chat history
            
        Yields:
            Dict with event type, content, and metadata
        """
        if not self.is_ready():
            yield {
                "type": "error",
                "content": "Agent not ready. Please wait for initialization to complete.",
                "metadata": {"error": "not_ready"}
            }
            return
        
        # Convert history to internal format
        internal_history = []
        if history:
            for msg in history:
                internal_history.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add user message to history
        self.conversation_history.append(ChatMessage(role="user", content=message))
        
        # Create streaming callback
        events = []
        
        def event_callback(event):
            events.append(event)
        
        callback_handler = StreamingCallbackHandler(event_callback)
        
        try:
            # Process with core agent streaming
            async for event in self.core_agent.process_question_stream(
                question=message,
                chat_history=internal_history,
                conversation_id="default"
            ):
                # Convert to our event format
                if event.get("event_type") == "answer":
                    yield {
                        "type": "content",
                        "content": event["content"],
                        "metadata": {"source": "llm"}
                    }
                elif event.get("event_type") == "llm_start":
                    yield {
                        "type": "thinking",
                        "content": "🤖 **LLM is thinking...**",
                        "metadata": {"llm_type": "unknown"}
                    }
                elif event.get("event_type") == "tool_start":
                    yield {
                        "type": "tool_use",
                        "content": f"🔧 **Using tool: {event.get('content', 'unknown')}**",
                        "metadata": {"tool_name": event.get('content', 'unknown')}
                    }
                elif event.get("event_type") == "success":
                    yield {
                        "type": "success",
                        "content": f"✅ **{event['content']}**",
                        "metadata": {}
                    }
                elif event.get("event_type") == "error":
                    yield {
                        "type": "error",
                        "content": f"❌ **{event['content']}**",
                        "metadata": {"error": True}
                    }
                else:
                    yield {
                        "type": "info",
                        "content": event.get("content", ""),
                        "metadata": {}
                    }
        
        except Exception as e:
            yield {
                "type": "error",
                "content": f"Error processing message: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    def get_conversation_history(self) -> List[ChatMessage]:
        """Get the current conversation history"""
        return self.conversation_history.copy()
    
    def clear_conversation(self):
        """Clear the conversation history"""
        self.conversation_history.clear()
    
    def get_llm_info(self) -> Dict[str, Any]:
        """Get information about the current LLM"""
        if not self.current_llm_instance:
            return {"error": "No LLM instance available"}
        
        return {
            "provider": self.current_llm_instance.provider.value,
            "model_name": self.current_llm_instance.model_name,
            "config": self.current_llm_instance.config,
            "is_healthy": self.current_llm_instance.is_healthy,
            "last_used": self.current_llm_instance.last_used,
            "error_count": self.current_llm_instance.error_count
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        return {
            "agent_status": self.get_status(),
            "llm_info": self.get_llm_info(),
            "core_agent_stats": self.core_agent.get_agent_stats() if self.core_agent else {},
            "llm_manager_stats": self.llm_manager.get_stats(),
            "stats_manager_stats": self.stats_manager.get_stats(),
            "conversation_stats": {
                "message_count": len(self.conversation_history),
                "user_messages": len([m for m in self.conversation_history if m.role == "user"]),
                "assistant_messages": len([m for m in self.conversation_history if m.role == "assistant"])
            }
        }


# Global agent instance
_agent_instance = None
_agent_lock = asyncio.Lock()


async def get_agent_ng() -> NextGenAgent:
    """Get the global next-gen agent instance (singleton pattern)"""
    global _agent_instance
    if _agent_instance is None:
        async with _agent_lock:
            if _agent_instance is None:
                _agent_instance = NextGenAgent()
                # Wait for initialization
                while not _agent_instance.is_ready():
                    await asyncio.sleep(0.1)
    return _agent_instance


def reset_agent_ng():
    """Reset the global agent (useful for testing)"""
    global _agent_instance
    _agent_instance = None
