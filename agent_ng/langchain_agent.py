"""
LangChain Native Agent
=====================

A modern agent implementation using pure LangChain patterns for multi-turn
conversations with tool calls, memory management, and streaming.

Key Features:
- Pure LangChain conversation chains
- Native memory management
- Proper tool calling support
- Streaming responses
- Multi-turn conversation support
- LangChain Expression Language (LCEL)

Based on LangChain's official documentation and best practices.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from dataclasses import dataclass
from pathlib import Path

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.tools import BaseTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler, StreamingStdOutCallbackHandler

# Simple memory implementation to avoid import issues
class ConversationBufferMemory:
    """Simple conversation buffer memory implementation"""
    def __init__(self, memory_key="chat_history", return_messages=True):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.chat_memory = []
    
    def load_memory_variables(self, inputs):
        return {self.memory_key: self.chat_memory}
    
    def save_context(self, inputs, outputs):
        # Add user input and AI output to memory
        if "input" in inputs:
            self.chat_memory.append(HumanMessage(content=inputs["input"]))
        if "output" in outputs:
            self.chat_memory.append(AIMessage(content=outputs["output"]))
    
    def clear(self):
        self.chat_memory.clear()

# Local imports
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .llm_manager import get_llm_manager, LLMInstance
    from .langchain_memory import get_memory_manager, create_conversation_chain
    from .error_handler import get_error_handler
    from .streaming_manager import get_streaming_manager
    from .message_processor import get_message_processor
    from .response_processor import get_response_processor
    from .stats_manager import get_stats_manager
    from .trace_manager import get_trace_manager
    from .utils import ensure_valid_answer
except ImportError:
    try:
        from agent_ng.llm_manager import get_llm_manager, LLMInstance
        from agent_ng.langchain_memory import get_memory_manager, create_conversation_chain
        from agent_ng.error_handler import get_error_handler
        from agent_ng.streaming_manager import get_streaming_manager
        from agent_ng.message_processor import get_message_processor
        from agent_ng.response_processor import get_response_processor
        from agent_ng.stats_manager import get_stats_manager
        from agent_ng.trace_manager import get_trace_manager
        from agent_ng.utils import ensure_valid_answer
    except ImportError:
        get_llm_manager = lambda: None
        LLMInstance = None
        get_memory_manager = lambda: None
        create_conversation_chain = lambda *args: None
        get_error_handler = lambda: None
        get_streaming_manager = lambda: None
        get_message_processor = lambda: None
        get_response_processor = lambda: None
        get_stats_manager = lambda: None
        get_trace_manager = lambda: None
        ensure_valid_answer = lambda x: str(x) if x is not None else "No answer provided"


@dataclass
class ChatMessage:
    """Structured chat message for Gradio compatibility"""
    role: str  # "user", "assistant", "system"
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentResponse:
    """Structured agent response"""
    answer: str
    tool_calls: List[Dict[str, Any]]
    conversation_id: str
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming responses"""
    
    def __init__(self, event_callback=None):
        self.event_callback = event_callback
        self.current_tool = None
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts"""
        if self.event_callback:
            self.event_callback({
                "type": "llm_start",
                "content": "🤖 **LLM is thinking...**",
                "metadata": {"llm_type": serialized.get("name", "unknown")}
            })
    
    def on_llm_stream(self, chunk, **kwargs):
        """Called when LLM streams content"""
        if hasattr(chunk, 'content') and chunk.content and self.event_callback:
            self.event_callback({
                "type": "content",
                "content": chunk.content,
                "metadata": {"chunk_type": "llm_response"}
            })
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Called when a tool starts"""
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
        """Called when a tool ends"""
        if self.event_callback:
            self.event_callback({
                "type": "tool_end",
                "content": f"✅ **Tool completed: {self.current_tool}**",
                "metadata": {"tool_name": self.current_tool}
            })
        self.current_tool = None
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM ends"""
        if self.event_callback:
            self.event_callback({
                "type": "llm_end",
                "content": "✅ **LLM processing completed**",
                "metadata": {}
            })


class LangChainAgent:
    """
    Modern agent using pure LangChain patterns with full modular architecture.
    
    This agent implements multi-turn conversations with tool calls using
    LangChain's native memory management and conversation chains, while
    maintaining all the modular components from NextGenAgent.
    """
    
    def __init__(self, system_prompt: str = None):
        """
        Initialize the LangChain agent with full modular architecture.
        
        Args:
            system_prompt: System prompt for the agent
        """
        # Initialize all modular components
        self.llm_manager = get_llm_manager()
        self.memory_manager = get_memory_manager()
        self.error_handler = get_error_handler()
        self.streaming_manager = get_streaming_manager()
        self.message_processor = get_message_processor()
        self.response_processor = get_response_processor()
        self.stats_manager = get_stats_manager()
        self.trace_manager = get_trace_manager()
        
        # Load system prompt
        self.system_prompt = system_prompt or self._load_system_prompt()
        
        # Initialize LLM and tools
        self.llm_instance = None
        self.tools = []
        self.conversation_chains = {}
        
        # Agent state
        self.is_initialized = False
        self.conversation_history = []
        self.active_streams = {}
        
        # Initialize in background
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize the agent asynchronously"""
        try:
            # Get LLM instance
            self.llm_instance = self.llm_manager.get_agent_llm()
            if not self.llm_instance:
                raise Exception("No LLM provider available. Check AGENT_PROVIDER environment variable.")
            
            # Initialize tools
            self.tools = self._initialize_tools()
            
            self.is_initialized = True
            print(f"✅ LangChain Agent initialized with {self.llm_instance.provider} ({self.llm_instance.model_name}) and {len(self.tools)} tools")
            
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            self.is_initialized = False
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from file"""
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.json")
            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("system_prompt", "You are a helpful AI assistant.")
            return "You are a helpful AI assistant."
        except Exception as e:
            print(f"Warning: Could not load system prompt: {e}")
            return "You are a helpful AI assistant."
    
    def _initialize_tools(self) -> List[BaseTool]:
        """Initialize available tools"""
        try:
            import tools.tools as tools_module
            tool_list = []
            
            for name, obj in tools_module.__dict__.items():
                if (callable(obj) and 
                    not name.startswith("_") and 
                    not isinstance(obj, type) and
                    hasattr(obj, '__module__') and
                    (obj.__module__ == 'tools.tools' or obj.__module__ == 'langchain_core.tools.structured') and
                    name not in ["CmwAgent", "CodeInterpreter", "submit_answer", "submit_intermediate_step"]):
                    
                    if hasattr(obj, 'name') and hasattr(obj, 'description'):
                        tool_list.append(obj)
                    elif callable(obj) and not name.startswith("_"):
                        tool_list.append(obj)
            
            return tool_list
        except ImportError:
            print("Warning: Could not import tools module")
            return []
    
    def _get_conversation_chain(self, conversation_id: str = "default"):
        """Get or create conversation chain for a conversation"""
        if conversation_id not in self.conversation_chains:
            self.conversation_chains[conversation_id] = create_conversation_chain(
                self.llm_instance, self.tools, self.system_prompt
            )
        return self.conversation_chains[conversation_id]
    
    def process_message(self, message: str, conversation_id: str = "default") -> AgentResponse:
        """
        Process a message using LangChain patterns.
        
        Args:
            message: User message
            conversation_id: Conversation identifier
            
        Returns:
            AgentResponse with answer and metadata
        """
        if not self.llm_instance:
            return AgentResponse(
                answer="Agent not initialized",
                tool_calls=[],
                conversation_id=conversation_id,
                success=False,
                error="not_initialized"
            )
        
        try:
            # Get conversation chain
            chain = self._get_conversation_chain(conversation_id)
            
            # Process with tools
            result = chain.process_with_tools(message, conversation_id)
            
            return AgentResponse(
                answer=result["response"],
                tool_calls=result["tool_calls"],
                conversation_id=conversation_id,
                success=result["success"],
                error=result.get("error")
            )
            
        except Exception as e:
            return AgentResponse(
                answer=f"Error processing message: {str(e)}",
                tool_calls=[],
                conversation_id=conversation_id,
                success=False,
                error=str(e)
            )
    
    async def stream_message(self, message: str, conversation_id: str = "default") -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a message response using simple streaming implementation.
        
        Args:
            message: User message
            conversation_id: Conversation identifier
            
        Yields:
            Dict with event type, content, and metadata
        """
        if not self.llm_instance:
            yield {
                "type": "error",
                "content": "Agent not initialized",
                "metadata": {"error": "not_initialized"}
            }
            return
        
        try:
            # Use simple streaming with improved tool filtering
            from .simple_streaming import get_simple_streaming_manager
            
            # Get conversation chain
            chain = self._get_conversation_chain(conversation_id)
            
            # Process the message first to get the response
            result = chain.process_with_tools(message, conversation_id)
            
            # Get simple streaming manager
            streaming_manager = get_simple_streaming_manager()
            
            # Stream thinking process
            async for event in streaming_manager.stream_thinking(message):
                yield {
                    "type": event.event_type,
                    "content": event.content,
                    "metadata": event.metadata or {}
                }
            
            # Stream response with tool calls (ensure response is a string)
            response_text = ensure_valid_answer(result["response"])
            async for event in streaming_manager.stream_response_with_tools(
                response_text, 
                result.get("tool_calls", [])
            ):
                yield {
                    "type": event.event_type,
                    "content": event.content,
                    "metadata": event.metadata or {}
                }
            
        except Exception as e:
            # Stream error
            from .simple_streaming import get_simple_streaming_manager
            streaming_manager = get_simple_streaming_manager()
            
            async for event in streaming_manager.stream_error(str(e)):
                yield {
                    "type": event.event_type,
                    "content": event.content,
                    "metadata": event.metadata or {}
                }
    
    def get_conversation_history(self, conversation_id: str = "default") -> List[BaseMessage]:
        """Get conversation history"""
        chain = self._get_conversation_chain(conversation_id)
        return chain.get_conversation_history(conversation_id)
    
    async def stream_chat(self, message: str, history: List[ChatMessage] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat response with real-time updates using LangChain native streaming.
        
        Args:
            message: User message
            history: Chat history as ChatMessage objects
            
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
        
        # Add user message to history
        self.conversation_history.append(ChatMessage(role="user", content=message))
        
        try:
            # Stream the response using the updated stream_message method
            async for event in self.stream_message(message, "default"):
                yield event
                
        except Exception as e:
            yield {
                "type": "error",
                "content": f"Error processing message: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    def clear_conversation(self, conversation_id: str = "default") -> None:
        """Clear conversation history"""
        chain = self._get_conversation_chain(conversation_id)
        chain.clear_conversation(conversation_id)
    
    def is_ready(self) -> bool:
        """Check if the agent is ready to process requests"""
        return self.is_initialized and self.llm_instance is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "is_initialized": self.is_initialized,
            "is_ready": self.is_ready(),
            "current_llm": self.llm_instance.model_name if self.llm_instance else None,
            "current_provider": self.llm_instance.provider.value if self.llm_instance else None,
            "tools_count": len(self.tools),
            "conversation_length": len(self.conversation_history)
        }
    
    def get_llm_info(self) -> Dict[str, Any]:
        """Get information about the current LLM"""
        if not self.llm_instance:
            return {"error": "No LLM instance available"}
        
        return {
            "provider": self.llm_instance.provider.value,
            "model_name": self.llm_instance.model_name,
            "config": self.llm_instance.config,
            "is_healthy": self.llm_instance.is_healthy,
            "last_used": self.llm_instance.last_used,
            "error_count": self.llm_instance.error_count
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        return {
            "agent_status": self.get_status(),
            "llm_info": self.get_llm_info(),
            "core_agent_stats": {
                "tools_count": len(self.tools),
                "conversation_chains": len(self.conversation_chains)
            },
            "llm_manager_stats": self.llm_manager.get_stats() if self.llm_manager else {},
            "stats_manager_stats": self.stats_manager.get_stats() if self.stats_manager else {},
            "memory_manager_stats": {
                "total_memories": len(self.memory_manager.memories) if self.memory_manager else 0
            },
            "conversation_stats": {
                "message_count": len(self.conversation_history),
                "user_messages": len([m for m in self.conversation_history if hasattr(m, 'role') and m.role == "user"]),
                "assistant_messages": len([m for m in self.conversation_history if hasattr(m, 'role') and m.role == "assistant"])
            }
        }


# Global agent instance
_agent_instance = None
_agent_lock = asyncio.Lock()


async def get_agent_ng() -> LangChainAgent:
    """Get the global LangChain agent instance (compatible with NextGenAgent)"""
    global _agent_instance
    if _agent_instance is None:
        async with _agent_lock:
            if _agent_instance is None:
                _agent_instance = LangChainAgent()
                # Wait for initialization
                while not _agent_instance.is_ready():
                    await asyncio.sleep(0.1)
    return _agent_instance


def reset_agent_ng():
    """Reset the global agent (compatible with NextGenAgent)"""
    global _agent_instance
    _agent_instance = None


# Backward compatibility aliases
get_langchain_agent = get_agent_ng
reset_langchain_agent = reset_agent_ng
