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
import os
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from dataclasses import dataclass
from .token_counter import TokenCount
from pathlib import Path

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.tools import BaseTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler, StreamingStdOutCallbackHandler


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
    # from .streaming_manager import get_streaming_manager  # Moved to .unused
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


class CmwAgent:
    """
    Modern agent using pure LangChain patterns with full modular architecture.
    
    This agent implements multi-turn conversations with tool calls using
    LangChain's native memory management and conversation chains, while
    maintaining all the modular components from NextGenAgent.
    """
    
    def __init__(self, system_prompt: str = None, session_id: str = "default"):
        """
        Initialize the LangChain agent with full modular architecture.
        
        Args:
            system_prompt: System prompt for the agent
            session_id: Unique session ID for conversation isolation
        """
        # Store session ID for conversation isolation
        self.session_id = session_id
        
        # Initialize all modular components
        self.llm_manager = get_llm_manager()
        self.memory_manager = get_memory_manager()
        self.error_handler = get_error_handler()
        self.message_processor = get_message_processor()
        self.response_processor = get_response_processor()
        self.stats_manager = get_stats_manager()
        self.trace_manager = get_trace_manager(self.session_id)
        
        # Initialize token tracker
        from .token_counter import get_token_tracker
        self.token_tracker = get_token_tracker(self.session_id)
        
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
        
        # File registry system (lean and secure) - session isolated
        self.file_registry = {}  # Maps (session_id, original_filename) -> full_file_path
        self.session_cache_path = None  # Gradio cache path for this session
        self.current_files = []  # Current conversation turn files
        
        # Initialize in background - handle case when no event loop is running
        try:
            asyncio.create_task(self._initialize_async())
        except RuntimeError:
            # No event loop running, initialize synchronously
            import threading
            def run_async_init():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._initialize_async())
                finally:
                    loop.close()
            
            thread = threading.Thread(target=run_async_init, daemon=True)
            thread.start()
    
    async def _initialize_async(self):
        """Initialize the agent asynchronously"""
        try:
            # Get LLM instance
            self.llm_instance = self.llm_manager.get_agent_llm()
            if not self.llm_instance:
                raise Exception("No LLM provider available. Check AGENT_PROVIDER environment variable.")
            
            # Initialize tools using LLM manager's cached tools
            self.tools = self.llm_manager.get_tools()
            
            self.is_initialized = True
            print(f"✅ LangChain Agent initialized with {self.llm_instance.provider} ({self.llm_instance.model_name}) and {len(self.tools)} tools")
            
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            self.is_initialized = False
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from file - FAILS if not found"""
        prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.json")
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Build system prompt from the JSON structure
            system_prompt = f"""You are a {data.get('role', 'helpful AI assistant')}.

{data.get('platform_description', '')}

## Tool Usage Policy:
{data.get('cmw_tools', {}).get('purpose', '')}

**CRITICAL: For platform operations ALWAYS use tools first - never provide information without using the appropriate tool first.**

**For math calculations, use the available math tools (add, multiply, subtract, divide, etc.) to show your work step by step.**

## Answer Format:
{data.get('answer_format', {}).get('template', '')}

## Answer Rules:
{chr(10).join(f"- {rule}" for rule in data.get('answer_format', {}).get('answer_rules', []))}

## Tool Usage Guidelines:
{chr(10).join(f"- {rule}" for rule in data.get('cmw_tools', {}).get('tool_usage_policy', []))}

## Example Tasks:
{chr(10).join(f"- {task.get('Task', '')}: {task.get('Intent', '')}" for task in data.get('example_tasks_solutions', []))}

Always use the appropriate tools to answer questions and show your work step by step."""
            return system_prompt
    
    
    def _get_conversation_chain(self, conversation_id: str = "default"):
        """Get or create conversation chain for a conversation"""
        if conversation_id not in self.conversation_chains:
            self.conversation_chains[conversation_id] = create_conversation_chain(
                self.llm_instance, self.tools, self.system_prompt, self
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
        Stream a message response using proper LangChain streaming.
        
        This method uses LangChain's native streaming methods with correctly
        configured LLM instances for real-time token-by-token streaming.
        
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
            # Use native LangChain streaming
            from .native_langchain_streaming import get_native_streaming
            
            # Get native streaming manager
            streaming_manager = get_native_streaming()
            
            # Stream agent response using native LangChain streaming
            async for event in streaming_manager.stream_agent_response(
                self, message, conversation_id
            ):
                # Convert to the expected format
                yield {
                    "type": event.event_type,
                    "content": event.content,
                    "metadata": event.metadata or {}
                }
            
        except Exception as e:
            # Stream error
            yield {
                "type": "error",
                "content": f"❌ **Error: {str(e)}**",
                "metadata": {"error": str(e)}
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
        
        # Track conversation start
        start_time = time.time()
        llm_type = "unknown"
        if self.llm_instance:
            llm_type = self.llm_instance.provider.value
        
        try:
            # Stream the response using the updated stream_message method
            async for event in self.stream_message(message, "default"):
                yield event
                
            # Track successful LLM usage
            response_time = time.time() - start_time
            print(f"📊 Tracking stats - LLM: {llm_type}, Session: {self.session_id}, Response time: {response_time:.2f}s")
            if self.stats_manager:
                self.stats_manager.track_llm_usage(
                    llm_type=llm_type,
                    event_type="success",
                    response_time=response_time,
                    session_id=self.session_id
                )
                
                # Track conversation
                self.stats_manager.track_conversation(
                    conversation_id=self.session_id,
                    question=message,
                    answer="[Streamed response]",
                    llm_used=llm_type,
                    tool_calls=0,  # TODO: Track actual tool calls
                    duration=response_time,
                    session_id=self.session_id
                )
                print(f"📊 Stats tracked successfully for session {self.session_id}")
            else:
                print(f"❌ No stats manager available for session {self.session_id}")
                
        except Exception as e:
            # Track failed LLM usage
            if self.stats_manager:
                self.stats_manager.track_llm_usage(
                    llm_type=llm_type,
                    event_type="failure",
                    response_time=time.time() - start_time,
                    session_id=self.session_id
                )
                self.stats_manager.track_error(
                    error_type=str(e),
                    llm_type=llm_type
                )
            
            yield {
                "type": "error",
                "content": f"Error processing message: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    def clear_conversation(self, conversation_id: str = None) -> None:
        """Clear conversation history and file data"""
        # Use session ID if no conversation ID provided
        if conversation_id is None:
            conversation_id = self.session_id
            
        chain = self._get_conversation_chain(conversation_id)
        chain.clear_conversation(conversation_id)
        # Clear file registry when clearing conversation
        self.file_registry = {}
        self.current_files = []
    
    def is_ready(self) -> bool:
        """Check if the agent is ready to process requests"""
        return self.is_initialized and self.llm_instance is not None
    
    def get_file_path(self, original_filename: str) -> str:
        """
        Get the full file path for a file by its original filename.
        
        Args:
            original_filename (str): Original filename from user upload
            
        Returns:
            str: Full path to the file, or None if not found
        """
        # Check if we have this file in our session-isolated registry
        registry_key = (self.session_id, original_filename)
        
        if registry_key in self.file_registry:
            full_path = self.file_registry[registry_key]
            if os.path.exists(full_path):
                return full_path
        
        return None
    
    def register_file(self, original_filename: str, file_path: str) -> None:
        """
        Register a file in the session-isolated file registry.
        Creates a unique filename and moves the file to Gradio cache.
        
        Args:
            original_filename (str): Original filename from user upload
            file_path (str): Full path to the original file
        """
        import shutil
        from tools.file_utils import FileUtils
        
        # Generate unique filename with timestamp and hash
        unique_filename = FileUtils.generate_unique_filename(original_filename, self.session_id)
        
        # Use Gradio cache directory (files will be accessible via Gradio's file access)
        if not self.session_cache_path:
            self.session_cache_path = FileUtils.get_gradio_cache_path()
        
        # Create unique file path in Gradio cache
        unique_file_path = os.path.join(self.session_cache_path, unique_filename)
        
        # Move and rename file to unique location in Gradio cache
        try:
            shutil.move(file_path, unique_file_path)
            
            # Register the unique file path in session-isolated registry
            registry_key = (self.session_id, original_filename)
            self.file_registry[registry_key] = unique_file_path
            print(f"📁 Registered file: {original_filename} -> {unique_file_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to move file {original_filename} to Gradio cache: {e}")
            # Fallback: register original path
            registry_key = (self.session_id, original_filename)
            self.file_registry[registry_key] = file_path
    
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
            "stats_manager_stats": self.stats_manager.get_stats(self.session_id) if self.stats_manager else {},
            "memory_manager_stats": {
                "total_memories": len(self.memory_manager.memories) if self.memory_manager else 0
            },
            "conversation_stats": self._get_conversation_stats()
        }
    
    def get_conversation_stats_debug(self) -> Dict[str, int]:
        """Get conversation statistics with debug output enabled"""
        return self._get_conversation_stats(debug=True)
    
    def log_conversation_event(self, event_type: str, details: str = ""):
        """Log conversation-related events with debug information"""
        print(f"🔍 EVENT: {event_type} - {details}")
        if event_type in ["new_message", "conversation_start", "conversation_end"]:
            # Show debug stats for important events
            self.get_conversation_stats_debug()
    
    def _get_conversation_stats(self, debug: bool = False) -> Dict[str, int]:
        """Get conversation statistics from memory manager
        
        Args:
            debug: If True, show detailed debug messages. If False, only log on changes.
        """
        try:
            if debug:
                print(f"🔍 DEBUG: Getting conversation stats from memory manager")
            
            if self.memory_manager:
                if debug:
                    print(f"🔍 DEBUG: Memory manager type: {type(self.memory_manager)}")
                    print(f"🔍 DEBUG: Memory manager has {len(self.memory_manager.memories)} conversations")
                    print(f"🔍 DEBUG: Memory manager memories: {self.memory_manager.memories}")
                
                # Get all conversations and count messages for this session
                total_messages = 0
                user_messages = 0
                assistant_messages = 0
                
                # Only count messages for this session's conversation
                session_conversation = self.memory_manager.memories.get(self.session_id)
                if session_conversation:
                    conversations_to_process = {self.session_id: session_conversation}
                else:
                    conversations_to_process = {}
                
                for conversation_id, conversation in conversations_to_process.items():
                    if debug:
                        print(f"🔍 DEBUG: Conversation {conversation_id} type: {type(conversation)}")
                    
                    # Handle different memory types
                    if hasattr(conversation, 'chat_memory') and hasattr(conversation.chat_memory, 'chat_memory'):
                        # ToolAwareMemory with chat_memory.chat_memory
                        messages = conversation.chat_memory.chat_memory
                        if debug:
                            print(f"🔍 DEBUG: Conversation {conversation_id} has {len(messages)} messages (from chat_memory.chat_memory)")
                        for i, message in enumerate(messages):
                            total_messages += 1
                            if debug:
                                print(f"🔍 DEBUG: Message {i}: type={type(message)}, role={getattr(message, 'role', 'No role')}, content={getattr(message, 'content', 'No content')[:50]}...")
                            if hasattr(message, 'role'):
                                if message.role == "user":
                                    user_messages += 1
                                elif message.role == "assistant":
                                    assistant_messages += 1
                            elif hasattr(message, 'type'):
                                # Handle different message types
                                if message.type == "human":
                                    user_messages += 1
                                elif message.type == "ai":
                                    assistant_messages += 1
                    elif hasattr(conversation, '__iter__'):
                        # Direct list of messages
                        messages = list(conversation)
                        if debug:
                            print(f"🔍 DEBUG: Conversation {conversation_id} has {len(messages)} messages")
                        for i, message in enumerate(messages):
                            total_messages += 1
                            if debug:
                                print(f"🔍 DEBUG: Message {i}: type={type(message)}, role={getattr(message, 'role', 'No role')}, content={getattr(message, 'content', 'No content')[:50]}...")
                            if hasattr(message, 'role'):
                                if message.role == "user":
                                    user_messages += 1
                                elif message.role == "assistant":
                                    assistant_messages += 1
                            elif hasattr(message, 'type'):
                                # Handle different message types
                                if message.type == "human":
                                    user_messages += 1
                                elif message.type == "ai":
                                    assistant_messages += 1
                    else:
                        if debug:
                            print(f"🔍 DEBUG: Unknown conversation type: {type(conversation)}")
                
                if debug:
                    print(f"🔍 DEBUG: Total stats: {total_messages} total, {user_messages} user, {assistant_messages} assistant")
                return {
                    "message_count": total_messages,
                    "user_messages": user_messages,
                    "assistant_messages": assistant_messages
                }
            else:
                if debug:
                    print("🔍 DEBUG: No memory manager available")
                return {
                    "message_count": 0,
                    "user_messages": 0,
                    "assistant_messages": 0
                }
        except Exception as e:
            print(f"🔍 DEBUG: Error getting conversation stats: {e}")
            import traceback
            traceback.print_exc()
            return {
                "message_count": 0,
                "user_messages": 0,
                "assistant_messages": 0
            }
    
    def get_token_counts(self, messages: List[Any]) -> Dict[str, Any]:
        """Get token counts for display"""
        if hasattr(self, 'langchain_wrapper') and self.langchain_wrapper:
            return self.langchain_wrapper.get_token_counts(messages)
        return {"prompt_tokens": None, "cumulative_stats": {}}
    
    def get_token_display_info(self) -> Dict[str, Any]:
        """Get comprehensive token display information"""
        if hasattr(self, 'token_tracker'):
            return self.token_tracker.get_token_display_info()
        return {"prompt_tokens": None, "api_tokens": None, "cumulative_stats": {}}
    
    def count_prompt_tokens_for_chat(self, history: List[Dict[str, str]], current_message: str) -> Optional[TokenCount]:
        """Count prompt tokens for chat history and current message"""
        if hasattr(self, 'token_tracker'):
            from .token_counter import convert_chat_history_to_messages
            messages = convert_chat_history_to_messages(history, current_message)
            return self.token_tracker.count_prompt_tokens(messages)
        return None
    
    def get_last_api_tokens(self) -> Optional[TokenCount]:
        """Get the last API token count"""
        if hasattr(self, 'token_tracker'):
            return self.token_tracker.get_last_api_tokens()
        return None
    
    def get_token_budget_info(self) -> Dict[str, Any]:
        """Get token budget information for the current LLM context window"""
        if not hasattr(self, 'token_tracker') or not self.token_tracker:
            return {
                "used_tokens": 0,
                "context_window": 0,
                "percentage": 0.0,
                "remaining_tokens": 0,
                "status": "unknown"
            }
        
        # Get context window from LLM manager
        context_window = 0
        if hasattr(self, 'llm_manager') and self.llm_manager:
            context_window = self.llm_manager.get_current_llm_context_window()
        
        return self.token_tracker.get_token_budget_info(context_window)


# Global agent instance
_agent_instance = None
_agent_lock = asyncio.Lock()


async def get_agent_ng() -> CmwAgent:
    """Get the global LangChain agent instance (compatible with NextGenAgent)"""
    global _agent_instance
    if _agent_instance is None:
        async with _agent_lock:
            if _agent_instance is None:
                _agent_instance = CmwAgent()
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
