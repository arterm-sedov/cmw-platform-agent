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

try:
    from ..utils import get_tool_call_count
except ImportError:
    # Fallback for when running as script
    from utils import get_tool_call_count

# LangChain imports
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

# LangSmith tracing
try:
    from langsmith import traceable

    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

    def traceable(func):
        return func


from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
)
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
    from .utils import ensure_valid_answer

    # LangSmith tracing is now handled via direct imports and environment variables
    print("✅ Successfully imported all modules using relative imports")
except ImportError as e1:
    print(f"❌ Relative import failed: {e1}")
    try:
        from agent_ng.llm_manager import get_llm_manager, LLMInstance
        from agent_ng.langchain_memory import (
            get_memory_manager,
            create_conversation_chain,
        )
        from agent_ng.error_handler import get_error_handler
        from agent_ng.message_processor import get_message_processor
        from agent_ng.response_processor import get_response_processor
        from agent_ng.stats_manager import get_stats_manager
        from agent_ng.utils import ensure_valid_answer

        # LangSmith tracing is now handled via direct imports and environment variables
        print("✅ Successfully imported all modules using absolute imports")
    except ImportError as e2:
        print(f"❌ Absolute import failed: {e2}")
        print("💥 CRITICAL ERROR: Cannot import required modules!")
        print("🔧 Please check:")
        print(
            "   1. All dependencies are installed: pip install -r requirements_ng.txt"
        )
        print("   2. Python path is correct")
        print("   3. No circular import issues")
        print("   4. All required modules exist")
        raise ImportError(
            f"Failed to import required modules. Relative import: {e1}, Absolute import: {e2}"
        )


@dataclass
class ChatMessage:
    """Structured chat message for Gradio compatibility"""

    role: str  # "user", "assistant", "system"
    content: str
    metadata: Optional[Dict[str, Any]] = None

class CmwAgent:
    """
    Modern agent using pure LangChain patterns with full modular architecture.

    This agent implements multi-turn conversations with tool calls using
    LangChain's native memory management and conversation chains, while
    maintaining all the modular components from NextGenAgent.
    """

    def __init__(self, system_prompt: str = None, session_id: str = "default", language: str = "en"):
        """
        Initialize the LangChain agent with full modular architecture.

        Args:
            system_prompt: System prompt for the agent
            session_id: Unique session ID for conversation isolation
            language: Language for the agent (default: "en")
        """
        # Store session ID for conversation isolation
        self.session_id = session_id
        # Store language for internationalization
        self.language = language

        # Initialize all modular components
        self.llm_manager = get_llm_manager()

        self.memory_manager = get_memory_manager()
        self.error_handler = get_error_handler()
        self.message_processor = get_message_processor()
        self.response_processor = get_response_processor()
        self.stats_manager = get_stats_manager()

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

        # File registry system (lean and secure) - session isolated
        self.file_registry = {}  # Maps (session_id, original_filename) -> full_file_path
        self.session_cache_path = None  # Gradio cache path for this session

        # Initialize in background - handle case when no event loop is running
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # Create task in the current event loop
            loop.create_task(self._initialize_async())
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
            # Wait for initialization to complete
            thread.join(timeout=60)  # Wait up to 60 seconds for initialization

    async def _initialize_async(self):
        """Initialize the agent asynchronously"""
        try:
            # Get LLM instance
            self.llm_instance = self.llm_manager.get_agent_llm()
            if not self.llm_instance:
                raise Exception(
                    "No LLM provider available. Check AGENT_PROVIDER environment variable."
                )

            # Initialize tools using LLM manager's cached tools
            self.tools = self.llm_manager.get_tools()

            self.is_initialized = True
            print(
                f"✅ LangChain Agent initialized with {self.llm_instance.provider} ({self.llm_instance.model_name}) and {len(self.tools)} tools"
            )

        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            self.is_initialized = False

    def _load_system_prompt(self) -> str:
        """Load system prompt directly from JSON file"""
        prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.json")
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"System prompt file not found: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _get_conversation_chain(self, conversation_id: str = "default"):
        """Get or create conversation chain for a conversation"""
        if conversation_id not in self.conversation_chains:
            self.conversation_chains[conversation_id] = create_conversation_chain(
                self.llm_instance, self.tools, self.system_prompt, self
            )
        return self.conversation_chains[conversation_id]

    async def stream_message(
        self, message: str, conversation_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
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
                "metadata": {"error": "not_initialized"},
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
                # Safety check for None event
                if event is None:
                    print(
                        "🔍 DEBUG: Received None event from streaming manager, skipping..."
                    )
                    continue

                # Convert to the expected format
                yield {
                    "type": getattr(event, "event_type", "unknown"),
                    "content": getattr(event, "content", ""),
                    "metadata": getattr(event, "metadata", None) or {},
                }

        except Exception as e:
            # Stream error with icon - let native error wording speak for itself
            content = f"❌ {str(e)}"

            yield {"type": "error", "content": content, "metadata": {"error": str(e)}}

    def get_conversation_history(
        self, conversation_id: str = "default"
    ) -> List[BaseMessage]:
        """Get conversation history"""
        chain = self._get_conversation_chain(conversation_id)
        return chain.get_conversation_history(conversation_id)

    def clear_conversation(self, conversation_id: str = None) -> None:
        """Clear conversation history and file data"""
        # Use session ID if no conversation ID provided
        if conversation_id is None:
            conversation_id = self.session_id

        chain = self._get_conversation_chain(conversation_id)
        chain.clear_conversation(conversation_id)
        # Clear file registry when clearing conversation
        self.file_registry = {}

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
        unique_filename = FileUtils.generate_unique_filename(
            original_filename, self.session_id
        )

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
            "current_provider": self.llm_instance.provider.value
            if self.llm_instance
            else None,
            "tools_count": len(self.tools),
            "conversation_length": self._get_conversation_length(),
        }

    def _get_conversation_length(self) -> int:
        """Get actual conversation length from memory manager"""
        try:
            if self.memory_manager:
                # Get conversation history from memory manager
                conversation_history = self.memory_manager.get_conversation_history(self.session_id)
                return len(conversation_history) if conversation_history else 0
            return 0
        except Exception:
            # Fallback to 0 if there's any error
            return 0

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
            "error_count": self.llm_instance.error_count,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        return {
            "agent_status": self.get_status(),
            "llm_info": self.get_llm_info(),
            "core_agent_stats": {
                "tools_count": len(self.tools),
                "conversation_chains": len(self.conversation_chains),
            },
            "llm_manager_stats": self.llm_manager.get_stats()
            if self.llm_manager
            else {},
            "stats_manager_stats": self.stats_manager.get_stats(self.session_id)
            if self.stats_manager
            else {},
            "memory_manager_stats": {
                "total_memories": len(self.memory_manager.memories)
                if self.memory_manager
                else 0
            },
            "conversation_stats": self._get_conversation_stats(),
        }

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
                    print(
                        f"🔍 DEBUG: Memory manager has {len(self.memory_manager.memories)} conversations"
                    )
                    print(
                        f"🔍 DEBUG: Memory manager memories: {self.memory_manager.memories}"
                    )

                # Get all conversations and count messages for this session
                total_messages = 0
                user_messages = 0
                assistant_messages = 0
                system_prompt_count = 0

                # Only count messages for this session's conversation
                session_conversation = self.memory_manager.memories.get(self.session_id)
                if session_conversation:
                    conversations_to_process = {self.session_id: session_conversation}
                else:
                    conversations_to_process = {}

                for conversation_id, conversation in conversations_to_process.items():
                    if debug:
                        print(
                            f"🔍 DEBUG: Conversation {conversation_id} type: {type(conversation)}"
                        )

                    # Handle different memory types
                    if hasattr(conversation, "chat_memory") and hasattr(
                        conversation.chat_memory, "chat_memory"
                    ):
                        # ToolAwareMemory with chat_memory.chat_memory
                        messages = conversation.chat_memory.chat_memory
                        if debug:
                            print(
                                f"🔍 DEBUG: Conversation {conversation_id} has {len(messages)} messages (from chat_memory.chat_memory)"
                            )
                        for i, message in enumerate(messages):
                            total_messages += 1
                            if debug:
                                print(
                                    f"🔍 DEBUG: Message {i}: type={type(message)}, role={getattr(message, 'role', 'No role')}, content={getattr(message, 'content', 'No content')[:50]}..."
                                )
                            if hasattr(message, "role"):
                                if message.role == "user":
                                    user_messages += 1
                                elif message.role == "assistant":
                                    assistant_messages += 1
                                elif message.role == "system":
                                    system_prompt_count += 1
                            elif hasattr(message, "type"):
                                # Handle different message types
                                if message.type == "human":
                                    user_messages += 1
                                elif message.type == "ai":
                                    assistant_messages += 1
                                elif message.type == "system":
                                    system_prompt_count += 1
                            # Also check for SystemMessage instances
                            elif hasattr(message, "__class__") and "System" in message.__class__.__name__:
                                system_prompt_count += 1
                    elif hasattr(conversation, "__iter__"):
                        # Direct list of messages
                        messages = list(conversation)
                        if debug:
                            print(
                                f"🔍 DEBUG: Conversation {conversation_id} has {len(messages)} messages"
                            )
                        for i, message in enumerate(messages):
                            total_messages += 1
                            if debug:
                                print(
                                    f"🔍 DEBUG: Message {i}: type={type(message)}, role={getattr(message, 'role', 'No role')}, content={getattr(message, 'content', 'No content')[:50]}..."
                                )
                            if hasattr(message, "role"):
                                if message.role == "user":
                                    user_messages += 1
                                elif message.role == "assistant":
                                    assistant_messages += 1
                                elif message.role == "system":
                                    system_prompt_count += 1
                            elif hasattr(message, "type"):
                                # Handle different message types
                                if message.type == "human":
                                    user_messages += 1
                                elif message.type == "ai":
                                    assistant_messages += 1
                                elif message.type == "system":
                                    system_prompt_count += 1
                            # Also check for SystemMessage instances
                            elif hasattr(message, "__class__") and "System" in message.__class__.__name__:
                                system_prompt_count += 1
                    else:
                        if debug:
                            print(
                                f"🔍 DEBUG: Unknown conversation type: {type(conversation)}"
                            )

                if debug:
                    print(
                        f"🔍 DEBUG: Total stats: {total_messages} total, {user_messages} user, {assistant_messages} assistant"
                    )
                # Get tool call count using the shared utility function
                tool_call_count = get_tool_call_count(self, self.session_id)
                
                return {
                    "message_count": total_messages,
                    "user_messages": user_messages,
                    "assistant_messages": assistant_messages,
                    "system_prompt_count": system_prompt_count,
                    "total_tool_calls": tool_call_count,
                }
            else:
                if debug:
                    print("🔍 DEBUG: No memory manager available")
                return {"message_count": 0, "user_messages": 0, "assistant_messages": 0, "system_prompt_count": 0, "total_tool_calls": 0}
        except Exception as e:
            print(f"🔍 DEBUG: Error getting conversation stats: {e}")
            import traceback

            traceback.print_exc()
            return {"message_count": 0, "user_messages": 0, "assistant_messages": 0}

    def get_token_counts(self, messages: List[Any]) -> Dict[str, Any]:
        """Get token counts for display"""
        if hasattr(self, "langchain_wrapper") and self.langchain_wrapper:
            return self.langchain_wrapper.get_token_counts(messages)
        return {"prompt_tokens": None, "cumulative_stats": {}}

    def get_token_display_info(self) -> Dict[str, Any]:
        """Get comprehensive token display information"""
        if hasattr(self, "token_tracker"):
            return self.token_tracker.get_token_display_info()
        return {"prompt_tokens": None, "api_tokens": None, "cumulative_stats": {}}

    def count_prompt_tokens_for_chat(
        self, history: List[Dict[str, str]], current_message: str
    ) -> Optional[TokenCount]:
        """Count prompt tokens for chat history and current message"""
        if hasattr(self, "token_tracker"):
            from .token_counter import convert_chat_history_to_messages

            messages = convert_chat_history_to_messages(history, current_message)
            return self.token_tracker.count_prompt_tokens(messages)
        return None

    def get_last_api_tokens(self) -> Optional[TokenCount]:
        """Get the last API token count"""
        if hasattr(self, "token_tracker"):
            return self.token_tracker.get_last_api_tokens()
        return None

    def get_token_budget_info(self) -> Dict[str, Any]:
        """Get token budget information for the current LLM context window"""
        if not hasattr(self, "token_tracker") or not self.token_tracker:
            return {
                "used_tokens": 0,
                "context_window": 0,
                "percentage": 0.0,
                "remaining_tokens": 0,
                "status": "unknown",
            }

        # Get context window from the agent's own LLM instance (session-specific)
        context_window = 0
        if hasattr(self, "llm_instance") and self.llm_instance and self.llm_instance.config:
            context_window = self.llm_instance.config.get("token_limit", 0)

        return self.token_tracker.get_token_budget_info(context_window)


# Note: Global agent instances have been removed in favor of session-specific agents
# Use SessionManager.get_session_agent(session_id) to get agent instances
