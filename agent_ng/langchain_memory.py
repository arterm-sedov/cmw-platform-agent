"""
LangChain Memory Management
==========================

This module provides proper LangChain memory management for multi-turn conversations
with tool calls using LangChain's native memory classes and patterns.

Key Features:
- ConversationBufferMemory for maintaining chat history
- Proper tool call context preservation
- LangChain native message formatting
- Support for multiple memory backends
- Integration with tool calling chains

Based on LangChain's memory documentation and best practices.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from threading import Lock

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.memory import BaseMemory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# LangSmith tracing
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    def traceable(func):
        return func

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
    from .utils import ensure_valid_answer
except ImportError as e1:
    try:
        from agent_ng.llm_manager import get_llm_manager, LLMInstance
        from agent_ng.utils import ensure_valid_answer
    except ImportError as e2:
        print(f"💥 CRITICAL ERROR: Cannot import required modules in langchain_memory!")
        print(f"   Relative import failed: {e1}")
        print(f"   Absolute import failed: {e2}")
        raise ImportError(f"Failed to import required modules in langchain_memory. Relative: {e1}, Absolute: {e2}")


@dataclass
class ConversationContext:
    """Context for a conversation with tool calls"""
    conversation_id: str
    messages: List[BaseMessage]
    tool_calls: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class ToolAwareMemory:
    """
    Custom memory class that properly handles tool calls in conversations.
    
    This provides a simple memory implementation that stores conversation
    history and tool calls for multi-turn conversations.
    """
    
    def __init__(self, memory_key: str = "chat_history", return_messages: bool = True):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.chat_memory = ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=return_messages
        )
        self.tool_calls_memory: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = Lock()
    
    @property
    def memory_variables(self) -> List[str]:
        """Return the list of memory variables"""
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables"""
        with self._lock:
            return self.chat_memory.load_memory_variables(inputs)
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save context including tool calls"""
        with self._lock:
            # Save regular chat memory
            self.chat_memory.save_context(inputs, outputs)
            
            # Save tool calls if present
            if "tool_calls" in outputs:
                conversation_id = inputs.get("conversation_id", "default")
                if conversation_id not in self.tool_calls_memory:
                    self.tool_calls_memory[conversation_id] = []
                
                # Store tool calls in separate memory
                self.tool_calls_memory[conversation_id].extend(outputs["tool_calls"])
                
                # Also add ToolMessage objects to the main chat history
                for tool_call in outputs["tool_calls"]:
                    tool_result = tool_call.get('result', '')
                    tool_call_id = tool_call.get('id', '')
                    tool_name = tool_call.get('name', '')
                    
                    if tool_result and tool_call_id:
                        tool_message = ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call_id,
                            name=tool_name
                        )
                        self.chat_memory.chat_memory.append(tool_message)
    
    def clear(self) -> None:
        """Clear memory"""
        with self._lock:
            self.chat_memory.clear()
            self.tool_calls_memory.clear()
    
    def get_tool_calls(self, conversation_id: str = "default") -> List[Dict[str, Any]]:
        """Get tool calls for a specific conversation"""
        with self._lock:
            return self.tool_calls_memory.get(conversation_id, []).copy()
    
    def add_tool_call(self, conversation_id: str, tool_call: Dict[str, Any]) -> None:
        """Add a tool call to memory"""
        with self._lock:
            if conversation_id not in self.tool_calls_memory:
                self.tool_calls_memory[conversation_id] = []
            self.tool_calls_memory[conversation_id].append(tool_call)


class ConversationMemoryManager:
    """
    Manages conversation memory using LangChain's native patterns.
    
    This class provides a clean interface for managing conversation
    memory with proper tool call support.
    """
    
    def __init__(self):
        self.memories: Dict[str, ToolAwareMemory] = {}
        self._lock = Lock()
    
    def get_memory(self, conversation_id: str = "default") -> ToolAwareMemory:
        """Get or create memory for a conversation"""
        with self._lock:
            if conversation_id not in self.memories:
                self.memories[conversation_id] = ToolAwareMemory()
            return self.memories[conversation_id]
    
    def clear_memory(self, conversation_id: str = "default") -> None:
        """Clear memory for a specific conversation"""
        with self._lock:
            if conversation_id in self.memories:
                self.memories[conversation_id].clear()
    
    def get_conversation_history(self, conversation_id: str = "default") -> List[BaseMessage]:
        """Get conversation history as LangChain messages"""
        memory = self.get_memory(conversation_id)
        variables = memory.load_memory_variables({})
        return variables.get("chat_history", [])
    
    def add_message(self, conversation_id: str, message: BaseMessage) -> None:
        """Add a message to conversation history"""
        memory = self.get_memory(conversation_id)
        memory.chat_memory.chat_memory.append(message)
    
    def add_tool_call(self, conversation_id: str, tool_call: Dict[str, Any]) -> None:
        """Add a tool call to memory"""
        memory = self.get_memory(conversation_id)
        memory.add_tool_call(conversation_id, tool_call)
    
    def get_tool_calls(self, conversation_id: str = "default") -> List[Dict[str, Any]]:
        """Get tool calls for a conversation"""
        memory = self.get_memory(conversation_id)
        return memory.get_tool_calls(conversation_id)


class LangChainConversationChain:
    """
    LangChain conversation chain with proper tool calling support.
    
    This class implements a conversation chain using LangChain's native
    patterns for multi-turn conversations with tool calls.
    """
    
    def __init__(self, llm_instance: LLMInstance, tools: List[BaseTool], system_prompt: str, agent=None):
        self.llm_instance = llm_instance
        self.tools = tools
        self.agent = agent
        self.system_prompt = system_prompt
        # Use agent's memory manager if available, otherwise create new one
        if agent and hasattr(agent, 'memory_manager'):
            self.memory_manager = agent.memory_manager
        else:
            self.memory_manager = ConversationMemoryManager()
        
        # Create the chain
        self.chain = self._create_chain()
    
    def _create_chain(self):
        """Create the LangChain conversation chain"""
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Use LLM with pre-bound tools (tools are already bound in the LLM instance)
        llm_with_tools = self.llm_instance.llm
        
        # Create the chain
        chain = prompt | llm_with_tools | StrOutputParser()
        
        return chain
    
    def process_message(self, message: str, conversation_id: str = "default") -> Dict[str, Any]:
        """
        Process a message in the conversation chain.
        
        Args:
            message: User message
            conversation_id: Conversation identifier
            
        Returns:
            Dict with response and metadata
        """
        try:
            # Get conversation history
            chat_history = self.memory_manager.get_conversation_history(conversation_id)
            
            # Prepare inputs
            inputs = {
                "input": message,
                "chat_history": chat_history,
                "conversation_id": conversation_id
            }
            
            # Process with chain
            response = self.chain.invoke(inputs)
            
            # Add user message to history
            self.memory_manager.add_message(conversation_id, HumanMessage(content=message))
            
            # Add AI response to history
            self.memory_manager.add_message(conversation_id, AIMessage(content=response))
            
            return {
                "response": response,
                "conversation_id": conversation_id,
                "tool_calls": [],
                "success": True
            }
            
        except Exception as e:
            return {
                "response": f"Error processing message: {str(e)}",
                "conversation_id": conversation_id,
                "tool_calls": [],
                "success": False,
                "error": str(e)
            }
    
    def process_with_tools(self, message: str, conversation_id: str = "default") -> Dict[str, Any]:
        """
        Process a message with tool calling support.
        
        This method handles the full tool calling loop using LangChain patterns.
        """
        try:
            # Get conversation history
            chat_history = self.memory_manager.get_conversation_history(conversation_id)
            
            # Create messages list for LLM context
            messages = []
            
            # Always add system message to LLM context (required for every call)
            system_message = SystemMessage(content=self.system_prompt)
            messages.append(system_message)
            
            # Check if system message is already in memory, if not add it
            system_in_history = any(isinstance(msg, SystemMessage) for msg in chat_history)
            if not system_in_history:
                # Store system message in memory only once
                self.memory_manager.add_message(conversation_id, system_message)
            
            # Add conversation history (excluding system messages to avoid duplication)
            non_system_history = [msg for msg in chat_history if not isinstance(msg, SystemMessage)]
            messages.extend(non_system_history)
            messages.append(HumanMessage(content=message))
            
            # Streaming is the single executor; memory does not run tool loop
            # Return a minimal response; persistence is handled by streaming turn
            return {
                "response": "Streaming-managed execution; memory does not execute tools",
                "conversation_id": conversation_id,
                "tool_calls": [],
                "success": True
            }
            
        except Exception as e:
            return {
                "response": f"Error processing message with tools: {str(e)}",
                "conversation_id": conversation_id,
                "tool_calls": [],
                "success": False,
                "error": str(e)
            }
    
    # NOTE: _run_tool_calling_loop was removed; streaming is the sole executor
    
    def _deduplicate_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Deduplicate tool calls and count duplicates BEFORE execution.
        
        Args:
            tool_calls: List of tool calls from LLM response
            
        Returns:
            Tuple of (deduplicated_tool_calls, duplicate_counts)
        """
        unique_tool_calls = []
        duplicate_counts = {}
        
        for tool_call in tool_calls:
            tool_name = tool_call.get('name', 'unknown')
            tool_args = tool_call.get('args', {})
            try:
                tool_key = f"{tool_name}:{hash(json.dumps(tool_args, sort_keys=True, default=str))}"
            except Exception:
                tool_key = f"{tool_name}:{hash(str(tool_args))}"
            
            if tool_key in duplicate_counts:
                # Increment count for duplicate
                duplicate_counts[tool_key] += 1
            else:
                # First occurrence - add to unique list and initialize count
                unique_tool_calls.append(tool_call)
                duplicate_counts[tool_key] = 1
        
        return unique_tool_calls, duplicate_counts

    def _track_token_usage(self, response, messages, conversation_id: str = "default"):
        """Track token usage for LLM response"""
        try:
            # Get token tracker from the agent
            if hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'token_tracker'):
                self.agent.token_tracker.track_llm_response(response, messages)
            else:
                # Create a simple token tracker if none exists
                from .token_counter import get_token_tracker
                token_tracker = get_token_tracker(conversation_id)
                token_tracker.track_llm_response(response, messages)
        except Exception as e:
            # Silently fail - token counting is not critical
            pass
    
    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """Execute a tool and return the result"""
        try:
            # Find the tool
            tool_func = None
            for tool in self.tools:
                if hasattr(tool, 'name') and tool.name == tool_name:
                    tool_func = tool
                    break
                elif hasattr(tool, '__name__') and tool.__name__ == tool_name:
                    tool_func = tool
                    break
            
            if not tool_func:
                return f"Error: Tool '{tool_name}' not found"
            
            # Inject agent instance for file resolution if available
            if hasattr(self, 'agent') and self.agent:
                tool_args['agent'] = self.agent
            
            # Execute the tool
            result = tool_func.invoke(tool_args) if hasattr(tool_func, 'invoke') else tool_func(**tool_args)
            
            # Ensure result is a string
            return ensure_valid_answer(result)
            
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"
    
    
    def get_conversation_history(self, conversation_id: str = "default") -> List[BaseMessage]:
        """Get conversation history"""
        return self.memory_manager.get_conversation_history(conversation_id)
    
    def clear_conversation(self, conversation_id: str = "default") -> None:
        """Clear conversation history"""
        self.memory_manager.clear_memory(conversation_id)


# Global memory manager instance
_memory_manager = None
_memory_lock = Lock()


def get_memory_manager() -> ConversationMemoryManager:
    """Get the global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        with _memory_lock:
            if _memory_manager is None:
                _memory_manager = ConversationMemoryManager()
    return _memory_manager


def create_conversation_chain(llm_instance: LLMInstance, tools: List[BaseTool], system_prompt: str, agent=None) -> LangChainConversationChain:
    """Create a new conversation chain"""
    return LangChainConversationChain(llm_instance, tools, system_prompt, agent)
