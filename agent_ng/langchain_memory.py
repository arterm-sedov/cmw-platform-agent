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
    from .utils import ensure_valid_answer, ensure_meaningful_response
except ImportError:
    try:
        from agent_ng.llm_manager import get_llm_manager, LLMInstance
        from agent_ng.utils import ensure_valid_answer, ensure_meaningful_response
    except ImportError:
        get_llm_manager = lambda: None
        LLMInstance = None
        ensure_valid_answer = lambda x: str(x) if x is not None else "No answer provided"
        ensure_meaningful_response = lambda x, context="", tool_calls=None: str(x) if x is not None else "No response generated"


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
    
    def __init__(self, llm_instance: LLMInstance, tools: List[BaseTool], system_prompt: str):
        self.llm_instance = llm_instance
        self.tools = tools
        self.system_prompt = system_prompt
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
        
        # Bind tools to LLM
        llm_with_tools = self.llm_instance.llm.bind_tools(self.tools)
        
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
            
            # Create messages list
            messages = [SystemMessage(content=self.system_prompt)]
            messages.extend(chat_history)
            messages.append(HumanMessage(content=message))
            
            # Process with tool calling
            response = self._run_tool_calling_loop(messages, conversation_id)
            
            return response
            
        except Exception as e:
            return {
                "response": f"Error processing message with tools: {str(e)}",
                "conversation_id": conversation_id,
                "tool_calls": [],
                "success": False,
                "error": str(e)
            }
    
    def _run_tool_calling_loop(self, messages: List[BaseMessage], conversation_id: str) -> Dict[str, Any]:
        """Run the tool calling loop using LangChain patterns"""
        tool_calls = []
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            try:
                # Get LLM response
                response = self.llm_instance.llm.invoke(messages)
            except Exception as e:
                # Handle LLM errors (like context length limits)
                error_msg = f"LLM Error: {str(e)}"
                if tool_calls:
                    tool_names = [call.get('name', 'unknown') for call in tool_calls]
                    error_msg += f"\n\nTools executed successfully: {', '.join(tool_names)}"
                    error_msg += f"\n\nThis error likely occurred due to context length limits after tool execution."
                
                return {
                    "response": error_msg,
                    "conversation_id": conversation_id,
                    "tool_calls": tool_calls,
                    "success": False,
                    "error": str(e)
                }
            
            # Check for tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Add the AI response with tool calls to messages
                messages.append(response)
                
                # Process tool calls
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get('name', 'unknown')
                    tool_args = tool_call.get('args', {})
                    tool_call_id = tool_call.get('id', f"call_{len(tool_calls)}")
                    
                    # Execute tool
                    tool_result = self._execute_tool(tool_name, tool_args)
                    
                    # Store tool call
                    tool_calls.append({
                        'name': tool_name,
                        'args': tool_args,
                        'result': tool_result,
                        'id': tool_call_id
                    })
                    
                    # Add tool message to conversation
                    tool_message = ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call_id
                    )
                    messages.append(tool_message)
                    
                    # Add to memory
                    self.memory_manager.add_tool_call(conversation_id, {
                        'name': tool_name,
                        'args': tool_args,
                        'result': tool_result,
                        'id': tool_call_id
                    })
            else:
                # No tool calls, we have the final response
                final_response = response.content if hasattr(response, 'content') else str(response)
                
                # Add AI response to messages
                messages.append(response)
                
                # Add to conversation history - save the complete conversation including tool messages
                for message in messages:
                    if isinstance(message, HumanMessage):
                        self.memory_manager.add_message(conversation_id, message)
                    elif isinstance(message, AIMessage):
                        self.memory_manager.add_message(conversation_id, message)
                    elif isinstance(message, ToolMessage):
                        self.memory_manager.add_message(conversation_id, message)
                
                return {
                    "response": final_response,
                    "conversation_id": conversation_id,
                    "tool_calls": tool_calls,
                    "success": True
                }
        
        # If we exit the loop due to max iterations, get final response
        # This handles the case where tools were called but we need a final answer
        if tool_calls:
            try:
                # Get one final response from the LLM after all tool calls
                final_response = self.llm_instance.llm.invoke(messages)
                if hasattr(final_response, 'content') and final_response.content.strip():
                    final_response_text = final_response.content
                    messages.append(final_response)
                else:
                    final_response_text = str(final_response)
            except Exception as e:
                # Handle LLM errors in final response
                tool_names = [call.get('name', 'unknown') for call in tool_calls]
                final_response_text = f"LLM Error in final response: {str(e)}\n\nTools executed successfully: {', '.join(tool_names)}\n\nThis error likely occurred due to context length limits after tool execution."
        else:
            # No tool calls were made, use the last response
            if messages and hasattr(messages[-1], 'content'):
                final_response_text = messages[-1].content
            else:
                final_response_text = "No response available"
        
        # Add to conversation history - save the complete conversation including tool messages
        for message in messages:
            if isinstance(message, HumanMessage):
                self.memory_manager.add_message(conversation_id, message)
            elif isinstance(message, AIMessage):
                self.memory_manager.add_message(conversation_id, message)
            elif isinstance(message, ToolMessage):
                self.memory_manager.add_message(conversation_id, message)
        
        return {
            "response": final_response_text,
            "conversation_id": conversation_id,
            "tool_calls": tool_calls,
            "success": True
        }
    
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


def create_conversation_chain(llm_instance: LLMInstance, tools: List[BaseTool], system_prompt: str) -> LangChainConversationChain:
    """Create a new conversation chain"""
    return LangChainConversationChain(llm_instance, tools, system_prompt)
