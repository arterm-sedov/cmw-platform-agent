"""
Tool Manager Module
==================

This module handles all tool-related functionality including execution, management, and integration.
It provides utilities for tool calling, execution tracking, and tool registry management.

Key Features:
- Tool execution and management
- Tool calling loop orchestration
- Tool registry and discovery
- Tool call tracking and history
- LangChain tool integration

Usage:
    from tool_manager import ToolManager
    
    tool_mgr = ToolManager()
    result = tool_mgr.execute_tool("multiply", {"a": 5, "b": 7})
    tools = tool_mgr.get_available_tools()
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Union, Generator
from dataclasses import dataclass
from langchain_core.tools import Tool
from langchain_core.messages import ToolMessage, AIMessage


@dataclass
class ToolCall:
    """Represents a tool call"""
    name: str
    args: Dict[str, Any]
    call_id: str
    timestamp: float
    result: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class ToolExecutionResult:
    """Represents the result of a tool execution"""
    success: bool
    result: str
    error: Optional[str] = None
    execution_time: float = 0.0
    call_id: str = None


class ToolManager:
    """Manages tool execution, calling, and integration"""
    
    def __init__(self):
        self.tool_registry: Dict[str, Any] = {}
        self.tool_call_history: List[ToolCall] = []
        self.called_tools: set = set()
        self.tool_results_history: List[str] = []
        self._load_tools()
    
    def _load_tools(self):
        """Load available tools from tools.py"""
        try:
            import tools
            for name, obj in tools.__dict__.items():
                if (callable(obj) and 
                    not name.startswith("_") and 
                    not isinstance(obj, type) and
                    hasattr(obj, 'name') and 
                    hasattr(obj, 'description') and 
                    hasattr(obj, 'args_schema')):
                    self.tool_registry[name] = obj
        except Exception as e:
            print(f"[ToolManager] Warning: Failed to load tools: {e}")
    
    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any], call_id: str = None) -> ToolExecutionResult:
        """
        Execute a tool with the given name and arguments.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            call_id: Optional call ID for tracking
            
        Returns:
            ToolExecutionResult: Result of tool execution
        """
        if call_id is None:
            call_id = str(uuid.uuid4())
        
        start_time = time.time()
        
        try:
            # Check if tool exists
            if tool_name not in self.tool_registry:
                error_msg = f"Tool '{tool_name}' not found in registry"
                return ToolExecutionResult(
                    success=False,
                    result="",
                    error=error_msg,
                    execution_time=0.0,
                    call_id=call_id
                )
            
            # Get tool function
            tool_func = self.tool_registry[tool_name]
            
            # Execute tool
            result = tool_func.invoke(tool_args)
            
            execution_time = time.time() - start_time
            
            # Track tool call
            tool_call = ToolCall(
                name=tool_name,
                args=tool_args,
                call_id=call_id,
                timestamp=start_time,
                result=str(result),
                execution_time=execution_time
            )
            self.tool_call_history.append(tool_call)
            self.called_tools.add(tool_name)
            self.tool_results_history.append(f"[{tool_name}] {result}")
            
            return ToolExecutionResult(
                success=True,
                result=str(result),
                execution_time=execution_time,
                call_id=call_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            
            # Track failed tool call
            tool_call = ToolCall(
                name=tool_name,
                args=tool_args,
                call_id=call_id,
                timestamp=start_time,
                error=error_msg,
                execution_time=execution_time
            )
            self.tool_call_history.append(tool_call)
            
            return ToolExecutionResult(
                success=False,
                result="",
                error=error_msg,
                execution_time=execution_time,
                call_id=call_id
            )
    
    def run_tool_calling_loop(self, llm, messages: List[Any], llm_type: str = "unknown", 
                            model_index: int = 0, call_id: str = None, 
                            streaming_generator=None, max_steps: int = 20) -> Dict[str, Any]:
        """
        Run a tool-calling loop: repeatedly invoke the LLM, detect tool calls, execute tools, 
        and feed results back until a final answer is produced.
        
        Args:
            llm: LLM instance
            messages: List of messages
            llm_type: Type of LLM
            model_index: Model index
            call_id: Optional call ID
            streaming_generator: Optional streaming generator
            max_steps: Maximum number of steps
            
        Returns:
            Dict: Result containing answer, tool_calls, and metadata
        """
        if call_id is None:
            call_id = str(uuid.uuid4())
        
        # Adaptive step limits based on LLM type
        step_limits = {
            "gemini": 25,
            "groq": 15,
            "huggingface": 20,
            "openrouter": 20,
            "mistral": 20,
            "gigachat": 20
        }
        max_steps = step_limits.get(llm_type, max_steps)
        
        # Reset tracking for new loop
        self.called_tools.clear()
        self.tool_results_history.clear()
        
        step = 0
        consecutive_no_change_steps = 0
        last_response_content = ""
        
        while step < max_steps:
            step += 1
            print(f"[Tool Loop] Step {step}/{max_steps}")
            
            try:
                # Invoke LLM
                if streaming_generator:
                    # Handle streaming
                    response = next(streaming_generator, None)
                    if response is None:
                        break
                else:
                    # Regular invocation
                    response = llm.invoke(messages)
                
                # Check for tool calls
                tool_calls = self._extract_tool_calls(response)
                
                if tool_calls:
                    print(f"[Tool Loop] Found {len(tool_calls)} tool calls")
                    
                    # Execute tools
                    tool_messages = []
                    for tool_call in tool_calls:
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('args', {})
                        tool_call_id = tool_call.get('id', str(uuid.uuid4()))
                        
                        # Execute tool
                        result = self.execute_tool(tool_name, tool_args, tool_call_id)
                        
                        # Create tool message
                        tool_message = ToolMessage(
                            content=result.result if result.success else f"Error: {result.error}",
                            tool_call_id=tool_call_id
                        )
                        tool_messages.append(tool_message)
                    
                    # Add tool results to messages
                    messages.extend(tool_messages)
                    
                    # Reset no-change counter
                    consecutive_no_change_steps = 0
                    
                else:
                    # No tool calls - check if we have a final answer
                    final_answer = self._extract_final_answer(response)
                    if final_answer:
                        print(f"[Tool Loop] Final answer found: {final_answer[:100]}...")
                        return {
                            'answer': final_answer,
                            'tool_calls': len(self.tool_call_history),
                            'steps': step,
                            'llm_used': llm_type,
                            'tool_results': self.tool_results_history
                        }
                    
                    # Check for content changes
                    current_content = getattr(response, 'content', '')
                    if current_content == last_response_content:
                        consecutive_no_change_steps += 1
                        if consecutive_no_change_steps >= 3:
                            print("[Tool Loop] No progress detected, forcing final answer")
                            break
                    else:
                        consecutive_no_change_steps = 0
                        last_response_content = current_content
                
            except Exception as e:
                print(f"[Tool Loop] Error in step {step}: {e}")
                break
        
        # If we reach here, we didn't get a final answer
        print("[Tool Loop] Max steps reached or error occurred")
        return {
            'answer': 'I apologize, but I was unable to provide a complete answer within the allowed steps.',
            'tool_calls': len(self.tool_call_history),
            'steps': step,
            'llm_used': llm_type,
            'tool_results': self.tool_results_history,
            'error': 'Max steps reached'
        }
    
    def _extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM response"""
        tool_calls = []
        
        try:
            # Check for tool_calls attribute
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_calls.extend(response.tool_calls)
            
            # Check for function_call attribute (legacy)
            if hasattr(response, 'function_call') and response.function_call:
                tool_calls.append(response.function_call)
        
        except Exception as e:
            print(f"[ToolManager] Error extracting tool calls: {e}")
        
        return tool_calls
    
    def _extract_final_answer(self, response: Any) -> Optional[str]:
        """Extract final answer from response"""
        try:
            # Check for submit_answer tool call
            tool_calls = self._extract_tool_calls(response)
            for tool_call in tool_calls:
                if tool_call.get('name') == 'submit_answer':
                    args = tool_call.get('args', {})
                    return args.get('answer', '')
            
            # Check content for answer patterns
            content = getattr(response, 'content', '')
            if isinstance(content, str) and len(content.strip()) > 10:
                return content.strip()
        
        except Exception as e:
            print(f"[ToolManager] Error extracting final answer: {e}")
        
        return None
    
    def get_available_tools(self) -> List[Any]:
        """Get list of available tools"""
        return list(self.tool_registry.values())
    
    def get_tool_names(self) -> List[str]:
        """Get list of available tool names"""
        return list(self.tool_registry.keys())
    
    def get_langchain_tools(self) -> List[Tool]:
        """Get tools in LangChain format"""
        try:
            tools = []
            for name, tool_obj in self.tool_registry.items():
                if hasattr(tool_obj, 'name') and hasattr(tool_obj, 'description'):
                    tools.append(tool_obj)
            return tools
        except Exception as e:
            print(f"[ToolManager] Error getting LangChain tools: {e}")
            return []
    
    def bind_tools(self, llm, tools: List[Any] = None) -> Any:
        """Bind tools to an LLM instance"""
        if tools is None:
            tools = self.get_available_tools()
        
        try:
            return llm.bind_tools(tools)
        except Exception as e:
            print(f"[ToolManager] Error binding tools: {e}")
            return llm
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics"""
        total_calls = len(self.tool_call_history)
        successful_calls = sum(1 for call in self.tool_call_history if call.error is None)
        failed_calls = total_calls - successful_calls
        
        tool_usage = {}
        for call in self.tool_call_history:
            tool_usage[call.name] = tool_usage.get(call.name, 0) + 1
        
        return {
            'total_tools': len(self.tool_registry),
            'total_calls': total_calls,
            'successful_calls': successful_calls,
            'failed_calls': failed_calls,
            'success_rate': successful_calls / total_calls if total_calls > 0 else 0,
            'tool_usage': tool_usage,
            'called_tools': list(self.called_tools)
        }
    
    def clear_history(self):
        """Clear tool call history"""
        self.tool_call_history.clear()
        self.called_tools.clear()
        self.tool_results_history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive tool manager statistics"""
        return {
            'tool_registry_size': len(self.tool_registry),
            'tool_names': self.get_tool_names(),
            'tool_stats': self.get_tool_stats()
        }


# Global tool manager instance
_tool_manager = None

def get_tool_manager() -> ToolManager:
    """Get the global tool manager instance"""
    global _tool_manager
    if _tool_manager is None:
        _tool_manager = ToolManager()
    return _tool_manager

def reset_tool_manager():
    """Reset the global tool manager instance"""
    global _tool_manager
    _tool_manager = None
