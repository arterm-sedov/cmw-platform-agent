"""
Core Agent Module
=================

This module provides the core agent functionality that uses persistent LLM objects
from the LLM manager and handles user questions with tool calling, conversation
management, and streaming capabilities.

Key Features:
- Uses persistent LLM instances from LLM manager
- Tool calling and execution
- Conversation management and history
- Streaming responses
- Error handling and fallback logic
- Vector similarity for answer matching

Usage:
    agent = CoreAgent()
    response = agent.process_question("What is the capital of France?")
"""

import json
import time
import uuid
from typing import Dict, List, Optional, Any, Generator, AsyncGenerator, Tuple
from dataclasses import dataclass
from collections import defaultdict
from threading import Lock

# LangChain imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.callbacks import BaseCallbackHandler

# Local imports
from llm_manager import get_llm_manager, LLMInstance
from error_handler import get_error_handler, ErrorInfo
from utils import ensure_valid_answer, TRACES_DIR
import tools
from tool_call_manager import tool_call_manager

# Vector store imports (conditional)
try:
    from similarity_manager import similarity_manager
    from vector_store import vector_store
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False


@dataclass
class ConversationMessage:
    """Structured conversation message"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentResponse:
    """Structured agent response"""
    answer: str
    confidence: float
    sources: List[str]
    reasoning: Optional[str]
    llm_used: str
    execution_time: float
    tool_calls: List[Dict[str, Any]]
    conversation_id: str


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming responses"""
    
    def __init__(self, agent_instance, streaming_generator=None):
        self.agent = agent_instance
        self.streaming_generator = streaming_generator
        self.current_tool_calls = []
        self.tool_results = []
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts"""
        if self.streaming_generator:
            self.streaming_generator("llm_start", "Starting LLM processing...")
    
    def on_llm_stream(self, chunk, **kwargs):
        """Called when LLM streams content"""
        if hasattr(chunk, 'content') and chunk.content:
            if self.streaming_generator:
                self.streaming_generator("content", chunk.content)
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Called when tool starts"""
        tool_name = serialized.get("name", "unknown_tool")
        if self.streaming_generator:
            self.streaming_generator("tool_start", f"Using tool: {tool_name}")
    
    def on_tool_end(self, output, **kwargs):
        """Called when tool ends"""
        if self.streaming_generator:
            self.streaming_generator("tool_end", f"Tool completed")
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM ends"""
        if self.streaming_generator:
            self.streaming_generator("llm_end", "LLM processing completed")


class CoreAgent:
    """
    Core agent that uses persistent LLM objects and handles user questions.
    
    This agent provides the main functionality for processing user questions
    using the modular LLM manager and error handler.
    """
    
    def __init__(self, enable_vector_similarity: bool = True):
        """
        Initialize the core agent.
        
        Args:
            enable_vector_similarity: Whether to enable vector similarity calculations
        """
        self.llm_manager = get_llm_manager()
        self.error_handler = get_error_handler()
        self.enable_vector_similarity = enable_vector_similarity and VECTOR_STORE_AVAILABLE
        
        # Conversation management
        self.conversations: Dict[str, List[ConversationMessage]] = defaultdict(list)
        self.conversation_metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.conversation_lock = Lock()
        
        # Agent state
        self.current_question = None
        self.current_file_data = None
        self.current_file_name = None
        self.total_questions = 0
        
        # Configuration
        self.max_conversation_history = 50
        self.similarity_threshold = 0.95
        self.tool_calls_similarity_threshold = 0.90
        self.max_tool_calls = 10
        self.max_consecutive_no_progress = 3
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt()
        self.sys_msg = SystemMessage(content=self.system_prompt)
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        print(f"🤖 Core Agent initialized with {len(self.tools)} tools")
        if self.enable_vector_similarity:
            print("📊 Vector similarity enabled")
        else:
            print("📊 Vector similarity disabled")
    
    def _load_system_prompt(self) -> str:
        """Load the system prompt from system_prompt.json"""
        try:
            with open("system_prompt.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("system_prompt", "You are a helpful AI assistant.")
        except FileNotFoundError:
            return "You are a helpful AI assistant."
        except Exception as e:
            print(f"Warning: Could not load system prompt: {e}")
            return "You are a helpful AI assistant."
    
    def _initialize_tools(self) -> List[Any]:
        """Initialize available tools"""
        tool_list = []
        for name, obj in tools.__dict__.items():
            if (callable(obj) and 
                not name.startswith("_") and 
                not isinstance(obj, type) and
                hasattr(obj, '__module__') and
                obj.__module__ == 'tools' and
                name not in ["CmwAgent", "CodeInterpreter"]):
                
                if hasattr(obj, 'name') and hasattr(obj, 'description'):
                    tool_list.append(obj)
                elif callable(obj) and not name.startswith("_"):
                    tool_list.append(obj)
        
        return tool_list
    
    def _get_reference_answer(self, question: str) -> Optional[str]:
        """Get reference answer using vector similarity if available"""
        if not self.enable_vector_similarity:
            return None
        
        try:
            # Use similarity manager to find similar questions
            similar_questions = similarity_manager.find_similar_questions(question, top_k=1)
            if similar_questions and similar_questions[0]['similarity'] > self.similarity_threshold:
                return similar_questions[0]['answer']
        except Exception as e:
            print(f"Warning: Vector similarity lookup failed: {e}")
        
        return None
    
    def _format_messages(self, question: str, reference: Optional[str] = None, 
                        chat_history: Optional[List[Dict[str, Any]]] = None) -> List[Any]:
        """Format messages for LLM"""
        messages = [self.sys_msg]
        
        # Add chat history if provided
        if chat_history:
            for msg in chat_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        
        # Add current question
        if reference:
            question_with_ref = f"Question: {question}\n\nReference Answer: {reference}\n\nPlease provide a comprehensive answer based on the reference and your knowledge."
        else:
            question_with_ref = question
        
        messages.append(HumanMessage(content=question_with_ref))
        
        return messages
    
    def _execute_tool(self, tool_name: str, tool_args: dict, call_id: str = None) -> str:
        """Execute a tool and return the result"""
        try:
            # Find the tool function
            tool_func = None
            for tool in self.tools:
                if hasattr(tool, 'name') and tool.name == tool_name:
                    tool_func = tool
                    break
                elif callable(tool) and tool.__name__ == tool_name:
                    tool_func = tool
                    break
            
            if not tool_func:
                return f"Error: Tool '{tool_name}' not found"
            
            # Inject file data if available
            if self.current_file_data and self.current_file_name:
                tool_args = self._inject_file_data_to_tool_args(tool_name, tool_args)
            
            # Execute the tool
            start_time = time.time()
            result = tool_func.invoke(tool_args) if hasattr(tool_func, 'invoke') else tool_func(**tool_args)
            execution_time = time.time() - start_time
            
            # Ensure result is a string
            result_str = ensure_valid_answer(result)
            
            print(f"🔧 Tool '{tool_name}' executed in {execution_time:.2f}s")
            return result_str
            
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def _inject_file_data_to_tool_args(self, tool_name: str, tool_args: dict) -> dict:
        """Inject file data into tool arguments if the tool supports it"""
        # List of tools that can handle file data
        file_tools = [
            'read_file', 'write_file', 'create_file', 'update_file',
            'analyze_file', 'process_file', 'extract_text', 'convert_file'
        ]
        
        if tool_name in file_tools and 'file_data' not in tool_args:
            tool_args['file_data'] = self.current_file_data
            tool_args['file_name'] = self.current_file_name
        
        return tool_args
    
    def _run_tool_calling_loop(self, llm_instance: LLMInstance, messages: List[Any], 
                              call_id: str = None, streaming_generator=None) -> Tuple[str, List[Dict[str, Any]]]:
        """Run the tool calling loop for the LLM"""
        tool_calls = []
        tool_results_history = []
        consecutive_no_progress = 0
        step = 0
        
        while step < self.max_tool_calls:
            step += 1
            
            try:
                # Make LLM call
                response = llm_instance.llm.invoke(messages)
                
                # Check if response has tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    # Process tool calls
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('args', {})
                        
                        # Execute tool
                        tool_result = self._execute_tool(tool_name, tool_args, call_id)
                        
                        # Store tool call and result
                        tool_calls.append({
                            'name': tool_name,
                            'args': tool_args,
                            'result': tool_result,
                            'step': step
                        })
                        
                        tool_results_history.append({
                            'tool_name': tool_name,
                            'tool_args': tool_args,
                            'tool_result': tool_result
                        })
                        
                        # Add tool message to conversation
                        messages.append(ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call.get('id', f"call_{len(tool_calls)}")
                        ))
                    
                    consecutive_no_progress = 0
                else:
                    # No tool calls, check if we should continue
                    consecutive_no_progress += 1
                    if consecutive_no_progress >= self.max_consecutive_no_progress:
                        break
                    
                    # Add AI message to conversation
                    messages.append(response)
                
            except Exception as e:
                print(f"❌ Error in tool calling loop: {e}")
                break
        
        # Get final response
        if messages and isinstance(messages[-1], AIMessage):
            final_response = messages[-1].content
        else:
            final_response = "I apologize, but I encountered an error while processing your request."
        
        return final_response, tool_calls
    
    def _process_with_llm(self, llm_instance: LLMInstance, messages: List[Any], 
                         call_id: str = None, streaming_generator=None) -> Tuple[str, List[Dict[str, Any]]]:
        """Process messages with a specific LLM instance"""
        try:
            if llm_instance.bound_tools:
                # Use tool calling loop
                return self._run_tool_calling_loop(llm_instance, messages, call_id, streaming_generator)
            else:
                # Simple LLM call without tools
                response = llm_instance.llm.invoke(messages)
                if hasattr(response, 'content'):
                    return response.content, []
                else:
                    return str(response), []
        
        except Exception as e:
            error_info = self.error_handler.classify_error(e, llm_instance.provider.value)
            print(f"❌ Error with {llm_instance.provider.value}: {error_info.description}")
            raise e
    
    def process_question(self, question: str, file_data: str = None, file_name: str = None,
                        llm_sequence: Optional[List[str]] = None, 
                        chat_history: Optional[List[Dict[str, Any]]] = None,
                        conversation_id: str = "default") -> AgentResponse:
        """
        Process a single question and return a structured response.
        
        Args:
            question: The question to answer
            file_data: Base64 encoded file data if a file is attached
            file_name: Name of the attached file
            llm_sequence: List of LLM provider names to try
            chat_history: Prior conversation history
            conversation_id: ID for conversation tracking
            
        Returns:
            AgentResponse with the answer and metadata
        """
        start_time = time.time()
        call_id = str(uuid.uuid4())
        
        # Store current question context
        self.current_question = question
        self.current_file_data = file_data
        self.current_file_name = file_name
        self.total_questions += 1
        
        print(f"\n🔎 Processing question: {question}")
        if file_data and file_name:
            print(f"📁 File attached: {file_name}")
        
        # Get reference answer
        reference = self._get_reference_answer(question)
        
        # Format messages
        messages = self._format_messages(question, reference, chat_history)
        
        # Update conversation history
        self._add_to_conversation(conversation_id, "user", question)
        
        # Get single LLM instance from environment
        llm_instance = self.llm_manager.get_agent_llm()
        if not llm_instance:
            final_answer = "Error: No LLM provider available. Check AGENT_PROVIDER environment variable."
            return final_answer, [], "none"
        
        final_answer = "I apologize, but I encountered an error while processing your request."
        tool_calls = []
        llm_used = f"{llm_instance.provider} ({llm_instance.model_name})"
        
        try:
            print(f"🤖 Using {llm_instance.provider} ({llm_instance.model_name})")
            
            # Process with LLM
            answer, calls = self._process_with_llm(llm_instance, messages, call_id)
            
            if answer and answer.strip():
                final_answer = answer
                tool_calls = calls
                print(f"✅ Success with {llm_instance.provider}")
            else:
                print(f"⚠️ {llm_instance.provider} returned empty response")
                final_answer = f"LLM returned empty response"
                
        except Exception as e:
            error_info = self.error_handler.classify_error(e, llm_instance.provider)
            print(f"❌ {llm_instance.provider} failed: {error_info.description}")
            
            # Track provider failure
            self.error_handler.handle_provider_failure(llm_instance.provider, error_info.error_type.value)
            final_answer = f"Error: {error_info.description}"
        
        # Calculate confidence based on success
        confidence = 0.9 if llm_used != "unknown" else 0.1
        
        # Extract sources from tool calls
        sources = [call['name'] for call in tool_calls]
        
        # Create response
        response = AgentResponse(
            answer=final_answer,
            confidence=confidence,
            sources=sources,
            reasoning=f"Processed using {llm_used} with {len(tool_calls)} tool calls",
            llm_used=llm_used,
            execution_time=time.time() - start_time,
            tool_calls=tool_calls,
            conversation_id=conversation_id
        )
        
        # Add to conversation history
        self._add_to_conversation(conversation_id, "assistant", final_answer)
        
        return response
    
    async def process_question_stream(self, question: str, file_data: str = None, file_name: str = None,
                              llm_sequence: Optional[List[str]] = None,
                              chat_history: Optional[List[Dict[str, Any]]] = None,
                              conversation_id: str = "default") -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a question with streaming responses.
        
        Args:
            question: The question to answer
            file_data: Base64 encoded file data if a file is attached
            file_name: Name of the attached file
            llm_sequence: List of LLM provider names to try
            chat_history: Prior conversation history
            conversation_id: ID for conversation tracking
            
        Yields:
            Dict with event_type and content
        """
        start_time = time.time()
        call_id = str(uuid.uuid4())
        
        # Store current question context
        self.current_question = question
        self.current_file_data = file_data
        self.current_file_name = file_name
        self.total_questions += 1
        
        yield {"event_type": "start", "content": f"Processing question: {question}"}
        
        if file_data and file_name:
            yield {"event_type": "file_info", "content": f"File attached: {file_name}"}
        
        # Get reference answer
        reference = self._get_reference_answer(question)
        
        # Format messages
        messages = self._format_messages(question, reference, chat_history)
        
        # Update conversation history
        self._add_to_conversation(conversation_id, "user", question)
        
        # Get single provider from environment
        if not llm_sequence:
            import os
            agent_provider = os.environ.get("AGENT_PROVIDER", "mistral")
            llm_sequence = [agent_provider]
        
        # Use single provider
        final_answer = "I apologize, but I encountered an error while processing your request."
        tool_calls = []
        llm_used = "unknown"
        
        # Get single LLM instance from environment
        llm_instance = self.llm_manager.get_agent_llm()
        if not llm_instance:
            yield {"event_type": "error", "content": "No LLM provider available. Check AGENT_PROVIDER environment variable."}
            return
        
        try:
            yield {"event_type": "llm_start", "content": f"Using {llm_instance.provider} ({llm_instance.model_name})"}
            
            # Process with LLM
            answer, calls = self._process_with_llm(llm_instance, messages, call_id)
            
            if answer and answer.strip():
                final_answer = answer
                tool_calls = calls
                llm_used = f"{llm_instance.provider} ({llm_instance.model_name})"
                yield {"event_type": "success", "content": f"Success with {llm_instance.provider}"}
            else:
                yield {"event_type": "warning", "content": f"{llm_instance.provider} returned empty response"}
                final_answer = f"LLM returned empty response"
                
        except Exception as e:
            error_info = self.error_handler.classify_error(e, llm_instance.provider)
            yield {"event_type": "error", "content": f"{llm_instance.provider} failed: {error_info.description}"}
            
            # Track provider failure
            self.error_handler.handle_provider_failure(llm_instance.provider, error_info.error_type.value)
            final_answer = f"Error: {error_info.description}"
        
        # Stream the final answer
        yield {"event_type": "answer", "content": final_answer}
        
        # Add to conversation history
        self._add_to_conversation(conversation_id, "assistant", final_answer)
        
        # Final metadata
        yield {
            "event_type": "complete",
            "content": {
                "llm_used": llm_used,
                "tool_calls": len(tool_calls),
                "execution_time": time.time() - start_time
            }
        }
    
    def _add_to_conversation(self, conversation_id: str, role: str, content: str, 
                           metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation history"""
        with self.conversation_lock:
            message = ConversationMessage(
                role=role,
                content=content,
                timestamp=time.time(),
                metadata=metadata
            )
            self.conversations[conversation_id].append(message)
            
            # Trim conversation if too long
            if len(self.conversations[conversation_id]) > self.max_conversation_history:
                self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_conversation_history:]
    
    def get_conversation_history(self, conversation_id: str = "default") -> List[Dict[str, Any]]:
        """Get conversation history for a specific conversation"""
        with self.conversation_lock:
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "metadata": msg.metadata
                }
                for msg in self.conversations[conversation_id]
            ]
    
    def clear_conversation(self, conversation_id: str = "default"):
        """Clear conversation history for a specific conversation"""
        with self.conversation_lock:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
            if conversation_id in self.conversation_metadata:
                del self.conversation_metadata[conversation_id]
    
    def get_conversation_stats(self, conversation_id: str = "default") -> Dict[str, Any]:
        """Get statistics for a conversation"""
        with self.conversation_lock:
            messages = self.conversations.get(conversation_id, [])
            return {
                "message_count": len(messages),
                "user_messages": len([m for m in messages if m.role == "user"]),
                "assistant_messages": len([m for m in messages if m.role == "assistant"]),
                "last_message_time": messages[-1].timestamp if messages else None
            }
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get overall agent statistics"""
        return {
            "total_questions": self.total_questions,
            "active_conversations": len(self.conversations),
            "tools_available": len(self.tools),
            "vector_similarity_enabled": self.enable_vector_similarity,
            "llm_manager_stats": self.llm_manager.get_stats(),
            "error_handler_stats": self.error_handler.get_provider_failure_stats()
        }


# Global agent instance
_agent = None
_agent_lock = Lock()


def get_agent() -> CoreAgent:
    """Get the global agent instance (singleton pattern)"""
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = CoreAgent()
    return _agent


def reset_agent():
    """Reset the global agent (useful for testing)"""
    global _agent
    with _agent_lock:
        _agent = None
