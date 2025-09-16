"""
Simple LangChain Streaming Implementation
========================================

This module implements proper LangChain streaming using the official patterns
from the LangChain documentation. It uses the native streaming capabilities
without any artificial delays.

Key Features:
- Uses LangChain's native streaming methods (astream, stream)
- Properly configured LLM instances with streaming=True
- Real-time token-by-token streaming
- Tool calling support with streaming
- No artificial delays - only natural timing

Based on: https://python.langchain.com/docs/concepts/streaming/
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler


@dataclass
class StreamingEvent:
    """Represents a streaming event"""
    event_type: str
    content: str
    metadata: Dict[str, Any] = None


class SimpleStreamingCallbackHandler(BaseCallbackHandler):
    """Simple callback handler for streaming responses"""
    
    def __init__(self, event_callback=None):
        self.event_callback = event_callback
        self.current_tool = None
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts"""
        if self.event_callback:
            self.event_callback(StreamingEvent(
                event_type="thinking",
                content="🤖 **Thinking...**",
                metadata={"llm_type": serialized.get("name", "unknown")}
            ))
    
    def on_llm_stream(self, chunk, **kwargs):
        """Called when LLM streams content - REAL STREAMING"""
        if hasattr(chunk, 'content') and chunk.content and self.event_callback:
            self.event_callback(StreamingEvent(
                event_type="content",
                content=chunk.content,
                metadata={"chunk_type": "llm_response"}
            ))
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Called when a tool starts"""
        self.current_tool = serialized.get("name", "unknown_tool")
        if self.event_callback:
            self.event_callback(StreamingEvent(
                event_type="tool_start",
                content=f"🔧 **Using tool: {self.current_tool}**",
                metadata={
                    "tool_name": self.current_tool,
                    "tool_args": input_str
                }
            ))
    
    def on_tool_end(self, output, **kwargs):
        """Called when a tool ends"""
        if self.event_callback:
            self.event_callback(StreamingEvent(
                event_type="tool_end",
                content=f"✅ **{self.current_tool} completed**",
                metadata={"tool_name": self.current_tool}
            ))
        self.current_tool = None


class SimpleLangChainStreaming:
    """
    Simple LangChain streaming implementation using native patterns.
    
    This class implements proper streaming using LangChain's official
    streaming methods with correctly configured LLM instances.
    """
    
    def __init__(self):
        self.active_streams = {}
    
    async def stream_agent_response(
        self,
        agent,
        message: str,
        conversation_id: str = "default"
    ) -> AsyncGenerator[StreamingEvent, None]:
        """
        Stream agent response using proper LangChain streaming with tool calling loop.
        
        Args:
            agent: The LangChain agent instance
            message: User message
            conversation_id: Conversation identifier
            
        Yields:
            StreamingEvent objects with real-time content
        """
        try:
            # Get conversation history
            chat_history = agent.memory_manager.get_conversation_history(conversation_id)
            
            # Create messages list
            messages = [SystemMessage(content=agent.system_prompt)]
            messages.extend(chat_history)
            
            # Create user message and save to memory
            user_message = HumanMessage(content=message)
            messages.append(user_message)
            agent.memory_manager.add_message(conversation_id, user_message)
            
            # Get LLM with streaming enabled
            llm = agent.llm_instance.llm
            
            # Ensure streaming is enabled
            if hasattr(llm, 'streaming') and not llm.streaming:
                # Create a new LLM instance with streaming enabled
                llm_config = llm.dict() if hasattr(llm, 'dict') else {}
                llm_config['streaming'] = True
                
                # Recreate the LLM with streaming enabled
                llm_class = type(llm)
                llm = llm_class(**llm_config)
            
            # Bind tools if available
            if agent.tools:
                llm_with_tools = llm.bind_tools(agent.tools)
            else:
                llm_with_tools = llm
            
            # Track API tokens
            last_response = None
            max_iterations = 10
            iteration = 0
            
            # Tool calling loop with streaming
            while iteration < max_iterations:
                iteration += 1
                
                # Get LLM response (use invoke for better tool calling support)
                try:
                    # Try streaming first
                    response_content = ""
                    tool_calls_detected = False
                    last_response = None
                    
                    async for chunk in llm_with_tools.astream(messages):
                        # Handle content chunks
                        if hasattr(chunk, 'content') and chunk.content:
                            response_content += chunk.content
                            # Stream content in real-time
                            yield StreamingEvent(
                                event_type="content",
                                content=chunk.content,
                                metadata={"chunk_type": "llm_response"}
                            )
                        
                        # Check for tool calls
                        if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                            tool_calls_detected = True
                        
                        # Track the response for API token counting
                        last_response = chunk
                    
                    # If no streaming response, get the full response
                    if not last_response:
                        last_response = await llm_with_tools.ainvoke(messages)
                        if hasattr(last_response, 'content') and last_response.content:
                            response_content = last_response.content
                            yield StreamingEvent(
                                event_type="content",
                                content=last_response.content,
                                metadata={"chunk_type": "llm_response"}
                            )
                        
                        if hasattr(last_response, 'tool_calls') and last_response.tool_calls:
                            tool_calls_detected = True
                    
                except Exception as e:
                    # Fallback to invoke if streaming fails
                    last_response = await llm_with_tools.ainvoke(messages)
                    if hasattr(last_response, 'content') and last_response.content:
                        response_content = last_response.content
                        yield StreamingEvent(
                            event_type="content",
                            content=last_response.content,
                            metadata={"chunk_type": "llm_response"}
                        )
                    
                    if hasattr(last_response, 'tool_calls') and last_response.tool_calls:
                        tool_calls_detected = True
                
                # Process tool calls if detected
                if tool_calls_detected and last_response and hasattr(last_response, 'tool_calls') and last_response.tool_calls:
                    # Add the AI response to messages
                    messages.append(last_response)
                    
                    # Save AI response to memory
                    agent.memory_manager.add_message(conversation_id, last_response)
                    
                    # Process tool calls
                    async for event in self._process_tool_calls(
                        last_response.tool_calls, agent.tools, messages, 
                        conversation_id, agent.memory_manager
                    ):
                        yield event
                    
                    # Continue the loop to let LLM process tool results
                    continue
                else:
                    # No tool calls, we have the final response
                    if response_content:
                        # Add AI response to messages
                        if last_response:
                            messages.append(last_response)
                            # Save AI response to memory
                            agent.memory_manager.add_message(conversation_id, last_response)
                        break
                    else:
                        # Empty response, retry
                        messages.append(HumanMessage(content="Please provide a meaningful response."))
                        continue
            
            # Track API tokens after streaming
            if last_response and hasattr(agent, 'token_tracker'):
                try:
                    if hasattr(last_response, 'usage_metadata') and last_response.usage_metadata:
                        agent.token_tracker.track_llm_response(last_response, messages)
                except Exception as e:
                    print(f"🔍 DEBUG: Error tracking API tokens: {e}")
            
            # Final completion event
            yield StreamingEvent(
                event_type="completion",
                content="✅ **Response completed**",
                metadata={"final_response": last_response}
            )
                
        except Exception as e:
            yield StreamingEvent(
                event_type="error",
                content=f"❌ **Error: {str(e)}**",
                metadata={"error": str(e)}
            )
    
    async def _process_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        tools: List[BaseTool],
        messages: List[BaseMessage],
        conversation_id: str,
        memory_manager
    ) -> AsyncGenerator[StreamingEvent, None]:
        """
        Process tool calls with streaming events and proper result display.
        """
        # Create tool registry
        tool_registry = {tool.name: tool for tool in tools}
        
        for tool_call in tool_calls:
            tool_name = tool_call.get('name', 'unknown')
            tool_args = tool_call.get('args', {})
            tool_call_id = tool_call.get('id', f"call_{len(tool_calls)}")
            
            if tool_name in tool_registry:
                # Stream tool start
                yield StreamingEvent(
                    event_type="tool_start",
                    content=f"🔧 **Using tool: {tool_name}**",
                    metadata={
                        "tool_name": tool_name,
                        "tool_args": tool_args
                    }
                )
                
                try:
                    # Execute tool
                    tool_result = tool_registry[tool_name].invoke(tool_args)
                    
                    # Stream tool result content
                    yield StreamingEvent(
                        event_type="content",
                        content=f"\n\n**Tool Result:** {str(tool_result)}\n",
                        metadata={
                            "tool_name": tool_name,
                            "tool_output": str(tool_result)
                        }
                    )
                    
                    # Stream tool end
                    yield StreamingEvent(
                        event_type="tool_end",
                        content=f"✅ **{tool_name} completed**",
                        metadata={
                            "tool_name": tool_name,
                            "tool_output": str(tool_result)
                        }
                    )
                    
                    # Add tool message to conversation
                    tool_message = ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call_id
                    )
                    messages.append(tool_message)
                    
                    # Add to memory
                    memory_manager.add_tool_call(conversation_id, {
                        'name': tool_name,
                        'args': tool_args,
                        'result': tool_result,
                        'id': tool_call_id
                    })
                    
                    memory_manager.add_message(conversation_id, tool_message)
                    
                except Exception as e:
                    yield StreamingEvent(
                        event_type="error",
                        content=f"❌ **Tool error: {str(e)}**",
                        metadata={
                            "tool_name": tool_name,
                            "error": str(e)
                        }
                    )
            else:
                yield StreamingEvent(
                    event_type="error",
                    content=f"❌ **Unknown tool: {tool_name}**",
                    metadata={"tool_name": tool_name}
                )


# Global streaming instance
_streaming_instance = None

def get_simple_streaming() -> SimpleLangChainStreaming:
    """Get the global simple streaming instance"""
    global _streaming_instance
    if _streaming_instance is None:
        _streaming_instance = SimpleLangChainStreaming()
    return _streaming_instance
