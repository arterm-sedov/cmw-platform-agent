"""
Native LangChain Streaming Implementation
========================================

This module implements proper LangChain native streaming using astream()
and astream_events() for real-time token-by-token streaming.

Key Features:
- Native LangChain streaming (astream, astream_events)
- Real-time token-by-token streaming
- Proper tool calling with streaming
- No artificial delays
- Uses LangChain's built-in streaming capabilities
"""

import asyncio
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


class NativeLangChainStreaming:
    """
    Native LangChain streaming implementation using astream() and astream_events().
    """
    
    def __init__(self):
        self.active_streams = {}
        # Get configuration from centralized config
        from .streaming_config import get_streaming_config
        self.config = get_streaming_config()
        self.max_iterations = self.config.get_max_tool_call_iterations()
    
    async def stream_agent_response(
        self,
        agent,
        message: str,
        conversation_id: str = "default"
    ) -> AsyncGenerator[StreamingEvent, None]:
        """
        Stream agent response using native LangChain streaming with proper tool call handling.
        
        Based on: https://python.langchain.com/docs/how_to/tool_streaming/
        
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
            
            # Get LLM with tools
            llm = agent.llm_instance.llm
            if agent.tools:
                llm_with_tools = llm.bind_tools(agent.tools)
            else:
                llm_with_tools = llm
            
            # Multi-turn conversation loop for proper tool calling
            iteration = 0
            
            while iteration < self.max_iterations:
                iteration += 1
                print(f"🔍 DEBUG: Starting iteration {iteration}")
                
                # Use proper LangChain streaming with tool call handling
                accumulated_chunk = None
                tool_calls_in_progress = {}
                has_tool_calls = False
                
                async for chunk in llm_with_tools.astream(messages):
                    # Accumulate chunks for proper tool call parsing
                    if accumulated_chunk is None:
                        accumulated_chunk = chunk
                    else:
                        accumulated_chunk = accumulated_chunk + chunk
                    
                    # Stream content as it arrives
                    if hasattr(chunk, 'content') and chunk.content:
                        yield StreamingEvent(
                            event_type="content",
                            content=chunk.content,
                            metadata={"chunk_type": "llm_stream", "provider": agent.llm_instance.provider.value}
                        )
                    
                    # Process tool call chunks as they stream
                    if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                        for tool_call_chunk in chunk.tool_call_chunks:
                            tool_name = tool_call_chunk.get('name')
                            tool_call_id = tool_call_chunk.get('id')
                            tool_args = tool_call_chunk.get('args', '')
                            
                            if tool_name and tool_call_id not in tool_calls_in_progress:
                                # New tool call starting
                                tool_calls_in_progress[tool_call_id] = {
                                    'name': tool_name,
                                    'args': '',
                                    'id': tool_call_id
                                }
                                yield StreamingEvent(
                                    event_type="tool_start",
                                    content=f"\n\n🔧 **Using tool: {tool_name}**",
                                    metadata={
                                        "tool_name": tool_name,
                                        "tool_call_id": tool_call_id
                                    }
                                )
                            
                            if tool_call_id in tool_calls_in_progress:
                                # Accumulate tool arguments
                                tool_calls_in_progress[tool_call_id]['args'] += tool_args
                
                # Process completed tool calls
                if hasattr(accumulated_chunk, 'tool_calls') and accumulated_chunk.tool_calls:
                    has_tool_calls = True
                    print(f"🔍 DEBUG: Found {len(accumulated_chunk.tool_calls)} tool calls")
                    
                    for tool_call in accumulated_chunk.tool_calls:
                        tool_name = tool_call.get('name')
                        tool_args = tool_call.get('args', {})
                        tool_call_id = tool_call.get('id')
                        
                        if tool_name and tool_call_id in tool_calls_in_progress:
                            # Check for duplicate tool call
                            from .tool_deduplicator import get_deduplicator
                            deduplicator = get_deduplicator()
                            is_duplicate, cached_result = deduplicator.is_duplicate(tool_name, tool_args, conversation_id)
                            
                            try:
                                if is_duplicate:
                                    # Use cached result for duplicate tool call
                                    tool_result = cached_result['result'] if cached_result else "Error: Cached result not found"
                                    duplicate_count = deduplicator.get_duplicate_count(tool_name, tool_args, conversation_id)
                                    
                                    # Stream tool completion with deduplication info
                                    yield StreamingEvent(
                                        event_type="tool_end",
                                        content=f"\n🔧 Using tool: {tool_name}\n✅ {tool_name} completed\nCalled {duplicate_count + 1} times (deduplicated)\n**Result:** {str(tool_result)}",
                                        metadata={
                                            "tool_name": tool_name,
                                            "tool_output": str(tool_result),
                                            "duplicate": True,
                                            "duplicate_count": duplicate_count + 1
                                        }
                                    )
                                else:
                                    # Find the tool in our tools list
                                    tool_obj = None
                                    for tool in agent.tools:
                                        if tool.name == tool_name:
                                            tool_obj = tool
                                            break
                                    
                                    if tool_obj:
                                        # Execute tool for first time
                                        tool_result = tool_obj.invoke(tool_args)
                                        
                                        # Store the tool call result for future deduplication
                                        deduplicator.store_tool_call(tool_name, tool_args, tool_result, conversation_id)
                                        
                                        # Stream tool completion with result in one event
                                        yield StreamingEvent(
                                            event_type="tool_end",
                                            content=f"\n🔧 Using tool: {tool_name}\n✅ {tool_name} completed\nCalled 1 time\n**Result:** {str(tool_result)}",
                                            metadata={
                                                "tool_name": tool_name,
                                                "tool_output": str(tool_result),
                                                "duplicate": False,
                                                "duplicate_count": 1
                                            }
                                        )
                                    else:
                                        yield StreamingEvent(
                                            event_type="error",
                                            content=f"❌ **Unknown tool: {tool_name}**",
                                            metadata={"tool_name": tool_name}
                                        )
                                        # Remove from in-progress
                                        del tool_calls_in_progress[tool_call_id]
                                        continue
                                
                                # Add tool message to conversation
                                tool_message = ToolMessage(
                                    content=str(tool_result),
                                    tool_call_id=tool_call_id
                                )
                                messages.append(tool_message)
                                agent.memory_manager.add_message(conversation_id, tool_message)
                                
                                # Remove from in-progress
                                del tool_calls_in_progress[tool_call_id]
                                    
                            except Exception as e:
                                yield StreamingEvent(
                                    event_type="error",
                                    content=f"❌ **Tool error: {str(e)}**",
                                    metadata={
                                        "tool_name": tool_name,
                                        "error": str(e)
                                    }
                                )
                                # Remove from in-progress
                                if tool_call_id in tool_calls_in_progress:
                                    del tool_calls_in_progress[tool_call_id]
                
                # Add AI message to conversation
                if accumulated_chunk and hasattr(accumulated_chunk, 'content') and accumulated_chunk.content:
                    ai_message = AIMessage(content=accumulated_chunk.content)
                    messages.append(ai_message)
                    agent.memory_manager.add_message(conversation_id, ai_message)
                
                # If no tool calls, we're done
                if not has_tool_calls:
                    print(f"🔍 DEBUG: No tool calls in iteration {iteration}, conversation complete")
                    break
                
                print(f"🔍 DEBUG: Completed iteration {iteration}, continuing...")
            
            # Check if we hit max iterations
            if iteration >= self.max_iterations:
                print(f"🔍 DEBUG: Reached max iterations ({self.max_iterations}), stopping conversation")
                yield StreamingEvent(
                    event_type="warning",
                    content=f"⚠️ **Reached maximum iterations ({self.max_iterations}), conversation may be incomplete**",
                    metadata={"max_iterations_reached": True}
                )
            
            # Use the last accumulated chunk for token tracking
            final_chunk = accumulated_chunk
            
            # Track API tokens
            if final_chunk and hasattr(agent, 'token_tracker'):
                try:
                    if hasattr(final_chunk, 'usage_metadata') and final_chunk.usage_metadata:
                        agent.token_tracker.track_llm_response(final_chunk, messages)
                except Exception as e:
                    print(f"🔍 DEBUG: Error tracking API tokens: {e}")
            
            # Final completion event
            yield StreamingEvent(
                event_type="completion",
                content="✅ **Response completed**",
                metadata={"final_response": True}
            )
                
        except Exception as e:
            yield StreamingEvent(
                event_type="error",
                content=f"❌ **Error: {str(e)}**",
                metadata={"error": str(e)}
            )


# Global streaming instance
_native_streaming_instance = None

def get_native_streaming() -> NativeLangChainStreaming:
    """Get the global native streaming instance"""
    global _native_streaming_instance
    if _native_streaming_instance is None:
        _native_streaming_instance = NativeLangChainStreaming()
    return _native_streaming_instance
