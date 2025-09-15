"""
Hybrid Streaming Implementation
==============================

This module implements a hybrid approach where tool calling is handled
non-streaming for reliability, but the final response is streamed.

Key Features:
- Reliable tool calling using invoke()
- Streaming final response using astream()
- Proper conversation memory management
- Tool result display in chat
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


class HybridStreaming:
    """
    Hybrid streaming implementation that handles tool calling reliably
    and streams the final response.
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
        Stream agent response using hybrid approach.
        
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
            
            # Tool calling loop (non-streaming for reliability)
            max_iterations = 10
            iteration = 0
            last_response = None
            
            while iteration < max_iterations:
                iteration += 1
                
                # Get LLM response
                try:
                    response = await llm_with_tools.ainvoke(messages)
                except Exception as e:
                    yield StreamingEvent(
                        event_type="error",
                        content=f"❌ **LLM Error: {str(e)}**",
                        metadata={"error": str(e)}
                    )
                    return
                
                # Add response to messages
                messages.append(response)
                agent.memory_manager.add_message(conversation_id, response)
                
                # Check for tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    # Stream the response content if any
                    if hasattr(response, 'content') and response.content:
                        yield StreamingEvent(
                            event_type="content",
                            content=response.content,
                            metadata={"chunk_type": "llm_response"}
                        )
                    
                    # Process tool calls
                    async for event in self._process_tool_calls(
                        response.tool_calls, agent.tools, messages, 
                        conversation_id, agent.memory_manager
                    ):
                        yield event
                    
                    # Continue the loop to let LLM process tool results
                    continue
                else:
                    # No tool calls, we have the final response
                    print(f"🔍 DEBUG: Final response - has content: {hasattr(response, 'content')}, content: {getattr(response, 'content', 'NO_CONTENT')[:100]}...")
                    if hasattr(response, 'content') and response.content:
                        # Stream the final response
                        async for event in self._stream_final_response(response, llm_with_tools, messages):
                            yield event
                        last_response = response
                        break
                    else:
                        # Empty response, retry
                        print(f"🔍 DEBUG: Empty response, retrying...")
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
    
    async def _stream_final_response(self, response, llm_with_tools, messages):
        """Stream the final response content"""
        print(f"🔍 DEBUG: _stream_final_response called with content: {getattr(response, 'content', 'NO_CONTENT')[:100]}...")
        if hasattr(response, 'content') and response.content:
            # Stream the content word by word for better user experience
            content = response.content
            words = content.split()
            print(f"🔍 DEBUG: Streaming {len(words)} words")
            
            if len(words) <= 1:
                # Single word or short content, stream as-is
                print(f"🔍 DEBUG: Streaming single word/short content")
                yield StreamingEvent(
                    event_type="content",
                    content=content,
                    metadata={"chunk_type": "final_response"}
                )
            else:
                # Stream word by word with small delays
                print(f"🔍 DEBUG: Starting word-by-word streaming")
                for i, word in enumerate(words):
                    # Add space before word (except first word)
                    if i > 0:
                        word = " " + word
                    
                    print(f"🔍 DEBUG: Yielding word {i}: '{word}'")
                    yield StreamingEvent(
                        event_type="content",
                        content=word,
                        metadata={"chunk_type": "final_response", "word_index": i}
                    )
                    
                    # Small delay between words for streaming effect
                    await asyncio.sleep(0.05)
        else:
            print(f"🔍 DEBUG: No content to stream in final response")
    
    async def _process_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        tools: List[BaseTool],
        messages: List[BaseMessage],
        conversation_id: str,
        memory_manager
    ) -> AsyncGenerator[StreamingEvent, None]:
        """
        Process tool calls with proper result display.
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
_hybrid_streaming_instance = None

def get_hybrid_streaming() -> HybridStreaming:
    """Get the global hybrid streaming instance"""
    global _hybrid_streaming_instance
    if _hybrid_streaming_instance is None:
        _hybrid_streaming_instance = HybridStreaming()
    return _hybrid_streaming_instance
