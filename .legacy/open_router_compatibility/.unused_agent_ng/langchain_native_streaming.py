"""
LangChain Native Streaming Implementation
========================================

This module implements proper LangChain streaming using the official patterns
from the LangChain documentation for real-time streaming of chat model responses.

Key Features:
- Native LangChain streaming with astream_events
- Real-time tool execution streaming
- Proper callback propagation
- Integration with existing agent architecture
- No artificial delays or fake streaming

Based on: https://python.langchain.com/docs/concepts/streaming/
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from dataclasses import dataclass

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import BaseTool


@dataclass
class LangChainStreamingEvent:
    """Represents a streaming event with metadata"""
    event_type: str
    content: str
    timestamp: float
    metadata: Dict[str, Any]
    run_id: Optional[str] = None
    parent_ids: List[str] = None


class LangChainNativeCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler for LangChain native streaming events.
    
    This handler captures all streaming events and provides them
    in a structured format for real-time display.
    """
    
    def __init__(self, event_handler: Callable[[LangChainStreamingEvent], None] = None):
        self.event_handler = event_handler
        self.events = []
        self.current_run_id = None
        self.current_tool_name = None
        self.accumulated_content = ""
        self.tool_calls = []
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts generating"""
        self.current_run_id = kwargs.get("run_id")
        self.accumulated_content = ""
        
        event = LangChainStreamingEvent(
            event_type="llm_start",
            content="🤖 **Thinking...**",
            timestamp=time.time(),
            metadata={
                "llm_type": serialized.get("name", "unknown"),
                "prompts_count": len(prompts)
            },
            run_id=self.current_run_id
        )
        
        self._emit_event(event)
    
    def on_llm_stream(self, chunk: Any, **kwargs) -> None:
        """Called for each streaming chunk from LLM"""
        content = getattr(chunk, 'content', '') or str(chunk)
        if content:
            self.accumulated_content += content
            
            event = LangChainStreamingEvent(
                event_type="llm_chunk",
                content=content,
                timestamp=time.time(),
                metadata={
                    "chunk_type": "llm_response",
                    "accumulated_length": len(self.accumulated_content)
                },
                run_id=self.current_run_id
            )
            
            self._emit_event(event)
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool starts executing"""
        self.current_tool_name = serialized.get("name", "unknown_tool")
        
        # Filter out schema tools from display
        schema_tools = {'submit_answer', 'submit_intermediate_step'}
        if self.current_tool_name not in schema_tools:
            event = LangChainStreamingEvent(
                event_type="tool_start",
                content=f"🔧 **Using tool: {self.current_tool_name}**",
                timestamp=time.time(),
                metadata={
                    "tool_name": self.current_tool_name,
                    "tool_args": input_str
                },
                run_id=self.current_run_id
            )
            
            self._emit_event(event)
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes executing"""
        # Filter out schema tools from display
        schema_tools = {'submit_answer', 'submit_intermediate_step'}
        if self.current_tool_name not in schema_tools:
            # Truncate long outputs for display
            display_output = output[:500] + "..." if len(output) > 500 else output
            
            event = LangChainStreamingEvent(
                event_type="tool_end",
                content=f"✅ **{self.current_tool_name} completed**",
                timestamp=time.time(),
                metadata={
                    "tool_name": self.current_tool_name,
                    "tool_output": output
                },
                run_id=self.current_run_id
            )
            
            self._emit_event(event)
    
    def on_llm_end(self, response: Any, **kwargs) -> None:
        """Called when LLM finishes generating"""
        event = LangChainStreamingEvent(
            event_type="llm_end",
            content="🎯 **Response completed**",
            timestamp=time.time(),
            metadata={
                "response": str(response),
                "total_content": self.accumulated_content
            },
            run_id=self.current_run_id
        )
        
        self._emit_event(event)
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain starts"""
        event = LangChainStreamingEvent(
            event_type="chain_start",
            content="🔄 **Processing request...**",
            timestamp=time.time(),
            metadata={
                "chain_name": serialized.get("name", "unknown"),
                "inputs": inputs
            },
            run_id=kwargs.get("run_id")
        )
        
        self._emit_event(event)
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain ends"""
        event = LangChainStreamingEvent(
            event_type="chain_end",
            content="✅ **Request completed**",
            timestamp=time.time(),
            metadata={
                "outputs": outputs
            },
            run_id=kwargs.get("run_id")
        )
        
        self._emit_event(event)
    
    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Called when a chain encounters an error"""
        event = LangChainStreamingEvent(
            event_type="chain_error",
            content=f"❌ **Error: {str(error)}**",
            timestamp=time.time(),
            metadata={
                "error": str(error),
                "error_type": type(error).__name__
            },
            run_id=kwargs.get("run_id")
        )
        
        self._emit_event(event)
    
    def _emit_event(self, event: LangChainStreamingEvent) -> None:
        """Emit an event to the handler and store it"""
        self.events.append(event)
        
        if self.event_handler:
            try:
                self.event_handler(event)
            except Exception as e:
                print(f"[LangChainNativeCallbackHandler] Error in event handler: {e}")


class LangChainNativeStreamingManager:
    """
    Manages LangChain native streaming with real-time event handling.
    
    This class implements the streaming patterns from the LangChain documentation
    for real-time streaming of chat model responses with tool calls.
    """
    
    def __init__(self):
        self.active_streams = {}
        self.event_handlers = []
    
    def add_event_handler(self, handler: Callable[[LangChainStreamingEvent], None]) -> None:
        """Add an event handler for streaming events"""
        self.event_handlers.append(handler)
    
    def remove_event_handler(self, handler: Callable[[LangChainStreamingEvent], None]) -> None:
        """Remove an event handler"""
        if handler in self.event_handlers:
            self.event_handlers.remove(handler)
    
    async def stream_llm_response(
        self, 
        llm, 
        messages: List[BaseMessage], 
        tools: List[BaseTool] = None,
        config: Optional[RunnableConfig] = None
    ) -> AsyncGenerator[LangChainStreamingEvent, None]:
        """
        Stream LLM response using LangChain's native streaming.
        
        Args:
            llm: LangChain LLM instance
            messages: List of messages
            tools: Optional list of tools
            config: Optional runnable config
            
        Yields:
            LangChainStreamingEvent objects
        """
        try:
            # Create callback handler
            events = []
            
            def event_handler(event: LangChainStreamingEvent):
                events.append(event)
            
            callback_handler = LangChainNativeCallbackHandler(event_handler)
            
            # Prepare LLM with tools if provided
            if tools:
                llm_with_tools = llm.bind_tools(tools)
            else:
                llm_with_tools = llm
            
            # Create runnable config with callbacks
            if config is None:
                config = RunnableConfig(callbacks=[callback_handler])
            else:
                if config.get("callbacks"):
                    config["callbacks"].append(callback_handler)
                else:
                    config["callbacks"] = [callback_handler]
            
            # Stream the response using astream_events
            async for event in llm_with_tools.astream_events(
                messages, 
                version="v1",
                config=config
            ):
                # Convert LangChain event to our format
                streaming_event = self._convert_langchain_event(event)
                if streaming_event:
                    yield streaming_event
            
            # Yield any events from the callback handler
            for event in events:
                yield event
                
        except Exception as e:
            error_event = LangChainStreamingEvent(
                event_type="error",
                content=f"Streaming error: {str(e)}",
                timestamp=time.time(),
                metadata={"error": str(e)}
            )
            yield error_event
    
    async def stream_agent_response(
        self,
        agent,
        message: str,
        conversation_id: str = "default"
    ) -> AsyncGenerator[LangChainStreamingEvent, None]:
        """
        Stream agent response using LangChain's native streaming.
        
        This method implements real-time streaming for the entire agent workflow
        including tool calls and responses.
        
        Args:
            agent: The LangChain agent instance
            message: User message
            conversation_id: Conversation identifier
            
        Yields:
            LangChainStreamingEvent objects
        """
        try:
            # Get conversation chain
            chain = agent._get_conversation_chain(conversation_id)
            
            # Get conversation history
            chat_history = agent.memory_manager.get_conversation_history(conversation_id)
            
            # Create messages list
            messages = [SystemMessage(content=agent.system_prompt)]
            messages.extend(chat_history)
            messages.append(HumanMessage(content=message))
            
            # Stream the tool calling loop using astream_events
            async for event in self._stream_tool_calling_loop(
                agent.llm_instance.llm,
                messages,
                agent.tools,
                conversation_id,
                agent.memory_manager
            ):
                yield event
                
        except Exception as e:
            error_event = LangChainStreamingEvent(
                event_type="error",
                content=f"Agent streaming error: {str(e)}",
                timestamp=time.time(),
                metadata={"error": str(e)}
            )
            yield error_event
    
    async def _stream_tool_calling_loop(
        self,
        llm,
        messages: List[BaseMessage],
        tools: List[BaseTool],
        conversation_id: str,
        memory_manager
    ) -> AsyncGenerator[LangChainStreamingEvent, None]:
        """
        Stream the tool calling loop using LangChain's native streaming.
        
        This implements real-time streaming for the entire tool calling workflow.
        """
        tool_calls = []
        max_iterations = 10
        iteration = 0
        
        # Create tool registry
        tool_registry = {tool.name: tool for tool in tools}
        
        while iteration < max_iterations:
            iteration += 1
            
            # Stream LLM response
            async for event in self.stream_llm_response(llm, messages, tools):
                # Filter and transform events for agent workflow
                if event.event_type == "llm_chunk":
                    # Stream content in real-time
                    yield LangChainStreamingEvent(
                        event_type="content",
                        content=event.content,
                        timestamp=event.timestamp,
                        metadata=event.metadata,
                        run_id=event.run_id
                    )
                elif event.event_type == "tool_start":
                    yield event
                elif event.event_type == "tool_end":
                    yield event
                elif event.event_type == "llm_end":
                    # Check if we need to continue with tool calls
                    break
            
            # Get the final response to check for tool calls
            try:
                response = llm.invoke(messages)
            except Exception as e:
                yield LangChainStreamingEvent(
                    event_type="error",
                    content=f"LLM Error: {str(e)}",
                    timestamp=time.time(),
                    metadata={"error": str(e)}
                )
                return
            
            # Check for tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Add the AI response with tool calls to messages
                messages.append(response)
                
                # Process tool calls
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get('name', 'unknown')
                    tool_args = tool_call.get('args', {})
                    tool_call_id = tool_call.get('id', f"call_{len(tool_calls)}")
                    
                    # Stream tool execution
                    if tool_name in tool_registry:
                        yield LangChainStreamingEvent(
                            event_type="tool_start",
                            content=f"🔧 **Using tool: {tool_name}**",
                            timestamp=time.time(),
                            metadata={
                                "tool_name": tool_name,
                                "tool_args": tool_args
                            }
                        )
                        
                        # Execute tool
                        try:
                            tool_result = tool_registry[tool_name].invoke(tool_args)
                            
                            yield LangChainStreamingEvent(
                                event_type="tool_end",
                                content=f"✅ **{tool_name} completed**",
                                timestamp=time.time(),
                                metadata={
                                    "tool_name": tool_name,
                                    "tool_output": str(tool_result)
                                }
                            )
                            
                            # Store tool call
                            tool_calls.append({
                                'name': tool_name,
                                'args': tool_args,
                                'result': tool_result,
                                'id': tool_call_id
                            })
                            
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
                            
                            # Add tool message to memory manager
                            memory_manager.add_message(conversation_id, tool_message)
                            
                        except Exception as e:
                            yield LangChainStreamingEvent(
                                event_type="error",
                                content=f"❌ **Tool error: {str(e)}**",
                                timestamp=time.time(),
                                metadata={
                                    "tool_name": tool_name,
                                    "error": str(e)
                                }
                            )
                    else:
                        yield LangChainStreamingEvent(
                            event_type="error",
                            content=f"❌ **Unknown tool: {tool_name}**",
                            timestamp=time.time(),
                            metadata={"tool_name": tool_name}
                        )
            else:
                # No tool calls, we have the final response
                final_response = response.content if hasattr(response, 'content') else str(response)
                
                if final_response and final_response.strip():
                    # Stream the final response
                    yield LangChainStreamingEvent(
                        event_type="content",
                        content=final_response,
                        timestamp=time.time(),
                        metadata={
                            "final_response": True,
                            "tool_calls": tool_calls
                        }
                    )
                    
                    # Add AI response to messages
                    messages.append(response)
                    break
                else:
                    # Empty response, retry with reminder
                    reminder_msg = HumanMessage(content="Please provide a meaningful response. You should answer the user's question or use available tools to help.")
                    messages.append(reminder_msg)
                    continue
        
        # Final completion event
        yield LangChainStreamingEvent(
            event_type="completion",
            content="✅ **Response completed**",
            timestamp=time.time(),
            metadata={
                "tool_calls": tool_calls,
                "iterations": iteration
            }
        )
    
    def _convert_langchain_event(self, event: Dict[str, Any]) -> Optional[LangChainStreamingEvent]:
        """Convert LangChain event to our LangChainStreamingEvent format"""
        try:
            event_type = event.get("event", "unknown")
            data = event.get("data", {})
            name = event.get("name", "unknown")
            run_id = event.get("run_id")
            parent_ids = event.get("parent_ids", [])
            
            # Extract content based on event type
            content = ""
            metadata = {
                "name": name,
                "run_id": run_id,
                "parent_ids": parent_ids
            }
            
            if event_type == "on_llm_start":
                content = "🤖 **Thinking...**"
                metadata.update({
                    "llm_type": data.get("name", "unknown"),
                    "prompts": data.get("prompts", [])
                })
            elif event_type == "on_llm_stream":
                chunk = data.get("chunk", {})
                content = getattr(chunk, 'content', '') or str(chunk)
                metadata.update({
                    "chunk_type": "llm_response"
                })
            elif event_type == "on_tool_start":
                tool_name = data.get("name", "unknown_tool")
                content = f"🔧 **Using tool: {tool_name}**"
                metadata.update({
                    "tool_name": tool_name,
                    "tool_args": data.get("input", "")
                })
            elif event_type == "on_tool_end":
                tool_name = metadata.get("tool_name", "unknown_tool")
                output = data.get("output", "")
                content = f"✅ **{tool_name} completed**"
                metadata.update({
                    "tool_output": output
                })
            elif event_type == "on_llm_end":
                content = "🎯 **Response completed**"
                metadata.update({
                    "response": str(data.get("output", ""))
                })
            elif event_type == "on_chain_start":
                content = "🔄 **Processing request...**"
                metadata.update({
                    "chain_name": name,
                    "inputs": data
                })
            elif event_type == "on_chain_end":
                content = "✅ **Request completed**"
                metadata.update({
                    "outputs": data
                })
            elif event_type == "on_chain_error":
                error = data.get("error", "Unknown error")
                content = f"❌ **Error: {str(error)}**"
                metadata.update({
                    "error": str(error),
                    "error_type": type(error).__name__
                })
            else:
                # Skip unknown event types
                return None
            
            return LangChainStreamingEvent(
                event_type=event_type,
                content=content,
                timestamp=time.time(),
                metadata=metadata,
                run_id=run_id,
                parent_ids=parent_ids
            )
            
        except Exception as e:
            print(f"[LangChainNativeStreamingManager] Error converting event: {e}")
            return None
    
    def get_active_streams(self) -> List[str]:
        """Get list of active stream IDs"""
        return list(self.active_streams.keys())
    
    def close_stream(self, stream_id: str) -> None:
        """Close a specific stream"""
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]


# Global streaming manager instance
_native_streaming_manager = None

def get_native_streaming_manager() -> LangChainNativeStreamingManager:
    """Get the global native streaming manager instance"""
    global _native_streaming_manager
    if _native_streaming_manager is None:
        _native_streaming_manager = LangChainNativeStreamingManager()
    return _native_streaming_manager

def reset_native_streaming_manager() -> None:
    """Reset the global native streaming manager instance"""
    global _native_streaming_manager
    _native_streaming_manager = None
