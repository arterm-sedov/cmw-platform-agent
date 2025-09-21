"""
Async LangChain Streaming Implementation
======================================

This module implements truly asynchronous streaming using LangChain's native
streaming capabilities with async tool execution and parallel processing.

Key Features:
- Async tool execution with asyncio
- Parallel tool processing when possible
- True real-time streaming with astream_events
- Non-blocking LLM calls
- Proper async/await patterns throughout

Based on: https://python.langchain.com/docs/concepts/streaming/
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import BaseTool


@dataclass
class AsyncStreamingEvent:
    """Represents an async streaming event with metadata"""
    event_type: str
    content: str
    timestamp: float
    metadata: Dict[str, Any]
    run_id: Optional[str] = None
    parent_ids: List[str] = None


class AsyncLangChainCallbackHandler(BaseCallbackHandler):
    """
    Async callback handler for LangChain streaming events.
    
    This handler captures all streaming events and provides them
    in a structured format for real-time display.
    """
    
    def __init__(self, event_handler: Callable[[AsyncStreamingEvent], None] = None):
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
        
        event = AsyncStreamingEvent(
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
            
            event = AsyncStreamingEvent(
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
            event = AsyncStreamingEvent(
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
            event = AsyncStreamingEvent(
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
        event = AsyncStreamingEvent(
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
        event = AsyncStreamingEvent(
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
        event = AsyncStreamingEvent(
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
        event = AsyncStreamingEvent(
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
    
    def _emit_event(self, event: AsyncStreamingEvent) -> None:
        """Emit an event to the handler and store it"""
        self.events.append(event)
        
        if self.event_handler:
            try:
                self.event_handler(event)
            except Exception as e:
                print(f"[AsyncLangChainCallbackHandler] Error in event handler: {e}")


class AsyncLangChainStreamingManager:
    """
    Manages async LangChain streaming with real-time event handling.
    
    This class implements truly asynchronous streaming patterns for
    real-time streaming of chat model responses with async tool calls.
    """
    
    def __init__(self):
        self.active_streams = {}
        self.event_handlers = []
        self.executor = ThreadPoolExecutor(max_workers=4)  # For CPU-bound tool execution
    
    def add_event_handler(self, handler: Callable[[AsyncStreamingEvent], None]) -> None:
        """Add an event handler for streaming events"""
        self.event_handlers.append(handler)
    
    def remove_event_handler(self, handler: Callable[[AsyncStreamingEvent], None]) -> None:
        """Remove an event handler"""
        if handler in self.event_handlers:
            self.event_handlers.remove(handler)
    
    async def stream_llm_response(
        self, 
        llm, 
        messages: List[BaseMessage], 
        tools: List[BaseTool] = None,
        config: Optional[RunnableConfig] = None
    ) -> AsyncGenerator[AsyncStreamingEvent, None]:
        """
        Stream LLM response using LangChain's native async streaming.
        
        Args:
            llm: LangChain LLM instance
            messages: List of messages
            tools: Optional list of tools
            config: Optional runnable config
            
        Yields:
            AsyncStreamingEvent objects
        """
        try:
            # Create callback handler
            events = []
            
            def event_handler(event: AsyncStreamingEvent):
                events.append(event)
            
            callback_handler = AsyncLangChainCallbackHandler(event_handler)
            
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
            error_event = AsyncStreamingEvent(
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
    ) -> AsyncGenerator[AsyncStreamingEvent, None]:
        """
        Stream agent response using truly async LangChain streaming.
        
        This method implements real-time streaming for the entire agent workflow
        including async tool calls and responses.
        
        Args:
            agent: The LangChain agent instance
            message: User message
            conversation_id: Conversation identifier
            
        Yields:
            AsyncStreamingEvent objects
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
            
            # Stream the async tool calling loop
            async for event in self._stream_async_tool_calling_loop(
                agent.llm_instance.llm,
                messages,
                agent.tools,
                conversation_id,
                agent.memory_manager
            ):
                yield event
                
        except Exception as e:
            error_event = AsyncStreamingEvent(
                event_type="error",
                content=f"Agent streaming error: {str(e)}",
                timestamp=time.time(),
                metadata={"error": str(e)}
            )
            yield error_event
    
    async def _stream_async_tool_calling_loop(
        self,
        llm,
        messages: List[BaseMessage],
        tools: List[BaseTool],
        conversation_id: str,
        memory_manager
    ) -> AsyncGenerator[AsyncStreamingEvent, None]:
        """
        Stream the tool calling loop using truly async patterns.
        
        This implements real-time streaming for the entire tool calling workflow
        with async tool execution and parallel processing.
        """
        tool_calls = []
        max_iterations = 10
        iteration = 0
        
        # Create tool registry
        tool_registry = {tool.name: tool for tool in tools}
        
        while iteration < max_iterations:
            iteration += 1
            
            # Stream LLM response using async streaming
            async for event in self.stream_llm_response(llm, messages, tools):
                # Filter and transform events for agent workflow
                if event.event_type == "llm_chunk":
                    # Stream content in real-time
                    yield AsyncStreamingEvent(
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
                # Use async LLM call if available, otherwise run in executor
                if hasattr(llm, 'ainvoke'):
                    response = await llm.ainvoke(messages)
                else:
                    # Run synchronous invoke in thread pool
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        self.executor, 
                        lambda: llm.invoke(messages)
                    )
            except Exception as e:
                yield AsyncStreamingEvent(
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
                
                # Process tool calls asynchronously
                await self._process_tool_calls_async(
                    response.tool_calls,
                    tool_registry,
                    tool_calls,
                    messages,
                    conversation_id,
                    memory_manager
                )
            else:
                # No tool calls, we have the final response
                final_response = response.content if hasattr(response, 'content') else str(response)
                
                if final_response and final_response.strip():
                    # Stream the final response
                    yield AsyncStreamingEvent(
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
        yield AsyncStreamingEvent(
            event_type="completion",
            content="✅ **Response completed**",
            timestamp=time.time(),
            metadata={
                "tool_calls": tool_calls,
                "iterations": iteration
            }
        )
    
    async def _process_tool_calls_async(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_registry: Dict[str, BaseTool],
        tool_calls_list: List[Dict[str, Any]],
        messages: List[BaseMessage],
        conversation_id: str,
        memory_manager
    ) -> None:
        """
        Process tool calls asynchronously with parallel execution when possible.
        """
        # Group tool calls by dependency
        independent_tools = []
        dependent_tools = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get('name', 'unknown')
            # For now, treat all tools as independent for parallel execution
            # In the future, we could analyze dependencies
            independent_tools.append(tool_call)
        
        # Execute independent tools in parallel
        if independent_tools:
            tasks = []
            for tool_call in independent_tools:
                task = self._execute_tool_async(
                    tool_call, tool_registry, tool_calls_list, 
                    messages, conversation_id, memory_manager
                )
                tasks.append(task)
            
            # Wait for all tools to complete
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_tool_async(
        self,
        tool_call: Dict[str, Any],
        tool_registry: Dict[str, BaseTool],
        tool_calls_list: List[Dict[str, Any]],
        messages: List[BaseMessage],
        conversation_id: str,
        memory_manager
    ) -> None:
        """
        Execute a single tool asynchronously.
        """
        tool_name = tool_call.get('name', 'unknown')
        tool_args = tool_call.get('args', {})
        tool_call_id = tool_call.get('id', f"call_{len(tool_calls_list)}")
        
        if tool_name in tool_registry:
            # Stream tool start
            yield AsyncStreamingEvent(
                event_type="tool_start",
                content=f"🔧 **Using tool: {tool_name}**",
                timestamp=time.time(),
                metadata={
                    "tool_name": tool_name,
                    "tool_args": tool_args
                }
            )
            
            try:
                # Execute tool in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                tool_result = await loop.run_in_executor(
                    self.executor,
                    lambda: tool_registry[tool_name].invoke(tool_args)
                )
                
                # Stream tool end
                yield AsyncStreamingEvent(
                    event_type="tool_end",
                    content=f"✅ **{tool_name} completed**",
                    timestamp=time.time(),
                    metadata={
                        "tool_name": tool_name,
                        "tool_output": str(tool_result)
                    }
                )
                
                # Store tool call
                tool_calls_list.append({
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
                yield AsyncStreamingEvent(
                    event_type="error",
                    content=f"❌ **Tool error: {str(e)}**",
                    timestamp=time.time(),
                    metadata={
                        "tool_name": tool_name,
                        "error": str(e)
                    }
                )
        else:
            yield AsyncStreamingEvent(
                event_type="error",
                content=f"❌ **Unknown tool: {tool_name}**",
                timestamp=time.time(),
                metadata={"tool_name": tool_name}
            )
    
    def _convert_langchain_event(self, event: Dict[str, Any]) -> Optional[AsyncStreamingEvent]:
        """Convert LangChain event to our AsyncStreamingEvent format"""
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
            
            return AsyncStreamingEvent(
                event_type=event_type,
                content=content,
                timestamp=time.time(),
                metadata=metadata,
                run_id=run_id,
                parent_ids=parent_ids
            )
            
        except Exception as e:
            print(f"[AsyncLangChainStreamingManager] Error converting event: {e}")
            return None
    
    def get_active_streams(self) -> List[str]:
        """Get list of active stream IDs"""
        return list(self.active_streams.keys())
    
    def close_stream(self, stream_id: str) -> None:
        """Close a specific stream"""
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
    
    def __del__(self):
        """Cleanup executor on destruction"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Global async streaming manager instance
_async_streaming_manager = None

def get_async_streaming_manager() -> AsyncLangChainStreamingManager:
    """Get the global async streaming manager instance"""
    global _async_streaming_manager
    if _async_streaming_manager is None:
        _async_streaming_manager = AsyncLangChainStreamingManager()
    return _async_streaming_manager

def reset_async_streaming_manager() -> None:
    """Reset the global async streaming manager instance"""
    global _async_streaming_manager
    if _async_streaming_manager is not None:
        _async_streaming_manager.executor.shutdown(wait=False)
        _async_streaming_manager = None
