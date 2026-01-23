"""
Natural LangChain Streaming Implementation
=========================================

This module implements truly natural streaming using LangChain's native streaming
capabilities with NO artificial delays. All timing comes naturally from the LLM
and tool execution.

Key Features:
- Zero artificial delays - only natural timing
- True real-time streaming with astream_events
- Async tool execution without blocking
- Natural flow from LLM generation
- No character-by-character fake streaming

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
class NaturalStreamingEvent:
    """Represents a natural streaming event with metadata"""
    event_type: str
    content: str
    timestamp: float
    metadata: Dict[str, Any]
    run_id: Optional[str] = None
    parent_ids: List[str] = None


class NaturalLangChainCallbackHandler(BaseCallbackHandler):
    """
    Natural callback handler for LangChain streaming events.

    This handler captures all streaming events with natural timing
    and provides them in real-time without any artificial delays.
    """

    def __init__(self, event_handler: Callable[[NaturalStreamingEvent], None] = None):
        self.event_handler = event_handler
        self.events = []
        self.current_run_id = None
        self.current_tool_name = None
        self.accumulated_content = ""

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts generating"""
        self.current_run_id = kwargs.get("run_id")
        self.accumulated_content = ""

        event = NaturalStreamingEvent(
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
        """Called for each streaming chunk from LLM - NATURAL TIMING"""
        content = getattr(chunk, 'content', '') or str(chunk)
        if content:
            self.accumulated_content += content

            event = NaturalStreamingEvent(
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
            event = NaturalStreamingEvent(
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
            event = NaturalStreamingEvent(
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
        event = NaturalStreamingEvent(
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
        event = NaturalStreamingEvent(
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
        event = NaturalStreamingEvent(
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
        event = NaturalStreamingEvent(
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

    def _emit_event(self, event: NaturalStreamingEvent) -> None:
        """Emit an event to the handler and store it - NO DELAYS"""
        self.events.append(event)

        if self.event_handler:
            try:
                self.event_handler(event)
            except Exception as e:
                print(f"[NaturalLangChainCallbackHandler] Error in event handler: {e}")


class NaturalLangChainStreamingManager:
    """
    Manages natural LangChain streaming with zero artificial delays.

    This class implements truly natural streaming patterns where all timing
    comes from the LLM and tool execution, not artificial delays.
    """

    def __init__(self):
        self.active_streams = {}
        self.event_handlers = []
        self.executor = ThreadPoolExecutor(max_workers=4)  # For CPU-bound tool execution

    def add_event_handler(self, handler: Callable[[NaturalStreamingEvent], None]) -> None:
        """Add an event handler for streaming events"""
        self.event_handlers.append(handler)

    def remove_event_handler(self, handler: Callable[[NaturalStreamingEvent], None]) -> None:
        """Remove an event handler"""
        if handler in self.event_handlers:
            self.event_handlers.remove(handler)

    async def stream_llm_response(
        self, 
        llm, 
        messages: List[BaseMessage], 
        tools: List[BaseTool] = None,
        config: Optional[RunnableConfig] = None
    ) -> AsyncGenerator[NaturalStreamingEvent, None]:
        """
        Stream LLM response using LangChain's native streaming - NATURAL TIMING ONLY.

        Args:
            llm: LangChain LLM instance
            messages: List of messages
            tools: Optional list of tools
            config: Optional runnable config

        Yields:
            NaturalStreamingEvent objects with natural timing
        """
        try:
            # Create callback handler
            events = []

            def event_handler(event: NaturalStreamingEvent):
                events.append(event)

            callback_handler = NaturalLangChainCallbackHandler(event_handler)

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

            # Stream the response using astream_events - NATURAL TIMING
            async for event in llm_with_tools.astream_events(
                messages, 
                version="v1",
                config=config
            ):
                # Convert LangChain event to our format - NO DELAYS
                streaming_event = self._convert_langchain_event(event)
                if streaming_event:
                    yield streaming_event

            # Yield any events from the callback handler - NO DELAYS
            for event in events:
                yield event

        except Exception as e:
            error_event = NaturalStreamingEvent(
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
    ) -> AsyncGenerator[NaturalStreamingEvent, None]:
        """
        Stream agent response using truly natural LangChain streaming.

        This method implements real-time streaming for the entire agent workflow
        with ZERO artificial delays - only natural timing from LLM and tools.

        Args:
            agent: The LangChain agent instance
            message: User message
            conversation_id: Conversation identifier

        Yields:
            NaturalStreamingEvent objects with natural timing
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

            # Track API tokens for this conversation
            last_response = None

            # Stream the natural tool calling loop - NO ARTIFICIAL DELAYS
            async for event in self._stream_natural_tool_calling_loop(
                agent.llm_instance.llm,
                messages,
                agent.tools,
                conversation_id,
                agent.memory_manager
            ):
                # Track the last response for API token counting
                if event.event_type == "completion":
                    last_response = event.metadata.get("final_response")

                yield event

            # Track API tokens after streaming is complete
            if last_response and hasattr(agent, 'token_tracker'):
                try:
                    # For chunks, we need to get the final response with usage metadata
                    if hasattr(last_response, 'usage_metadata') and last_response.usage_metadata:
                        agent.token_tracker.track_llm_response(last_response, messages)
                    else:
                        # If no usage metadata in chunk, try to get it from the final response
                        print(f"🔍 DEBUG: No usage metadata in chunk, trying to get final response")
                        if hasattr(llm_with_tools, 'ainvoke'):
                            final_response = await llm_with_tools.ainvoke(messages)
                            if hasattr(final_response, 'usage_metadata') and final_response.usage_metadata:
                                agent.token_tracker.track_llm_response(final_response, messages)
                except Exception as e:
                    print(f"🔍 DEBUG: Error tracking API tokens: {e}")

        except Exception as e:
            error_event = NaturalStreamingEvent(
                event_type="error",
                content=f"Agent streaming error: {str(e)}",
                timestamp=time.time(),
                metadata={"error": str(e)}
            )
            yield error_event

    async def _stream_natural_tool_calling_loop(
        self,
        llm,
        messages: List[BaseMessage],
        tools: List[BaseTool],
        conversation_id: str,
        memory_manager
    ) -> AsyncGenerator[NaturalStreamingEvent, None]:
        """
        Stream the tool calling loop using natural timing only.

        This implements real-time streaming for the entire tool calling workflow
        with ZERO artificial delays - only natural timing from LLM and tools.
        """
        tool_calls = []
        max_iterations = 10
        iteration = 0

        # Create tool registry
        tool_registry = {tool.name: tool for tool in tools}

        while iteration < max_iterations:
            iteration += 1

            # Get LLM response with tools bound
            llm_with_tools = llm.bind_tools(tools) if tools else llm

            try:
                # Use astream_events for proper tool calling support
                response_content = ""
                response_obj = None
                tool_calls_detected = False

                # Use astream_events for proper tool calling support
                try:
                    # Use astream_events to properly handle tool calls
                    async for event in llm_with_tools.astream_events(messages, version="v1"):
                        event_type = event.get("event", "")
                        data = event.get("data", {})

                        if event_type == "on_llm_stream":
                            chunk = data.get("chunk", {})
                            if hasattr(chunk, 'content') and chunk.content:
                                response_content += chunk.content
                                # Stream content chunk-by-chunk - REAL-TIME STREAMING
                                yield NaturalStreamingEvent(
                                    event_type="content",
                                    content=chunk.content,
                                    timestamp=time.time(),
                                    metadata={
                                        "chunk_type": "llm_response",
                                        "accumulated_length": len(response_content)
                                    }
                                )
                            response_obj = chunk

                        elif event_type == "on_llm_end":
                            # Get the final response object
                            response_obj = data.get("output")
                            if response_obj and hasattr(response_obj, 'tool_calls') and response_obj.tool_calls:
                                tool_calls_detected = True

                        elif event_type == "on_tool_start":
                            # Stream tool start event
                            tool_name = data.get("name", "unknown_tool")
                            yield NaturalStreamingEvent(
                                event_type="tool_start",
                                content=f"🔧 **Using tool: {tool_name}**",
                                timestamp=time.time(),
                                metadata={
                                    "tool_name": tool_name,
                                    "tool_args": data.get("input", "")
                                }
                            )

                        elif event_type == "on_tool_end":
                            # Stream tool end event
                            tool_name = data.get("name", "unknown_tool")
                            output = data.get("output", "")
                            yield NaturalStreamingEvent(
                                event_type="tool_end",
                                content=f"✅ **{tool_name} completed**",
                                timestamp=time.time(),
                                metadata={
                                    "tool_name": tool_name,
                                    "tool_output": str(output)
                                }
                            )

                except Exception as stream_error:
                    print(f"🔍 DEBUG: astream failed, falling back to astream_events: {stream_error}")

                    # Fallback to astream_events if astream fails
                    async for event in llm_with_tools.astream_events(messages, version="v1"):
                        event_type = event.get("event", "")
                        data = event.get("data", {})

                        if event_type == "on_llm_stream":
                            chunk = data.get("chunk", {})
                            if hasattr(chunk, 'content') and chunk.content:
                                response_content += chunk.content
                                # Stream content in real-time - NATURAL TIMING
                                yield NaturalStreamingEvent(
                                    event_type="content",
                                    content=chunk.content,
                                    timestamp=time.time(),
                                    metadata={
                                        "chunk_type": "llm_response",
                                        "accumulated_length": len(response_content)
                                    }
                                )
                            response_obj = chunk

                        elif event_type == "on_llm_end":
                            # Get the final response object
                            response_obj = data.get("output")
                            if response_obj and hasattr(response_obj, 'tool_calls') and response_obj.tool_calls:
                                tool_calls_detected = True

                        elif event_type == "on_tool_start":
                            # Stream tool start event
                            tool_name = data.get("name", "unknown_tool")
                            yield NaturalStreamingEvent(
                                event_type="tool_start",
                                content=f"🔧 **Using tool: {tool_name}**",
                                timestamp=time.time(),
                                metadata={
                                    "tool_name": tool_name,
                                    "tool_args": data.get("input", "")
                                }
                            )

                        elif event_type == "on_tool_end":
                            # Stream tool end event
                            tool_name = data.get("name", "unknown_tool")
                            output = data.get("output", "")
                            yield NaturalStreamingEvent(
                                event_type="tool_end",
                                content=f"✅ **{tool_name} completed**",
                                timestamp=time.time(),
                                metadata={
                                    "tool_name": tool_name,
                                    "tool_output": str(output)
                                }
                            )

                # If no streaming response, get the full response
                if not response_obj:
                    if hasattr(llm_with_tools, 'ainvoke'):
                        response_obj = await llm_with_tools.ainvoke(messages)
                    else:
                        loop = asyncio.get_event_loop()
                        response_obj = await loop.run_in_executor(
                            self.executor, 
                            lambda: llm_with_tools.invoke(messages)
                        )

                # Check for tool calls
                if hasattr(response_obj, 'tool_calls') and response_obj.tool_calls:
                    tool_calls_detected = True

            except Exception as e:
                yield NaturalStreamingEvent(
                    event_type="error",
                    content=f"LLM Error: {str(e)}",
                    timestamp=time.time(),
                    metadata={"error": str(e)}
                )
                return

            # Process tool calls if detected
            if tool_calls_detected and hasattr(response_obj, 'tool_calls') and response_obj.tool_calls:
                # Stream the LLM's content first if it exists
                if hasattr(response_obj, 'content') and response_obj.content and response_obj.content.strip():
                    yield NaturalStreamingEvent(
                        event_type="content",
                        content=response_obj.content,
                        timestamp=time.time(),
                        metadata={
                            "chunk_type": "llm_response",
                            "accumulated_length": len(response_obj.content)
                        }
                    )

                # Add the AI response with tool calls to messages
                messages.append(response_obj)

                # Process tool calls naturally - NO ARTIFICIAL DELAYS
                async for event in self._process_tool_calls_naturally(
                    response_obj.tool_calls,
                    tool_registry,
                    tool_calls,
                    messages,
                    conversation_id,
                    memory_manager
                ):
                    yield event

                # Continue the loop to let LLM process tool results and generate final response
                continue
            else:
                # No tool calls, we have the final response
                final_response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)

                if final_response and final_response.strip():
                    # Stream the final response if it wasn't already streamed
                    if not response_content:  # Only stream if we didn't already stream it
                        yield NaturalStreamingEvent(
                            event_type="content",
                            content=final_response,
                            timestamp=time.time(),
                            metadata={
                                "chunk_type": "llm_response",
                                "accumulated_length": len(final_response)
                            }
                        )

                    # Add AI response to messages
                    messages.append(response_obj)
                    break
                else:
                    # Empty response, retry with reminder
                    reminder_msg = HumanMessage(content="Please provide a meaningful response. You should answer the user's question or use available tools to help.")
                    messages.append(reminder_msg)
                    continue

        # Final completion event - NATURAL TIMING
        yield NaturalStreamingEvent(
            event_type="completion",
            content="✅ **Response completed**",
            timestamp=time.time(),
            metadata={
                "tool_calls": tool_calls,
                "iterations": iteration,
                "final_response": response_obj if 'response_obj' in locals() else None
            }
        )

    async def _process_tool_calls_naturally(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_registry: Dict[str, BaseTool],
        tool_calls_list: List[Dict[str, Any]],
        messages: List[BaseMessage],
        conversation_id: str,
        memory_manager
    ) -> AsyncGenerator[NaturalStreamingEvent, None]:
        """
        Process tool calls with natural timing only - NO ARTIFICIAL DELAYS.
        """
        # Execute tools sequentially to maintain order and yield events
        for tool_call in tool_calls:
            async for event in self._execute_tool_naturally(
                tool_call, tool_registry, tool_calls_list, 
                messages, conversation_id, memory_manager
            ):
                yield event

    async def _execute_tool_naturally(
        self,
        tool_call: Dict[str, Any],
        tool_registry: Dict[str, BaseTool],
        tool_calls_list: List[Dict[str, Any]],
        messages: List[BaseMessage],
        conversation_id: str,
        memory_manager
    ) -> AsyncGenerator[NaturalStreamingEvent, None]:
        """
        Execute a single tool with natural timing - NO ARTIFICIAL DELAYS.
        """
        tool_name = tool_call.get('name', 'unknown')
        tool_args = tool_call.get('args', {})
        tool_call_id = tool_call.get('id', f"call_{len(tool_calls_list)}")

        if tool_name in tool_registry:
            # Stream tool start - NATURAL TIMING
            yield NaturalStreamingEvent(
                event_type="tool_start",
                content=f"🔧 **Using tool: {tool_name}**",
                timestamp=time.time(),
                metadata={
                    "tool_name": tool_name,
                    "tool_args": tool_args
                }
            )

            try:
                # Execute tool in thread pool to avoid blocking - NATURAL TIMING
                loop = asyncio.get_event_loop()
                tool_result = await loop.run_in_executor(
                    self.executor,
                    lambda: tool_registry[tool_name].invoke(tool_args)
                )

                # Stream tool end - NATURAL TIMING
                yield NaturalStreamingEvent(
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
                yield NaturalStreamingEvent(
                    event_type="error",
                    content=f"❌ **Tool error: {str(e)}**",
                    timestamp=time.time(),
                    metadata={
                        "tool_name": tool_name,
                        "error": str(e)
                    }
                )
        else:
            yield NaturalStreamingEvent(
                event_type="error",
                content=f"❌ **Unknown tool: {tool_name}**",
                timestamp=time.time(),
                metadata={"tool_name": tool_name}
            )

    def _convert_langchain_event(self, event: Dict[str, Any]) -> Optional[NaturalStreamingEvent]:
        """Convert LangChain event to our NaturalStreamingEvent format - NO DELAYS"""
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

            return NaturalStreamingEvent(
                event_type=event_type,
                content=content,
                timestamp=time.time(),
                metadata=metadata,
                run_id=run_id,
                parent_ids=parent_ids
            )

        except Exception as e:
            print(f"[NaturalLangChainStreamingManager] Error converting event: {e}")
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


# Global natural streaming manager instance
_natural_streaming_manager = None

def get_natural_streaming_manager() -> NaturalLangChainStreamingManager:
    """Get the global natural streaming manager instance"""
    global _natural_streaming_manager
    if _natural_streaming_manager is None:
        _natural_streaming_manager = NaturalLangChainStreamingManager()
    return _natural_streaming_manager

def reset_natural_streaming_manager() -> None:
    """Reset the global natural streaming manager instance"""
    global _natural_streaming_manager
    if _natural_streaming_manager is not None:
        _natural_streaming_manager.executor.shutdown(wait=False)
        _natural_streaming_manager = None
