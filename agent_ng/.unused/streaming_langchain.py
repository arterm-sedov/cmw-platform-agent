"""
LangChain Native Streaming Implementation
========================================

This module implements proper LangChain streaming using the official patterns
from the LangChain documentation for streaming chat model responses.

Key Features:
- Native LangChain streaming with astream_events
- Proper callback propagation
- Real-time streaming with metadata
- Tool calling with streaming
- Error handling and recovery
- Integration with Gradio streaming

Based on: https://python.langchain.com/docs/how_to/streaming/
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from dataclasses import dataclass

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig


@dataclass
class StreamingEvent:
    """Represents a streaming event with metadata"""
    event_type: str
    content: str
    timestamp: float
    metadata: Dict[str, Any]
    run_id: Optional[str] = None
    parent_ids: List[str] = None


class LangChainStreamingCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler for LangChain streaming events.
    
    This handler captures all streaming events and provides them
    in a structured format for real-time display.
    """
    
    def __init__(self, event_handler: Callable[[StreamingEvent], None] = None):
        self.event_handler = event_handler
        self.events = []
        self.current_run_id = None
        self.current_tool_name = None
        self.tool_args = None
        self.accumulated_content = ""
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts generating"""
        self.current_run_id = kwargs.get("run_id")
        self.accumulated_content = ""
        
        event = StreamingEvent(
            event_type="llm_start",
            content="🤖 **LLM is thinking...**",
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
            
            event = StreamingEvent(
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
        self.tool_args = input_str
        
        event = StreamingEvent(
            event_type="tool_start",
            content=f"🔧 **Using tool: {self.current_tool_name}**",
            timestamp=time.time(),
            metadata={
                "tool_name": self.current_tool_name,
                "tool_args": self.tool_args
            },
            run_id=self.current_run_id
        )
        
        self._emit_event(event)
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes executing"""
        # Truncate long outputs for display
        display_output = output[:500] + "..." if len(output) > 500 else output
        
        event = StreamingEvent(
            event_type="tool_end",
            content=f"✅ **Tool completed: {self.current_tool_name}**\n```\n{display_output}\n```",
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
        event = StreamingEvent(
            event_type="llm_end",
            content="🎯 **LLM completed**",
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
        event = StreamingEvent(
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
        event = StreamingEvent(
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
        event = StreamingEvent(
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
    
    def _emit_event(self, event: StreamingEvent) -> None:
        """Emit an event to the handler and store it"""
        self.events.append(event)
        
        if self.event_handler:
            try:
                self.event_handler(event)
            except Exception as e:
                print(f"[LangChainStreamingCallbackHandler] Error in event handler: {e}")


class LangChainStreamingManager:
    """
    Manages LangChain streaming with proper event handling and callback propagation.
    
    This class implements the streaming patterns from the LangChain documentation
    for real-time streaming of chat model responses with tool calls.
    """
    
    def __init__(self):
        self.active_streams = {}
        self.event_handlers = []
    
    def add_event_handler(self, handler: Callable[[StreamingEvent], None]) -> None:
        """Add an event handler for streaming events"""
        self.event_handlers.append(handler)
    
    def remove_event_handler(self, handler: Callable[[StreamingEvent], None]) -> None:
        """Remove an event handler"""
        if handler in self.event_handlers:
            self.event_handlers.remove(handler)
    
    async def stream_llm_response(
        self, 
        llm, 
        messages: List[BaseMessage], 
        tools: List[BaseTool] = None,
        config: Optional[RunnableConfig] = None
    ) -> AsyncGenerator[StreamingEvent, None]:
        """
        Stream LLM response using LangChain's native streaming.
        
        Args:
            llm: LangChain LLM instance
            messages: List of messages
            tools: Optional list of tools
            config: Optional runnable config
            
        Yields:
            StreamingEvent objects
        """
        try:
            # Create callback handler
            events = []
            
            def event_handler(event: StreamingEvent):
                events.append(event)
            
            callback_handler = LangChainStreamingCallbackHandler(event_handler)
            
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
            error_event = StreamingEvent(
                event_type="error",
                content=f"Streaming error: {str(e)}",
                timestamp=time.time(),
                metadata={"error": str(e)}
            )
            yield error_event
    
    async def stream_chain_response(
        self,
        chain,
        inputs: Dict[str, Any],
        config: Optional[RunnableConfig] = None
    ) -> AsyncGenerator[StreamingEvent, None]:
        """
        Stream chain response using LangChain's native streaming.
        
        Args:
            chain: LangChain chain
            inputs: Chain inputs
            config: Optional runnable config
            
        Yields:
            StreamingEvent objects
        """
        try:
            # Create callback handler
            events = []
            
            def event_handler(event: StreamingEvent):
                events.append(event)
            
            callback_handler = LangChainStreamingCallbackHandler(event_handler)
            
            # Create runnable config with callbacks
            if config is None:
                config = RunnableConfig(callbacks=[callback_handler])
            else:
                if config.get("callbacks"):
                    config["callbacks"].append(callback_handler)
                else:
                    config["callbacks"] = [callback_handler]
            
            # Stream the response using astream_events
            async for event in chain.astream_events(
                inputs,
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
            error_event = StreamingEvent(
                event_type="error",
                content=f"Chain streaming error: {str(e)}",
                timestamp=time.time(),
                metadata={"error": str(e)}
            )
            yield error_event
    
    def _convert_langchain_event(self, event: Dict[str, Any]) -> Optional[StreamingEvent]:
        """Convert LangChain event to our StreamingEvent format"""
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
                content = "🤖 **LLM is thinking...**"
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
                display_output = output[:500] + "..." if len(output) > 500 else output
                content = f"✅ **Tool completed: {tool_name}**\n```\n{display_output}\n```"
                metadata.update({
                    "tool_output": output
                })
            elif event_type == "on_llm_end":
                content = "🎯 **LLM completed**"
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
            
            return StreamingEvent(
                event_type=event_type,
                content=content,
                timestamp=time.time(),
                metadata=metadata,
                run_id=run_id,
                parent_ids=parent_ids
            )
            
        except Exception as e:
            print(f"[LangChainStreamingManager] Error converting event: {e}")
            return None
    
    def get_active_streams(self) -> List[str]:
        """Get list of active stream IDs"""
        return list(self.active_streams.keys())
    
    def close_stream(self, stream_id: str) -> None:
        """Close a specific stream"""
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]


# Global streaming manager instance
_streaming_manager = None

def get_streaming_manager() -> LangChainStreamingManager:
    """Get the global streaming manager instance"""
    global _streaming_manager
    if _streaming_manager is None:
        _streaming_manager = LangChainStreamingManager()
    return _streaming_manager

def reset_streaming_manager() -> None:
    """Reset the global streaming manager instance"""
    global _streaming_manager
    _streaming_manager = None
