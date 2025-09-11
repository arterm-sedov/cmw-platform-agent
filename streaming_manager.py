"""
Streaming Manager Module
========================

This module handles all streaming functionality for real-time LLM responses.
It provides utilities for streaming responses, event handling, and callback management.

Key Features:
- Real-time streaming responses
- Event-based streaming callbacks
- Streaming event management
- Progress tracking and reporting
- Streaming error handling

Usage:
    from streaming_manager import StreamingManager, StreamingCallbackHandler
    
    manager = StreamingManager()
    for event in manager.stream_response(llm, messages):
        print(event['content'])
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Generator, Callable
from dataclasses import dataclass
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage


@dataclass
class StreamingEvent:
    """Represents a streaming event"""
    event_type: str
    content: str
    timestamp: float
    metadata: Dict[str, Any]
    event_id: str


class StreamingCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler for streaming events during tool execution.
    Provides real-time visibility into LLM thinking and tool usage.
    """
    
    def __init__(self, streaming_generator=None, event_handler: Callable = None):
        self.streaming_generator = streaming_generator
        self.event_handler = event_handler
        self.current_tool_name = None
        self.tool_args = None
        self.event_count = 0
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts generating"""
        self._send_event({
            "type": "llm_start",
            "content": "🤖 **LLM is thinking...**\n",
            "metadata": {"llm_type": serialized.get("name", "unknown")}
        })
    
    def on_llm_stream(self, chunk, **kwargs):
        """Called for each streaming chunk from LLM"""
        content = getattr(chunk, 'content', '') or str(chunk)
        if content:
            self._send_event({
                "type": "llm_chunk",
                "content": content,
                "metadata": {"chunk_type": "llm_response"}
            })
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Called when a tool starts executing"""
        self.current_tool_name = serialized.get("name", "unknown_tool")
        self.tool_args = input_str
        
        self._send_event({
            "type": "tool_start",
            "content": f"\n🔧 **Using tool: {self.current_tool_name}**\n",
            "metadata": {
                "tool_name": self.current_tool_name,
                "tool_args": self.tool_args
            }
        })
    
    def on_tool_end(self, output, **kwargs):
        """Called when a tool finishes executing"""
        # Truncate long outputs for display
        display_output = output[:500] + "..." if len(output) > 500 else output
        self._send_event({
            "type": "tool_end",
            "content": f"✅ **Tool completed: {self.current_tool_name}**\n```\n{display_output}\n```\n",
            "metadata": {
                "tool_name": self.current_tool_name,
                "tool_output": output
            }
        })
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM finishes generating"""
        self._send_event({
            "type": "llm_end",
            "content": "\n🎯 **LLM completed**\n",
            "metadata": {"response": str(response)}
        })
    
    def _send_event(self, event_data: Dict[str, Any]):
        """Send event to streaming generator or event handler"""
        self.event_count += 1
        
        if self.streaming_generator:
            try:
                self.streaming_generator.send(event_data)
            except Exception as e:
                print(f"[StreamingCallbackHandler] Error sending event: {e}")
        
        if self.event_handler:
            try:
                self.event_handler(event_data)
            except Exception as e:
                print(f"[StreamingCallbackHandler] Error in event handler: {e}")


class StreamingManager:
    """Manages streaming functionality for LLM responses"""
    
    def __init__(self):
        self.active_streams: Dict[str, Generator] = {}
        self.stream_history: List[StreamingEvent] = []
        self.max_history_size = 1000
    
    def create_streaming_generator(self, stream_id: str = None) -> Generator[Dict[str, Any], None, None]:
        """
        Create a streaming generator for real-time responses.
        
        Args:
            stream_id: Optional stream identifier
            
        Returns:
            Generator: Streaming generator
        """
        if stream_id is None:
            stream_id = str(uuid.uuid4())
        
        def streaming_generator():
            try:
                while True:
                    event_data = yield
                    if event_data is None:
                        break
                    
                    # Create streaming event
                    event = StreamingEvent(
                        event_type=event_data.get('type', 'unknown'),
                        content=event_data.get('content', ''),
                        timestamp=time.time(),
                        metadata=event_data.get('metadata', {}),
                        event_id=str(uuid.uuid4())
                    )
                    
                    # Store in history
                    self._add_to_history(event)
                    
                    # Yield the event
                    yield event_data
                    
            except GeneratorExit:
                # Clean up when generator is closed
                if stream_id in self.active_streams:
                    del self.active_streams[stream_id]
            except Exception as e:
                print(f"[StreamingManager] Error in streaming generator: {e}")
        
        # Store active stream
        self.active_streams[stream_id] = streaming_generator()
        return streaming_generator()
    
    def stream_llm_response(self, llm, messages: List[Any], use_tools: bool = False) -> Generator[Dict[str, Any], None, None]:
        """
        Stream LLM response with real-time updates.
        
        Args:
            llm: LLM instance
            messages: List of messages
            use_tools: Whether to use tools
            
        Returns:
            Generator: Streaming response generator
        """
        try:
            # Check if LLM supports streaming
            if hasattr(llm, 'stream') and callable(getattr(llm, 'stream')):
                # Use native streaming
                for chunk in llm.stream(messages):
                    try:
                        content = getattr(chunk, 'content', '') or getattr(chunk, 'text', '') or str(chunk)
                        if content:
                            yield {
                                'type': 'llm_chunk',
                                'content': content,
                                'metadata': {
                                    'chunk_type': 'streaming',
                                    'use_tools': use_tools
                                }
                            }
                    except Exception as e:
                        print(f"[StreamingManager] Error processing chunk: {e}")
                        continue
            else:
                # Fallback to non-streaming with simulated streaming
                response = llm.invoke(messages)
                content = getattr(response, 'content', '') or str(response)
                
                if content:
                    # Simulate streaming by yielding content in chunks
                    chunk_size = 50
                    for i in range(0, len(content), chunk_size):
                        chunk = content[i:i + chunk_size]
                        yield {
                            'type': 'llm_chunk',
                            'content': chunk,
                            'metadata': {
                                'chunk_type': 'simulated',
                                'use_tools': use_tools
                            }
                        }
                        # Small delay to simulate streaming
                        time.sleep(0.01)
        
        except Exception as e:
            yield {
                'type': 'error',
                'content': f"Streaming error: {str(e)}",
                'metadata': {'error': str(e)}
            }
    
    def stream_tool_execution(self, tool_name: str, tool_args: Dict[str, Any], 
                            tool_func: Callable) -> Generator[Dict[str, Any], None, None]:
        """
        Stream tool execution with progress updates.
        
        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            tool_func: Tool function to execute
            
        Returns:
            Generator: Tool execution streaming generator
        """
        try:
            # Tool start event
            yield {
                'type': 'tool_start',
                'content': f"🔧 **Using tool: {tool_name}**\n",
                'metadata': {
                    'tool_name': tool_name,
                    'tool_args': tool_args
                }
            }
            
            # Execute tool
            start_time = time.time()
            result = tool_func.invoke(tool_args)
            execution_time = time.time() - start_time
            
            # Tool end event
            display_result = str(result)[:500] + "..." if len(str(result)) > 500 else str(result)
            yield {
                'type': 'tool_end',
                'content': f"✅ **Tool completed: {tool_name}**\n```\n{display_result}\n```\n",
                'metadata': {
                    'tool_name': tool_name,
                    'tool_result': str(result),
                    'execution_time': execution_time
                }
            }
        
        except Exception as e:
            yield {
                'type': 'tool_error',
                'content': f"❌ **Tool error: {tool_name}**\n```\n{str(e)}\n```\n",
                'metadata': {
                    'tool_name': tool_name,
                    'error': str(e)
                }
            }
    
    def stream_tool_calling_loop(self, llm, messages: List[Any], tool_registry: Dict[str, Any],
                               max_steps: int = 20) -> Generator[Dict[str, Any], None, None]:
        """
        Stream a tool calling loop with real-time updates.
        
        Args:
            llm: LLM instance
            messages: List of messages
            tool_registry: Registry of available tools
            max_steps: Maximum number of steps
            
        Returns:
            Generator: Tool calling loop streaming generator
        """
        try:
            # Loop start event
            yield {
                'type': 'loop_start',
                'content': f"🔄 **Starting tool calling loop** (max {max_steps} steps)\n",
                'metadata': {'max_steps': max_steps}
            }
            
            step = 0
            while step < max_steps:
                step += 1
                
                # Step start event
                yield {
                    'type': 'step_start',
                    'content': f"📝 **Step {step}/{max_steps}**\n",
                    'metadata': {'step': step, 'max_steps': max_steps}
                }
                
                # Get LLM response
                response = llm.invoke(messages)
                
                # Check for tool calls
                tool_calls = self._extract_tool_calls(response)
                
                if tool_calls:
                    # Execute tools
                    for tool_call in tool_calls:
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('args', {})
                        
                        if tool_name in tool_registry:
                            # Stream tool execution
                            for event in self.stream_tool_execution(tool_name, tool_args, tool_registry[tool_name]):
                                yield event
                            
                            # Add tool result to messages
                            tool_message = {
                                'type': 'tool',
                                'content': str(tool_registry[tool_name].invoke(tool_args)),
                                'tool_call_id': tool_call.get('id', str(uuid.uuid4()))
                            }
                            messages.append(tool_message)
                        else:
                            yield {
                                'type': 'tool_error',
                                'content': f"❌ **Unknown tool: {tool_name}**\n",
                                'metadata': {'tool_name': tool_name}
                            }
                else:
                    # No tool calls - check for final answer
                    final_answer = self._extract_final_answer(response)
                    if final_answer:
                        yield {
                            'type': 'final_answer',
                            'content': f"🎯 **Final Answer:**\n{final_answer}\n",
                            'metadata': {'answer': final_answer}
                        }
                        break
                
                # Step end event
                yield {
                    'type': 'step_end',
                    'content': f"✅ **Step {step} completed**\n",
                    'metadata': {'step': step}
                }
            
            # Loop end event
            yield {
                'type': 'loop_end',
                'content': f"🏁 **Tool calling loop completed**\n",
                'metadata': {'total_steps': step}
            }
        
        except Exception as e:
            yield {
                'type': 'loop_error',
                'content': f"❌ **Loop error:** {str(e)}\n",
                'metadata': {'error': str(e)}
            }
    
    def _extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM response"""
        tool_calls = []
        
        try:
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_calls.extend(response.tool_calls)
            elif hasattr(response, 'function_call') and response.function_call:
                tool_calls.append(response.function_call)
        except Exception as e:
            print(f"[StreamingManager] Error extracting tool calls: {e}")
        
        return tool_calls
    
    def _extract_final_answer(self, response: Any) -> Optional[str]:
        """Extract final answer from response"""
        try:
            content = getattr(response, 'content', '')
            if isinstance(content, str) and len(content.strip()) > 10:
                return content.strip()
        except Exception as e:
            print(f"[StreamingManager] Error extracting final answer: {e}")
        
        return None
    
    def _add_to_history(self, event: StreamingEvent):
        """Add event to history"""
        self.stream_history.append(event)
        
        # Keep history size manageable
        if len(self.stream_history) > self.max_history_size:
            self.stream_history = self.stream_history[-self.max_history_size:]
    
    def get_stream_history(self, limit: int = 100) -> List[StreamingEvent]:
        """Get streaming history"""
        return self.stream_history[-limit:]
    
    def get_active_streams(self) -> List[str]:
        """Get list of active stream IDs"""
        return list(self.active_streams.keys())
    
    def close_stream(self, stream_id: str):
        """Close a specific stream"""
        if stream_id in self.active_streams:
            try:
                self.active_streams[stream_id].close()
            except Exception as e:
                print(f"[StreamingManager] Error closing stream {stream_id}: {e}")
            finally:
                del self.active_streams[stream_id]
    
    def close_all_streams(self):
        """Close all active streams"""
        for stream_id in list(self.active_streams.keys()):
            self.close_stream(stream_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming manager statistics"""
        return {
            'active_streams': len(self.active_streams),
            'total_events': len(self.stream_history),
            'max_history_size': self.max_history_size
        }


# Global streaming manager instance
_streaming_manager = None

def get_streaming_manager() -> StreamingManager:
    """Get the global streaming manager instance"""
    global _streaming_manager
    if _streaming_manager is None:
        _streaming_manager = StreamingManager()
    return _streaming_manager

def reset_streaming_manager():
    """Reset the global streaming manager instance"""
    global _streaming_manager
    if _streaming_manager:
        _streaming_manager.close_all_streams()
    _streaming_manager = None
