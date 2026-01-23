"""
Simple Streaming Implementation
==============================

A simplified streaming implementation that works with the current LLM setup
and provides real-time streaming responses for Gradio.

Key Features:
- Simple character-by-character streaming
- Tool usage visualization
- Error handling
- Integration with existing LLM infrastructure
"""

import asyncio
import time
from typing import Dict, List, Any, AsyncGenerator
from dataclasses import dataclass


@dataclass
class SimpleStreamingEvent:
    """Simple streaming event"""
    event_type: str
    content: str
    metadata: Dict[str, Any] = None


class SimpleStreamingManager:
    """
    Simple streaming manager that provides character-by-character streaming
    for better user experience.
    """

    def __init__(self):
        self.chunk_size = 3  # Characters per chunk
        self.delay = 0.05    # Delay between chunks in seconds

    async def stream_text(self, text: str, event_type: str = "content") -> AsyncGenerator[SimpleStreamingEvent, None]:
        """
        Stream text character by character.

        Args:
            text: Text to stream
            event_type: Type of event

        Yields:
            SimpleStreamingEvent objects
        """
        if not text:
            return

        # Stream text in chunks
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]

            event = SimpleStreamingEvent(
                event_type=event_type,
                content=chunk,
                metadata={"position": i, "total": len(text)}
            )

            yield event

            # Small delay for streaming effect
            await asyncio.sleep(self.delay)

    async def stream_response_with_tools(
        self, 
        response: str, 
        tool_calls: List[Dict[str, Any]] = None
    ) -> AsyncGenerator[SimpleStreamingEvent, None]:
        """
        Stream response with tool usage visualization.

        Args:
            response: Response text
            tool_calls: List of tool calls made

        Yields:
            SimpleStreamingEvent objects
        """
        # Stream tool usage first (filter out schema tools)
        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call.get('name', 'unknown')
                tool_args = tool_call.get('args', {})

                # Skip empty or invalid tool names
                if not tool_name or tool_name.strip() == "":
                    continue

                # Skip SGR tools from display (they're executed but not shown)
                sgr_tools = {'submit_answer', 'submit_intermediate_step'}
                if tool_name in sgr_tools:
                    continue

                # Show meaningful tool usage
                if tool_name in {'add', 'subtract', 'multiply', 'divide'}:
                    # Math tools - show the calculation
                    if 'a' in tool_args and 'b' in tool_args:
                        yield SimpleStreamingEvent(
                            event_type="tool_start",
                            content=f"🧮 Calculating: {tool_args['a']} {tool_name} {tool_args['b']}",
                            metadata={"tool_name": tool_name, "args": tool_args}
                        )
                    else:
                        yield SimpleStreamingEvent(
                            event_type="tool_start",
                            content=f"🧮 Using {tool_name}",
                            metadata={"tool_name": tool_name}
                        )
                else:
                    # Other tools
                    yield SimpleStreamingEvent(
                        event_type="tool_start",
                        content=f"🔧 Using tool: {tool_name}",
                        metadata={"tool_name": tool_name}
                    )

                # Small delay
                await asyncio.sleep(0.3)

                # Tool end event
                yield SimpleStreamingEvent(
                    event_type="tool_end",
                    content=f"✅ {tool_name} completed",
                    metadata={"tool_name": tool_name}
                )

                # Small delay
                await asyncio.sleep(0.2)

        # Stream the response text if available
        if response and response.strip():
            async for event in self.stream_text(response, "content"):
                yield event
        else:
            # If no response text, show a completion message
            yield SimpleStreamingEvent(
                event_type="content",
                content="✅ Response completed",
                metadata={"completion": True}
            )

    async def stream_thinking(self, message: str) -> AsyncGenerator[SimpleStreamingEvent, None]:
        """
        Stream thinking process.

        Args:
            message: Message being processed

        Yields:
            SimpleStreamingEvent objects
        """
        thinking_messages = [
            "🤖 **Thinking...**",
            "🧠 **Analyzing request...**",
            "🔍 **Processing...**",
            "⚡ **Generating response...**"
        ]

        for thinking_msg in thinking_messages:
            yield SimpleStreamingEvent(
                event_type="thinking",
                content=thinking_msg,
                metadata={"message": message[:50] + "..." if len(message) > 50 else message}
            )
            await asyncio.sleep(0.8)

    async def stream_error(self, error: str) -> AsyncGenerator[SimpleStreamingEvent, None]:
        """
        Stream error message.

        Args:
            error: Error message

        Yields:
            SimpleStreamingEvent objects
        """
        yield SimpleStreamingEvent(
            event_type="error",
            content=f"❌ **Error: {error}**",
            metadata={"error": error}
        )


# Global streaming manager instance
_simple_streaming_manager = None

def get_simple_streaming_manager() -> SimpleStreamingManager:
    """Get the global simple streaming manager instance"""
    global _simple_streaming_manager
    if _simple_streaming_manager is None:
        _simple_streaming_manager = SimpleStreamingManager()
    return _simple_streaming_manager
