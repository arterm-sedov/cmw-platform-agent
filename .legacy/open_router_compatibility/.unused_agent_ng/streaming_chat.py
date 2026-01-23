"""
Modern Streaming Chat Interface
==============================

A sophisticated streaming chat interface that provides real-time thinking
transparency and tool usage visualization using Gradio's ChatMessage metadata.

Key Features:
- Real-time thinking transparency with collapsible sections
- Tool usage visualization with metadata
- Streaming response handling
- Integration with debug system
- Clean separation of concerns
- Support for multiple LLM providers

Based on Gradio's ChatMessage metadata system for thinking transparency.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass
import json

from .debug_streamer import get_debug_streamer, get_log_handler, get_thinking_transparency, LogLevel, LogCategory


@dataclass
class ChatMessage:
    """Enhanced ChatMessage with metadata support for thinking transparency"""
    role: str  # "user" or "assistant"
    content: str
    metadata: Optional[Dict[str, Any]] = None

    def to_gradio_format(self) -> Dict[str, Any]:
        """Convert to Gradio ChatMessage format"""
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata or {}
        }


class StreamingChatInterface:
    """
    Modern streaming chat interface with thinking transparency.

    This class handles real-time streaming of chat responses with
    thinking transparency, tool usage visualization, and debug logging.
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.debug_streamer = get_debug_streamer(session_id)
        self.log_handler = get_log_handler(session_id)
        self.thinking_transparency = get_thinking_transparency(session_id)

        # Chat state
        self.current_thinking = ""
        self.current_tool_usage = []
        self.is_thinking = False

    async def stream_chat_response(self, message: str, history: List[Tuple[str, str]], 
                                 agent: Any) -> AsyncGenerator[Tuple[List[Tuple[str, str]], str], None]:
        """
        Stream a chat response with thinking transparency.

        Args:
            message: User message
            history: Chat history as list of tuples
            agent: The agent instance to use

        Yields:
            Updated history and empty message for Gradio
        """
        if not message.strip():
            yield history, ""
            return

        self.debug_streamer.info(f"Starting chat response for: {message[:50]}...", LogCategory.STREAM)

        # Add user message to history
        working_history = history + [(message, "")]
        yield working_history, ""

        try:
            # Start thinking process
            await self._start_thinking_process(message)

            # Stream the response with timeout protection
            timeout_seconds = 60  # 60 second timeout
            stream_start = time.time()

            async for event in self._stream_agent_response(message, history, agent):
                # Check for timeout
                if time.time() - stream_start > timeout_seconds:
                    self.debug_streamer.warning("Streaming timeout reached", LogCategory.STREAM)
                    break

                try:
                    if event["type"] == "thinking":
                        await self._handle_thinking_event(event, working_history)
                    elif event["type"] == "tool_use":
                        await self._handle_tool_use_event(event, working_history)
                    elif event["type"] == "content":
                        await self._handle_content_event(event, working_history)
                    elif event["type"] == "error":
                        await self._handle_error_event(event, working_history)

                    yield working_history, ""
                except Exception as stream_error:
                    self.debug_streamer.warning(f"Stream event error: {str(stream_error)}", LogCategory.STREAM)
                    # Continue processing other events
                    continue

            # Complete the response
            await self._complete_response(working_history)
            yield working_history, ""

        except Exception as e:
            self.debug_streamer.error(f"Error in streaming chat: {str(e)}", LogCategory.STREAM)
            await self._handle_streaming_error(e, working_history)
            yield working_history, ""

    async def _start_thinking_process(self, message: str):
        """Start the thinking process"""
        self.is_thinking = True
        self.current_thinking = ""
        self.current_tool_usage = []

        self.thinking_transparency.start_thinking(
            title="🧠 Thinking",
            metadata={"message": message[:100] + "..." if len(message) > 100 else message}
        )

        self.debug_streamer.thinking(f"Starting to process: {message}")

    async def _stream_agent_response(self, message: str, history: List[Tuple[str, str]], 
                                   agent: Any) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response from the agent"""
        try:
            # Convert history to internal format
            internal_history = []
            for user_msg, assistant_msg in history:
                internal_history.append(ChatMessage(role="user", content=user_msg))
                if assistant_msg:
                    internal_history.append(ChatMessage(role="assistant", content=assistant_msg))

            # Stream from agent
            if hasattr(agent, 'stream_chat'):
                async for event in agent.stream_chat(message, internal_history):
                    yield event
            elif hasattr(agent, 'stream'):
                # Fallback to basic streaming
                for chunk in agent.stream(message, internal_history):
                    yield {"type": "content", "content": chunk}
            else:
                # Fallback to non-streaming
                response = agent(message, chat_history=internal_history)
                if hasattr(response, 'answer'):
                    yield {"type": "content", "content": response.answer}
                else:
                    yield {"type": "content", "content": str(response)}

        except Exception as e:
            self.debug_streamer.error(f"Error streaming from agent: {str(e)}", LogCategory.STREAM)
            yield {"type": "error", "content": f"Error: {str(e)}"}

    async def _handle_thinking_event(self, event: Dict[str, Any], working_history: List[Tuple[str, str]]):
        """Handle thinking events"""
        thinking_content = event.get("content", "")
        self.current_thinking += thinking_content

        # Update thinking in the last assistant message
        if working_history and len(working_history) > 0:
            last_user, last_assistant = working_history[-1]
            if last_assistant == "":
                # Create thinking message with metadata
                thinking_message = self.thinking_transparency._create_thinking_message()
                working_history[-1] = (last_user, thinking_message["content"])
            else:
                # Update existing thinking
                working_history[-1] = (last_user, self.current_thinking)

    async def _handle_tool_use_event(self, event: Dict[str, Any], working_history: List[Tuple[str, str]]):
        """Handle tool usage events"""
        tool_name = event.get("content", "unknown")
        tool_metadata = event.get("metadata", {})

        self.current_tool_usage.append({
            "tool_name": tool_name,
            "metadata": tool_metadata
        })

        # Create tool usage message
        tool_message = self.thinking_transparency.create_tool_usage_message(
            tool_name=tool_name,
            tool_args=tool_metadata.get("tool_args", {}),
            result=tool_metadata.get("result", "Tool executed")
        )

        # Add tool usage to history
        if working_history and len(working_history) > 0:
            last_user, last_assistant = working_history[-1]
            working_history[-1] = (last_user, tool_message["content"])

    async def _handle_content_event(self, event: Dict[str, Any], working_history: List[Tuple[str, str]]):
        """Handle content events"""
        content = event.get("content", "")

        if working_history and len(working_history) > 0:
            last_user, last_assistant = working_history[-1]

            # If we have thinking content, complete it first
            if self.is_thinking and self.current_thinking:
                self.thinking_transparency.complete_thinking()
                self.is_thinking = False

            # Update the response content
            if last_assistant == "":
                working_history[-1] = (last_user, content)
            else:
                # Append to existing content
                working_history[-1] = (last_user, last_assistant + content)

    async def _handle_error_event(self, event: Dict[str, Any], working_history: List[Tuple[str, str]]):
        """Handle error events"""
        error_content = event.get("content", "Unknown error")

        # Log the error for debugging
        self.debug_streamer.error(f"Streaming error: {error_content}", LogCategory.LLM)

        if working_history and len(working_history) > 0:
            last_user, last_assistant = working_history[-1]
            working_history[-1] = (last_user, f"❌ {error_content}")

    async def _complete_response(self, working_history: List[Tuple[str, str]]):
        """Complete the response process"""
        if self.is_thinking:
            self.thinking_transparency.complete_thinking()
            self.is_thinking = False

        self.debug_streamer.success("Chat response completed", LogCategory.STREAM)

    async def _handle_streaming_error(self, error: Exception, working_history: List[Tuple[str, str]]):
        """Handle streaming errors"""
        error_msg = f"❌ **Streaming Error**\n\n{str(error)}\n\nPlease try again."

        if working_history and len(working_history) > 0:
            last_user, last_assistant = working_history[-1]
            working_history[-1] = (last_user, error_msg)

    def create_thinking_message(self, content: str, title: str = "🧠 Thinking") -> ChatMessage:
        """Create a thinking message with metadata"""
        return ChatMessage(
            role="assistant",
            content=content,
            metadata={
                "title": title,
                "status": "pending" if self.is_thinking else "done"
            }
        )

    def create_tool_usage_message(self, tool_name: str, tool_args: Dict[str, Any], 
                                 result: str) -> ChatMessage:
        """Create a tool usage message with metadata"""
        return ChatMessage(
            role="assistant",
            content=f"🔧 **{tool_name}**\n\n**Arguments:**\n{json.dumps(tool_args, indent=2)}\n\n**Result:**\n{result}",
            metadata={
                "title": f"🔧 {tool_name}",
                "status": "done",
                "tool_name": tool_name,
                "tool_args": tool_args
            }
        )

    def create_error_message(self, error: str, error_type: str = "error") -> ChatMessage:
        """Create an error message with metadata"""
        return ChatMessage(
            role="assistant",
            content=f"❌ **{error_type.title()}**\n\n{error}",
            metadata={
                "title": f"❌ {error_type.title()}",
                "status": "done",
                "error": True
            }
        )


class GradioChatInterface:
    """
    Gradio-specific chat interface that handles the conversion between
    internal chat format and Gradio's expected format.
    """

    def __init__(self, session_id: str = "default"):
        self.streaming_chat = StreamingChatInterface(session_id)
        self.debug_streamer = get_debug_streamer(session_id)
        self.log_handler = get_log_handler(session_id)

    async def chat_with_agent(self, message: str, history: List[Tuple[str, str]], 
                            agent: Any) -> Tuple[List[Tuple[str, str]], str]:
        """
        Chat with the agent using Gradio format.

        Args:
            message: User message
            history: Chat history as list of tuples
            agent: The agent instance

        Returns:
            Updated history and empty message
        """
        try:
            # Stream the response with proper error handling
            updated_history = history
            async for updated_history, _ in self.streaming_chat.stream_chat_response(
                message, history, agent
            ):
                # Yield intermediate results for real-time updates
                # This helps prevent session management issues
                try:
                    pass
                except Exception as stream_error:
                    self.debug_streamer.warning(f"Streaming warning: {str(stream_error)}", LogCategory.STREAM)
                    break

            return updated_history, ""

        except Exception as e:
            self.debug_streamer.error(f"Error in Gradio chat interface: {str(e)}", LogCategory.STREAM)
            error_history = history + [(message, f"❌ Error: {str(e)}")]
            return error_history, ""

    def get_current_logs(self) -> str:
        """Get current logs for the Logs tab"""
        return self.log_handler.get_current_logs()

    def clear_logs(self):
        """Clear the logs"""
        self.log_handler.clear_logs()


# Global chat interface instances
_global_chat_interface: Optional[GradioChatInterface] = None


def get_chat_interface(session_id: str = "default") -> GradioChatInterface:
    """Get the global chat interface instance"""
    global _global_chat_interface
    if _global_chat_interface is None:
        _global_chat_interface = GradioChatInterface(session_id)
    return _global_chat_interface
