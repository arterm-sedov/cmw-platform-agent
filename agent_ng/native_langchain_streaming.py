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
    
    def _get_processing_complete_message(self, language: str = "en") -> str:
        """Get the localized processing complete message"""
        from .i18n_translations import get_translation_key
        return get_translation_key("processing_complete", language)
    
    def _get_response_completed_message(self, language: str = "en") -> str:
        """Get the localized response completed message"""
        from .i18n_translations import get_translation_key
        return get_translation_key("response_completed", language)
    
    def _get_processing_failed_message(self, language: str = "en") -> str:
        """Get the localized processing failed message"""
        from .i18n_translations import get_translation_key
        return get_translation_key("processing_failed", language)
    
    def _get_iteration_processing_message(self, iteration: int, max_iterations: int, language: str = "en") -> str:
        """Get the localized iteration processing message"""
        from .i18n_translations import get_translation_key
        template = get_translation_key("iteration_processing", language)
        return template.format(iteration=iteration, max_iterations=max_iterations)
    
    def _get_iteration_finished_message(self, iteration: int, max_iterations: int, language: str = "en") -> str:
        """Get the localized iteration finished message"""
        from .i18n_translations import get_translation_key
        template = get_translation_key("iteration_finished", language)
        return template.format(iteration=iteration, max_iterations=max_iterations)
    
    def _get_iteration_completed_message(self, iteration: int, language: str = "en") -> str:
        """Get the localized iteration completed message"""
        from .i18n_translations import get_translation_key
        template = get_translation_key("iteration_completed", language)
        return template.format(iteration=iteration)
    
    def _get_iteration_max_reached_message(self, iteration: int, max_iterations: int, language: str = "en") -> str:
        """Get the localized iteration max reached message"""
        from .i18n_translations import get_translation_key
        template = get_translation_key("iteration_max_reached", language)
        return template.format(iteration=iteration, max_iterations=max_iterations)
    
    def _get_max_iterations_warning_message(self, max_iterations: int, language: str = "en") -> str:
        """Get the localized max iterations warning message"""
        from .i18n_translations import get_translation_key
        template = get_translation_key("max_iterations_warning", language)
        return template.format(max_iterations=max_iterations)
    
    def _get_tool_called_message(self, tool_name: str, language: str = "en") -> str:
        """Get the localized tool called message"""
        from .i18n_translations import get_translation_key
        template = get_translation_key("tool_called", language)
        return template.format(tool_name=tool_name)
    
    def _get_call_count_message(self, total_calls: int, language: str = "en") -> str:
        """Get the localized call count message"""
        from .i18n_translations import get_translation_key
        template = get_translation_key("call_count", language)
        return template.format(total_calls=total_calls)
    
    def _get_result_message(self, tool_result: str, language: str = "en") -> str:
        """Get the localized result message"""
        from .i18n_translations import get_translation_key
        template = get_translation_key("result", language)
        return template.format(tool_result=tool_result)
    
    def _get_tool_error_message(self, error: str, language: str = "en") -> str:
        """Get the localized tool error message"""
        from .i18n_translations import get_translation_key
        template = get_translation_key("tool_error", language)
        return template.format(error=error)
    
    def _get_unknown_tool_message(self, tool_name: str, language: str = "en") -> str:
        """Get the localized unknown tool message"""
        from .i18n_translations import get_translation_key
        template = get_translation_key("unknown_tool", language)
        return template.format(tool_name=tool_name)
    
    def _get_error_message(self, error: str, language: str = "en") -> str:
        """Get the localized error message"""
        from .i18n_translations import get_translation_key
        template = get_translation_key("error", language)
        return template.format(error=error)
    
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
            
            # Create messages list for LLM context
            messages = []
            
            # Always add system message to LLM context (required for every call)
            system_message = SystemMessage(content=agent.system_prompt)
            messages.append(system_message)
            
            # Check if system message is already in memory, if not add it
            system_in_history = any(isinstance(msg, SystemMessage) for msg in chat_history)
            if not system_in_history:
                # Store system message in memory only once
                agent.memory_manager.add_message(conversation_id, system_message)
                print("🔍 DEBUG: Added system message to memory (first time)")
            else:
                print("🔍 DEBUG: System message already in memory, skipping storage")
            
            # Add conversation history (excluding system messages to avoid duplication)
            non_system_history = [msg for msg in chat_history if not isinstance(msg, SystemMessage)]
            messages.extend(non_system_history)
            
            # Create user message and save to memory
            user_message = HumanMessage(content=message)
            messages.append(user_message)
            agent.memory_manager.add_message(conversation_id, user_message)
            
            # Get LLM with tools (tools are already bound in the LLM instance)
            llm_with_tools = agent.llm_instance.llm
            print("🔍 DEBUG: Using LLM instance with pre-bound tools")
            
            # Multi-turn conversation loop for proper tool calling
            iteration = 0
            
            while iteration < self.max_iterations:
                iteration += 1
                print(f"🔍 DEBUG: Starting iteration {iteration}")
                
                # Stream iteration progress as separate message with pseudo-animation
                # Cycle through different processing icons for pseudo-animation
                processing_icons = ["🔄", "⚙️", "🔧", "⚡", "🔄", "⚙️", "🔧", "⚡"]
                icon_index = (iteration - 1) % len(processing_icons)
                current_icon = processing_icons[icon_index]
                
                # Get language from agent if available, otherwise default to English
                language = getattr(agent, 'language', 'en')
                
                yield StreamingEvent(
                    event_type="iteration_progress",
                    content=f'{current_icon} **{self._get_iteration_processing_message(iteration, self.max_iterations, language)}**',
                    metadata={"iteration": iteration, "max_iterations": self.max_iterations, "icon_index": icon_index}
                )
                
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
                                # yield StreamingEvent(
                                #     event_type="tool_start",
                                #     content=f"\n\n2🔧 **Using tool: {tool_name}**",
                                #     metadata={
                                #         "tool_name": tool_name,
                                #         "tool_call_id": tool_call_id,
                                #         "title": f"1 🔧 Tool called: {tool_name}"
                                #     }
                                # )
                            
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
                            # duplicate_count = 0
                            total_calls = 1
                            try:
                                if is_duplicate:
                                    # Use cached result for duplicate tool call
                                    tool_result = cached_result['result'] if cached_result else "Error: Cached result not found"
                                    total_calls += deduplicator.get_duplicate_count(tool_name, tool_args, conversation_id)
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
                                            content=f"\n{self._get_call_count_message(total_calls, language)}\n{self._get_result_message(str(tool_result), language)}",
                                            metadata={
                                                "tool_name": tool_name,
                                                "tool_output": str(tool_result),
                                                "duplicate": False,
                                                "duplicate_count": total_calls,
                                                "title": self._get_tool_called_message(tool_name, language)
                                            }
                                        )
                                    else:
                                        yield StreamingEvent(
                                            event_type="error",
                                            content=self._get_unknown_tool_message(tool_name, language),
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
                                # Don't add to memory here - will be added at the end
                                
                                # Remove from in-progress
                                del tool_calls_in_progress[tool_call_id]
                                    
                            except Exception as e:
                                yield StreamingEvent(
                                    event_type="error",
                                    content=self._get_tool_error_message(str(e), language),
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
                    # Don't add to memory here - will be added at the end
                
                # If no tool calls, we're done
                if not has_tool_calls:
                    print(f"🔍 DEBUG: No tool calls in iteration {iteration}, conversation complete")
                    
                    # Stream conversation completion
                    finish_icons = ["🎉", "🏁", "✨", "🎯"]
                    finish_icon = finish_icons[iteration % len(finish_icons)]
                    
                    yield StreamingEvent(
                        event_type="iteration_progress",
                        content=f"{finish_icon} **{self._get_iteration_finished_message(iteration, self.max_iterations, language)}**",
                        metadata={"iteration": iteration, "max_iterations": self.max_iterations, "conversation_complete": True}
                    )
                    break
                
                print(f"🔍 DEBUG: Completed iteration {iteration}, continuing...")
                
                # Stream iteration completion as separate message
                completion_icons = ["✅", "✔️", "🎯", "✨"]
                completion_icon = completion_icons[iteration % len(completion_icons)]
                
                yield StreamingEvent(
                    event_type="iteration_progress",
                    content=f"{completion_icon} **{self._get_iteration_completed_message(iteration, language)}**",
                    metadata={"iteration": iteration, "completed": True, "completion_icon": completion_icon}
                )
            
            # Check if we hit max iterations
            if iteration >= self.max_iterations:
                print(f"🔍 DEBUG: Reached max iterations ({self.max_iterations}), stopping conversation")
                
                # Stream max iterations completion
                max_icons = ["⚠️", "⏰", "🔄", "⚡"]
                max_icon = max_icons[iteration % len(max_icons)]
                
                yield StreamingEvent(
                    event_type="iteration_progress",
                    content=f"{max_icon} **{self._get_iteration_max_reached_message(iteration, self.max_iterations, language)}**",
                    metadata={"iteration": iteration, "max_iterations": self.max_iterations, "max_reached": True}
                )
                
                yield StreamingEvent(
                    event_type="warning",
                    content=self._get_max_iterations_warning_message(self.max_iterations, language),
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
            
            # Add all new messages to memory at the end (avoid duplication)
            print(f"🔍 DEBUG: Adding new messages to memory manager")
            
            # Get current memory content for deduplication
            current_memory = agent.memory_manager.get_conversation_history(conversation_id)
            memory_content = {(type(msg).__name__, msg.content) for msg in current_memory if hasattr(msg, 'content')}
            
            new_messages_added = 0
            for message in messages:
                # Skip system messages - they're handled separately above
                if isinstance(message, SystemMessage):
                    print(f"🔍 DEBUG: Skipped system message (handled separately)")
                    continue
                
                # Create a unique identifier for this message
                message_key = (type(message).__name__, message.content if hasattr(message, 'content') else str(message))
                
                # Only add if not already in memory
                if message_key not in memory_content:
                    agent.memory_manager.add_message(conversation_id, message)
                    new_messages_added += 1
                    print(f"🔍 DEBUG: Added new {type(message).__name__} to memory")
                else:
                    print(f"🔍 DEBUG: Skipped duplicate {type(message).__name__}")
            
            print(f"🔍 DEBUG: Added {new_messages_added} new messages to memory")
            
            # Final completion event
            # Get language from agent if available, otherwise default to English
            language = getattr(agent, 'language', 'en')
            
            completion_icons = ["✅", "🎯", "✨", "🏆"]
            completion_icon = completion_icons[0]  # Always use first icon for completion
            
            yield StreamingEvent(
                event_type="completion",
                content=f"{completion_icon} **{self._get_response_completed_message(language)}**",
                metadata={"final_response": True}
            )
            
            # Final iteration progress completion
            final_icons = ["🎉", "✨", "🏆", "🎯"]
            final_icon = final_icons[0]  # Always use first icon for final completion
            
            # Get language from agent if available, otherwise default to English
            language = getattr(agent, 'language', 'en')
            
            yield StreamingEvent(
                event_type="iteration_progress",
                content=f"{final_icon} **{self._get_processing_complete_message(language)}**",
                metadata={"conversation_complete": True, "final": True}
            )
                
        except Exception as e:
            # Get language from agent if available, otherwise default to English
            language = getattr(agent, 'language', 'en')
            
            yield StreamingEvent(
                event_type="error",
                content=self._get_error_message(str(e), language),
                metadata={"error": str(e)}
            )
            
            # Error completion for progress display
            
            error_icons = ["❌", "💥", "⚠️", "🚫"]
            error_icon = error_icons[0]  # Always use first icon for error
            
            yield StreamingEvent(
                event_type="iteration_progress",
                content=f"{error_icon} **{self._get_processing_failed_message(language)}**",
                metadata={"error": True, "final": True}
            )


# Global streaming instance
_native_streaming_instance = None

def get_native_streaming() -> NativeLangChainStreaming:
    """Get the global native streaming instance"""
    global _native_streaming_instance
    if _native_streaming_instance is None:
        _native_streaming_instance = NativeLangChainStreaming()
    return _native_streaming_instance
