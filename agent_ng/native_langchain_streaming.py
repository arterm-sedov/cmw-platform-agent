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

from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
from dataclasses import dataclass

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage


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
    
    # Get UI icons from i18n translations
    def _get_ui_icons(self, language: str = "en"):
        """Get UI icons from i18n translations"""
        from .i18n_translations import get_translation_key
        return {
            'finish_icons': get_translation_key("finish_icons", language),
            'completion_icons': get_translation_key("completion_icons", language),
            'max_icons': get_translation_key("max_icons", language),
            'completion_final_icons': get_translation_key("completion_final_icons", language),
            'error_icons': get_translation_key("error_icons", language)
        }
    
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
        # Safety check for None language
        if language is None:
            language = "en"
        template = get_translation_key("tool_called", language)
        return template.format(tool_name=tool_name)
    
    def _get_call_count_message(self, total_calls: int, language: str = "en") -> str:
        """Get the localized call count message"""
        from .i18n_translations import get_translation_key
        # Safety check for None language
        if language is None:
            language = "en"
        template = get_translation_key("call_count", language)
        return template.format(total_calls=total_calls)
    
    def _get_result_message(self, tool_result: str, language: str = "en") -> str:
        """Get the localized result message with chat display truncation"""
        from .i18n_translations import get_translation_key
        # Safety check for None language
        if language is None:
            language = "en"
        template = get_translation_key("result", language)
        
        # Lean truncation for chat display only (200 chars max)
        display_result = tool_result
        if len(tool_result) > 400:
            display_result = tool_result[:400] + "... [truncated]"
        
        return template.format(tool_result=display_result)
    
    def _deduplicate_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Deduplicate tool calls and count duplicates BEFORE execution.
        
        Args:
            tool_calls: List of tool calls from LLM response
            
        Returns:
            Tuple of (deduplicated_tool_calls, duplicate_counts)
        """
        unique_tool_calls = []
        duplicate_counts = {}
        
        for tool_call in tool_calls:
            if not tool_call:
                continue
            tool_name = tool_call.get('name', 'unknown')
            tool_args = tool_call.get('args', {})
            tool_key = f"{tool_name}:{hash(str(sorted(tool_args.items())))}"
            
            if tool_key in duplicate_counts:
                # Increment count for duplicate
                duplicate_counts[tool_key] += 1
                print(f"🔍 DEBUG: Found duplicate tool call {tool_name} (total count: {duplicate_counts[tool_key]})")
            else:
                # First occurrence - add to unique list and initialize count
                unique_tool_calls.append(tool_call)
                duplicate_counts[tool_key] = 1
                print(f"🔍 DEBUG: Added unique tool call {tool_name}")
        
        return unique_tool_calls, duplicate_counts

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
    
    def _get_agent_language(self, agent) -> str:
        """Get language from agent, defaulting to English"""
        if agent is None:
            return 'en'
        return getattr(agent, 'language', 'en')
    
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
            print(f"🔍 DEBUG: Streaming manager - chat_history length: {len(chat_history) if chat_history else 0}")
            
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
            # Filter out orphaned tool messages to prevent message order issues
            non_system_history = []
            for i, msg in enumerate(chat_history):
                # Safety check for None message
                if msg is None:
                    print(f"🔍 DEBUG: Skipping None message at index {i}")
                    continue
                    
                if isinstance(msg, SystemMessage):
                    continue
                elif isinstance(msg, ToolMessage):
                    # Only include tool messages that have a corresponding AI message with tool calls
                    for j in range(i-1, -1, -1):
                        # Safety check for None message in history
                        if chat_history[j] is None:
                            continue
                        if (isinstance(chat_history[j], AIMessage) and 
                            hasattr(chat_history[j], 'tool_calls') and chat_history[j].tool_calls):
                            # Safety check for None tool calls
                            tool_call_ids = {tc.get('id') for tc in chat_history[j].tool_calls if tc is not None and tc.get('id')}
                            if hasattr(msg, 'tool_call_id') and msg.tool_call_id in tool_call_ids:
                                non_system_history.append(msg)
                                break
                else:
                    non_system_history.append(msg)
            
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
                
                # Stream iteration progress as separate message (no icon - UI will add rotating clock)
                # Get language from agent
                language = self._get_agent_language(agent)
                
                yield StreamingEvent(
                    event_type="iteration_progress",
                    content=self._get_iteration_processing_message(iteration, self.max_iterations, language),
                    metadata={"iteration": iteration, "max_iterations": self.max_iterations}
                )
                
                # Use proper LangChain streaming with tool call handling
                accumulated_chunk = None
                tool_calls_in_progress = {}
                processed_tools = {}  # Track processed tools to avoid duplicates in same response
                has_tool_calls = False
                print(f"🔍 DEBUG: Starting streaming loop for iteration {iteration}")
                
                async for chunk in llm_with_tools.astream(messages):
                    try:
                        # Accumulate chunks for proper tool call parsing
                        if accumulated_chunk is None:
                            accumulated_chunk = chunk
                        else:
                            accumulated_chunk = accumulated_chunk + chunk
                    except Exception as e:
                        print(f"🔍 DEBUG: Error processing chunk: {e}")
                        print(f"🔍 DEBUG: Chunk type: {type(chunk)}")
                        print(f"🔍 DEBUG: Chunk content: {chunk}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
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
                            try:
                                # Safety check for None tool_call_chunk
                                if tool_call_chunk is None:
                                    continue
                                
                                tool_name = tool_call_chunk.get('name') if isinstance(tool_call_chunk, dict) else None
                                tool_call_id = tool_call_chunk.get('id') if isinstance(tool_call_chunk, dict) else None
                                tool_args = tool_call_chunk.get('args', '') if isinstance(tool_call_chunk, dict) else ''
                            except Exception as e:
                                print(f"🔍 DEBUG: Error processing tool_call_chunk: {e}")
                                print(f"🔍 DEBUG: tool_call_chunk type: {type(tool_call_chunk)}")
                                print(f"🔍 DEBUG: tool_call_chunk content: {tool_call_chunk}")
                                import traceback
                                traceback.print_exc()
                                continue
                            
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
                                # Accumulate tool arguments - safety check for None
                                safe_tool_args = tool_args if tool_args is not None else ""
                                tool_calls_in_progress[tool_call_id]['args'] += safe_tool_args
                
                # Process completed tool calls
                if hasattr(accumulated_chunk, 'tool_calls') and accumulated_chunk.tool_calls:
                    has_tool_calls = True
                    print(f"🔍 DEBUG: Found {len(accumulated_chunk.tool_calls)} tool calls")
                    
                    # STEP 1: Filter and count duplicates BEFORE execution
                    deduplicated_tool_calls, duplicate_counts = self._deduplicate_tool_calls(accumulated_chunk.tool_calls)
                    print(f"🔍 DEBUG: Original tool calls: {len(accumulated_chunk.tool_calls)}, Deduplicated: {len(deduplicated_tool_calls)}")
                    
                    # STEP 2: Execute only deduplicated tool calls and create result mapping
                    tool_result_cache = {}  # Map tool_key -> result for duplicate handling
                    
                    for tool_call in deduplicated_tool_calls:
                        if not tool_call:
                            continue
                        tool_name = tool_call.get('name')
                        tool_args = tool_call.get('args', {})
                        tool_call_id = tool_call.get('id')
                        
                        if tool_name and tool_call_id in tool_calls_in_progress:
                            # Get duplicate count for this tool call - safety check for None tool_args
                            safe_tool_args = tool_args if tool_args is not None else {}
                            tool_key = f"{tool_name}:{hash(str(sorted(safe_tool_args.items())))}"
                            duplicate_count = duplicate_counts.get(tool_key, 1)
                            
                            # Find the tool in our tools list
                            tool_obj = None
                            for tool in agent.tools:
                                if tool.name == tool_name:
                                    tool_obj = tool
                                    break
                            
                            try:
                                if tool_obj:
                                    # Execute tool once with agent context
                                    # Inject agent instance for file resolution (same as memory manager)
                                    tool_args_with_agent = {**safe_tool_args, 'agent': agent}
                                    tool_result = tool_obj.invoke(tool_args_with_agent)
                                    
                                    # Safety check for None tool_result
                                    if tool_result is None:
                                        print(f"⚠️ Tool {tool_name} returned None result")
                                        tool_result = "Tool execution completed but returned no result"
                                    
                                    # Store the tool call result for future deduplication
                                    from .tool_deduplicator import get_deduplicator
                                    deduplicator = get_deduplicator()
                                    deduplicator.store_tool_call(tool_name, safe_tool_args, tool_result, conversation_id)
                                    
                                    # Cache result for duplicate tool calls
                                    tool_result_cache[tool_key] = tool_result
                                    
                                    # CRITICAL: Stream tool completion only ONCE per unique tool
                                    # Show duplicate count in the message but don't stream multiple times
                                    
                                    # Safety check for None tool_result to prevent concatenation errors
                                    safe_tool_result = tool_result if tool_result is not None else "Tool execution returned no result"
                                    
                                    yield StreamingEvent(
                                        event_type="tool_end",
                                        content=f"\n{self._get_call_count_message(duplicate_count, language)}\n\n{self._get_result_message(str(safe_tool_result), language)}",
                                        metadata={
                                            "tool_name": tool_name,
                                            "tool_output": str(safe_tool_result),
                                            "duplicate": duplicate_count > 1,
                                            "duplicate_count": duplicate_count,
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
                    
                    # CRITICAL: Add AI message with tool_calls to working messages FIRST
                    if accumulated_chunk and hasattr(accumulated_chunk, 'tool_calls') and accumulated_chunk.tool_calls:
                        # Add the AIMessage with tool_calls to the working messages
                        ai_message_with_tool_calls = AIMessage(
                            content=accumulated_chunk.content or "",
                            tool_calls=accumulated_chunk.tool_calls
                        )
                        messages.append(ai_message_with_tool_calls)
                        print(f"🔍 DEBUG: Added AIMessage with {len(accumulated_chunk.tool_calls)} tool calls to working messages")
                        
                        # STEP 3: Create ToolMessage for EACH original tool call (including duplicates)
                        # This ensures proper tool_call_id mapping and message sequence
                        # CRITICAL: Add ToolMessages to working messages for proper sequence
                        tool_messages = []
                        for original_tool_call in accumulated_chunk.tool_calls:
                            if not original_tool_call:
                                continue
                            tool_name = original_tool_call.get('name')
                            tool_args = original_tool_call.get('args', {})
                            tool_call_id = original_tool_call.get('id')
                            
                            if tool_name and tool_call_id:
                                # Get the tool key for result lookup (exclude agent from key calculation)
                                cache_args = {k: v for k, v in tool_args.items() if k != 'agent'}
                                tool_key = f"{tool_name}:{hash(str(sorted(cache_args.items())))}"
                                
                                # Get cached result (same for all duplicates)
                                tool_result = tool_result_cache.get(tool_key, "Tool execution failed")
                                
                                # Create ToolMessage with original tool_call_id
                                tool_message = ToolMessage(
                                    content=str(tool_result),
                                    tool_call_id=tool_call_id,
                                    name=tool_name
                                )
                                tool_messages.append(tool_message)
                                print(f"🔍 DEBUG: Created ToolMessage for {tool_name} with ID {tool_call_id}")
                        
                        # CRITICAL: Add ToolMessages to working messages for next LLM call
                        # This ensures proper sequence: AIMessage(with tool_calls) → ToolMessages
                        messages.extend(tool_messages)
                        print(f"🔍 DEBUG: Added {len(tool_messages)} ToolMessages to working messages")
                    
                    # CRITICAL: Continue to next iteration to get final response
                    # Don't break here - we need the final AI response after tool calls
                elif accumulated_chunk and hasattr(accumulated_chunk, 'content') and accumulated_chunk.content:
                    # Regular AI message without tool calls
                    ai_message = AIMessage(content=accumulated_chunk.content)
                    messages.append(ai_message)
                    print(f"🔍 DEBUG: Added regular AIMessage to working messages")
                
                # If no tool calls, we're done
                if not has_tool_calls:
                    print(f"🔍 DEBUG: No tool calls in iteration {iteration}, conversation complete")
                    
                    # Stream conversation completion
                    icons = self._get_ui_icons(language)
                    finish_icon = icons['finish_icons'][iteration % len(icons['finish_icons'])]
                    
                    yield StreamingEvent(
                        event_type="iteration_progress",
                        content=f"{finish_icon} **{self._get_iteration_finished_message(iteration, self.max_iterations, language)}**",
                        metadata={"iteration": iteration, "max_iterations": self.max_iterations, "conversation_complete": True}
                    )
                    break
                
                print(f"🔍 DEBUG: Completed iteration {iteration}, continuing...")
                
                # Stream iteration completion as separate message
                icons = self._get_ui_icons(language)
                completion_icon = icons['completion_icons'][iteration % len(icons['completion_icons'])]
                
                yield StreamingEvent(
                    event_type="iteration_progress",
                    content=f"{completion_icon} **{self._get_iteration_completed_message(iteration, language)}**",
                    metadata={"iteration": iteration, "completed": True, "completion_icon": completion_icon}
                )
            
            # Check if we hit max iterations
            if iteration >= self.max_iterations:
                print(f"🔍 DEBUG: Reached max iterations ({self.max_iterations}), stopping conversation")
                
                # Stream max iterations completion
                icons = self._get_ui_icons(language)
                max_icon = icons['max_icons'][iteration % len(icons['max_icons'])]
                
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
                
                # Skip empty AIMessages - they cause message order issues
                if isinstance(message, AIMessage) and not message.content and not message.tool_calls:
                    print(f"🔍 DEBUG: Skipped empty AIMessage (no content or tool calls)")
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
            language = self._get_agent_language(agent)
            icons = self._get_ui_icons(language)
            completion_icon = icons['completion_final_icons'][0]  # Always use first icon for completion
            
            yield StreamingEvent(
                event_type="completion",
                content=f"{completion_icon} **{self._get_response_completed_message(language)}**",
                metadata={"final_response": True}
            )
            
            # Final iteration progress completion
            language = self._get_agent_language(agent)
            
            yield StreamingEvent(
                event_type="iteration_progress",
                content=self._get_processing_complete_message(language),
                metadata={"conversation_complete": True, "final": True}
            )
                
        except Exception as e:
            # Get language from agent
            language = self._get_agent_language(agent)
            
            yield StreamingEvent(
                event_type="error",
                content=self._get_error_message(str(e), language),
                metadata={"error": str(e)}
            )
            
            # Error completion for progress display
            icons = self._get_ui_icons(language)
            error_icon = icons['error_icons'][0]  # Always use first icon for error
            
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
