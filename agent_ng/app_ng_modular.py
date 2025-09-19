"""
LangChain-Native Gradio App - Modularized
=========================================

A modern Gradio application using pure LangChain patterns with modular tab architecture.
This version uses separate tab modules for better organization and maintainability.

Key Features:
- Modular tab architecture (Chat, Logs, Stats)
- Pure LangChain conversation chains and memory
- Multi-turn conversation support with tool calls
- Real-time streaming with metadata
- Native LangChain tool calling
- Modern Gradio UI with comprehensive monitoring
- Tool usage visualization
- Debug logging and statistics
- Responsive design with custom CSS

Based on LangChain's official documentation and best practices.
"""

import asyncio
import gradio as gr
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
import json
import time
from dataclasses import asdict
# Import configuration with fallback for direct execution
try:
    from agent_ng.agent_config import config, get_language_settings, get_port_settings
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from agent_config import config, get_language_settings, get_port_settings

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Local imports with robust fallback handling
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try absolute imports first (works from root directory)
try:
    from agent_ng.langchain_agent import CmwAgent as NextGenAgent, ChatMessage, get_agent_ng
    from agent_ng.llm_manager import get_llm_manager
    from agent_ng.debug_streamer import get_debug_streamer, get_log_handler, LogLevel, LogCategory
    # from agent_ng.streaming_chat import get_chat_interface  # Module moved to .unused
    from agent_ng.tabs import ChatTab, LogsTab, StatsTab
    from agent_ng.ui_manager import get_ui_manager
    from agent_ng.i18n_translations import create_i18n_instance, get_translation_key, format_translation
    print("✅ Successfully imported all modules using absolute imports")
except ImportError as e1:
    print(f"⚠️ Absolute imports failed: {e1}")
    # Fallback to relative imports (when running as module)
    try:
        from .langchain_agent import CmwAgent as NextGenAgent, ChatMessage, get_agent_ng
        from .llm_manager import get_llm_manager
        from .debug_streamer import get_debug_streamer, get_log_handler, LogLevel, LogCategory
        # from .streaming_chat import get_chat_interface  # Module moved to .unused
        from .tabs import ChatTab, LogsTab, StatsTab
        from .ui_manager import get_ui_manager
        from .i18n_translations import create_i18n_instance, get_translation_key, format_translation
        print("✅ Successfully imported all modules using relative imports")
    except ImportError as e2:
        print(f"❌ Both absolute and relative imports failed:")
        print(f"   Absolute: {e1}")
        print(f"   Relative: {e2}")
        print("⚠️ Running with fallback implementations - some features may not work")
        # Set defaults to prevent further errors
        NextGenAgent = None
        ChatMessage = None
        LogLevel = None
        LogCategory = None
        ChatTab = None
        LogsTab = None
        StatsTab = None


class NextGenApp:
    """LangChain-native Gradio application with modular tab architecture and i18n support"""
    
    def __init__(self, language: str = "en"):
        self.agent: Optional[NextGenAgent] = None
        self.llm_manager = get_llm_manager()
        self.initialization_logs = []
        self.is_initializing = False
        self.initialization_complete = False
        self.session_id = "default"  # LangChain session management
        self._ui_update_needed = False
        self.language = language
        
        # Create i18n instance for the specified language
        self.i18n = create_i18n_instance(language)
        
        # Initialize debug system
        self.debug_streamer = get_debug_streamer("app_ng")
        self.log_handler = get_log_handler("app_ng")
        # self.chat_interface = get_chat_interface("app_ng")  # Dead code - never used
        
        # Initialize UI manager with i18n support
        try:
            self.ui_manager = get_ui_manager(language=language, i18n_instance=self.i18n)
            if not self.ui_manager:
                raise ValueError("UI Manager not available")
        except Exception as e:
            print(f"❌ Failed to initialize UI Manager: {e}")
            self.ui_manager = None
        
        # Initialize tab modules
        self.tabs = {}
        self.tab_instances = {}  # Store tab instances for event handlers
        self.components = {}
        
        # Progress status storage with translation
        self.current_progress_status = get_translation_key("progress_ready", language)
        self.progress_icon_index = 0
        self.progress_icons = ["🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚", "🕛"]
        self.is_processing = False
        
        # Initialize synchronously first, then start async initialization
        self._start_async_initialization()
    
    def _start_async_initialization(self):
        """Start async initialization in a new event loop"""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an event loop, schedule the task
            loop.create_task(self._initialize_agent())
        except RuntimeError:
            # No event loop running, start a new one
            import threading
            def run_async_init():
                asyncio.run(self._initialize_agent())
            
            thread = threading.Thread(target=run_async_init, daemon=True)
            thread.start()
    
    async def _initialize_agent(self):
        """Initialize the agent asynchronously"""
        self.is_initializing = True
        self.debug_streamer.info("Starting agent initialization", LogCategory.INIT)
        self.initialization_logs.append("🚀 " + get_translation_key("logs_initializing", self.language))
        
        try:
            # Initialize agent (uses single provider from AGENT_PROVIDER)
            self.debug_streamer.info("Creating agent instance", LogCategory.INIT)
            self.agent = await get_agent_ng()
            
            # Wait for agent to be ready
            max_wait = 30  # 30 seconds timeout
            wait_time = 0
            while not self.agent.is_ready() and wait_time < max_wait:
                await asyncio.sleep(0.5)
                wait_time += 0.5
                self.debug_streamer.debug(f"Waiting for agent... ({wait_time:.1f}s)", LogCategory.INIT)
                self.initialization_logs.append(f"⏳ Ожидание агента... ({wait_time:.1f}s)" if self.language == "ru" else f"⏳ Waiting for agent... ({wait_time:.1f}s)")
            
            if self.agent.is_ready():
                status = self.agent.get_status()
                self.debug_streamer.success(f"Agent ready with {status['current_llm']}", LogCategory.INIT)
                if self.language == "ru":
                    self.initialization_logs.append(f"✅ Агент готов с {status['current_llm']}")
                    tools_msg = format_translation("tools_available", self.language, count=status['tools_count'])
                    self.initialization_logs.append(tools_msg)
                else:
                    self.initialization_logs.append(f"✅ Agent ready with {status['current_llm']}")
                    tools_msg = format_translation("tools_available", self.language, count=status['tools_count'])
                    self.initialization_logs.append(tools_msg)
                self.initialization_complete = True
                
                # Update agent reference in tab instances
                if 'stats' in self.tab_instances:
                    self.tab_instances['stats'].set_agent(self.agent)
                
                # Trigger UI update after agent is ready
                self._trigger_ui_update()
            else:
                self.debug_streamer.error("Agent initialization timeout", LogCategory.INIT)
                self.initialization_logs.append(get_translation_key("error_agent_timeout", self.language))
                
        except Exception as e:
            self.debug_streamer.error(f"Initialization failed: {str(e)}", LogCategory.INIT)
            self.initialization_logs.append(format_translation("error_initialization_failed", self.language, error=str(e)))
        
        self.is_initializing = False
    
    
    
    def is_ready(self) -> bool:
        """Check if the app is ready (LangChain-native pattern)"""
        return self.initialization_complete and self.agent is not None and self.agent.is_ready()
    
    def clear_conversation(self) -> Tuple[List[Dict[str, str]], str]:
        """Clear the conversation history (LangChain-native pattern)"""
        if self.agent:
            self.agent.clear_conversation(self.session_id)
            # Start new conversation tracking for token counting
            if hasattr(self.agent, 'token_tracker'):
                self.agent.token_tracker.start_new_conversation()
            self.debug_streamer.info("Conversation cleared")
        # Reset progress status
        self.current_progress_status = get_translation_key("progress_ready", self.language)
        return [], ""
    
    def get_progress_status(self) -> str:
        """Get the current progress status for the UI"""
        return self.current_progress_status
    
    def update_progress_display(self) -> str:
        """Update the progress display component with rotating icons"""
        if self.is_processing:
            # Always rotate icon during processing to ensure continuous rotation
            self.progress_icon_index = (self.progress_icon_index + 1) % len(self.progress_icons)
            current_icon = self.progress_icons[self.progress_icon_index]
            
            # Check if we have iteration processing status
            if "Iteration" in self.current_progress_status and "Processing..." in self.current_progress_status:
                # Extract iteration info and rebuild with new icon
                import re
                # Match both English and Russian patterns
                match = re.search(r'(\*\*Iteration \d+/\d+\*\* - Processing\.\.\.|\*\*Итерация \d+/\d+\*\* - Обработка\.\.\.)', self.current_progress_status)
                if match:
                    iteration_text = match.group(1)
                    return f"{current_icon} {iteration_text}"
            
            # For any processing status, replace existing icon with rotating one
            import re
            cleaned_status = re.sub(r'^[🕐🕑🕒🕓🕔🕕🕖🕗🕘🕙🕚🕛🔄⚙️🔧⚡] ', '', self.current_progress_status)
            return f"{current_icon} {cleaned_status}"
        
        return self.current_progress_status
    
    def start_processing(self):
        """Mark that processing has started"""
        self.is_processing = True
        self.progress_icon_index = 0
    
    def stop_processing(self):
        """Mark that processing has stopped"""
        self.is_processing = False
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the current conversation history (LangChain-native pattern)"""
        if self.agent:
            return self.agent.get_conversation_history(self.session_id)
        return []
    
    async def chat_with_agent(self, message: str, history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
        """
        Chat with the agent using LangChain-native patterns.
        
        Args:
            message: User message
            history: Chat history as list of message dicts
            
        Returns:
            Updated history and empty message
        """
        if not self.is_ready():
            error_msg = get_translation_key("agent_not_ready", self.language)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""
        
        if not message.strip():
            return history, ""
        
        try:
            self.debug_streamer.info(f"Processing message: {message[:50]}...")
            
            # Process message with LangChain agent
            response = self.agent.process_message(message, self.session_id)
            
            if response.success:
                # Add to history
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response.answer})
                
                # Log tool calls if any
                if response.tool_calls:
                    tool_names = [call['name'] for call in response.tool_calls]
                    tool_calls_msg = format_translation("tool_calls_made", self.language, tool_names=tool_names)
                    self.debug_streamer.info(tool_calls_msg)
                
                return history, ""
            else:
                error_msg = format_translation("error_processing", self.language, error=response.error)
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": error_msg})
                return history, ""
                
        except Exception as e:
            self.debug_streamer.error(f"Error in chat: {e}")
            error_msg = format_translation("error_processing", self.language, error=str(e))
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""
    
    async def stream_chat_with_agent(self, message: str, history: List[Dict[str, str]]) -> AsyncGenerator[Tuple[List[Dict[str, str]], str], None]:
        """
        Stream chat with the agent using LangChain-native streaming patterns.
        
        Args:
            message: User message
            history: Chat history
            progress: Optional Gradio Progress tracker
            
        Yields:
            Updated history and empty message
        """
        if not self.is_ready():
            error_msg = get_translation_key("agent_not_ready", self.language)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            yield history, ""
            return
        
        if not message.strip():
            yield history, ""
            return
        
        # Track execution time
        start_time = time.time()
        
        try:
            self.debug_streamer.info(f"Streaming message: {message[:50]}...")
            
            # Initialize response
            
            # Add user message to history
            working_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
            
            # Get prompt token count for user message (will be displayed below assistant response)
            prompt_tokens = None
            if self.agent:
                try:
                    prompt_tokens = self.agent.count_prompt_tokens_for_chat(history, message)
                except Exception as e:
                    self.debug_streamer.warning(f"Failed to get prompt token count: {e}")
            
            yield working_history, ""
            
            # Stream response using simple streaming
            response_content = ""
            tool_usage = ""
            tool_messages = []  # Store tool messages separately for metadata handling
            
            async for event in self.agent.stream_message(message, self.session_id):
                event_type = event.get("type", "unknown")
                content = event.get("content", "")
                metadata = event.get("metadata", {})
                
                if event_type == "thinking":
                    # Agent is thinking
                    working_history[-1] = {"role": "assistant", "content": content}
                    yield working_history, ""
                    
                elif event_type == "iteration_progress":
                    # Iteration progress - update progress display in sidebar
                    # Store progress status for UI update
                    self.current_progress_status = content

                    # Update progress display instead of adding to chat
                    # This will be handled by the UI update mechanism
                    # For now, just yield the current history without changes
                    yield working_history, ""
                    
                elif event_type == "tool_start":
                    # Tool is starting - create a separate message with metadata
                    tool_name = metadata.get("tool_name", "unknown")
                    tool_title = metadata.get("title", format_translation("tool_called", self.language, tool_name=tool_name))
                    
                    # Create tool message with metadata for collapsible section
                    tool_message = {
                        "role": "assistant", 
                        "content": content,
                        "metadata": {"title": tool_title}
                    }
                    tool_messages.append(tool_message)
                    
                    # Don't add tool usage to main response during streaming - will be added as collapsible sections at the end
                    working_history[-1] = {"role": "assistant", "content": response_content}
                    yield working_history, ""
                    
                elif event_type == "tool_end":
                    # Tool completed - update the last tool message or create new one
                    tool_name = metadata.get("tool_name", "unknown")
                    tool_title = metadata.get("title", format_translation("tool_called", self.language, tool_name=tool_name))
                    
                    # Update the last tool message or create new one
                    if tool_messages and tool_messages[-1].get("metadata", {}).get("title") == tool_title:
                        # Update existing tool message
                        tool_messages[-1]["content"] += content
                    else:
                        # Create new tool message
                        tool_message = {
                            "role": "assistant", 
                            "content": content,
                            "metadata": {"title": tool_title}
                        }
                        tool_messages.append(tool_message)
                    
                    # Don't add tool usage to main response during streaming - will be added as collapsible sections at the end
                    working_history[-1] = {"role": "assistant", "content": response_content}
                    yield working_history, ""
                    
                elif event_type == "content":
                    # Stream content from response
                    response_content += content
                    working_history[-1] = {"role": "assistant", "content": response_content}
                    yield working_history, ""
                    
                elif event_type == "error":
                    # Error occurred
                    error_msg = f"\n{content}"
                    working_history[-1] = {"role": "assistant", "content": response_content + error_msg}
                    yield working_history, ""
            
            # Add API token count to final response
            # Add token counts below assistant response
            token_displays = []
            
            # Add prompt tokens if available
            if prompt_tokens:
                token_displays.append(format_translation("prompt_tokens", self.language, tokens=prompt_tokens.formatted))
            
            # Add API tokens if available
            if self.agent:
                try:
                    print(f"🔍 DEBUG: Getting last API tokens from agent")
                    last_api_tokens = self.agent.get_last_api_tokens()
                    print(f"🔍 DEBUG: Last API tokens: {last_api_tokens}")
                    if last_api_tokens:
                        token_displays.append(format_translation("api_tokens", self.language, tokens=last_api_tokens.formatted))
                        print(f"🔍 DEBUG: Added API token display")
                    else:
                        print("🔍 DEBUG: No API tokens available")
                except Exception as e:
                    print(f"🔍 DEBUG: API token error: {e}")
                    self.debug_streamer.warning(f"Failed to get API token count: {e}")
            
            # Add execution time
            execution_time = time.time() - start_time
            token_displays.append(format_translation("execution_time", self.language, time=execution_time))
            
            # Add deduplication stats if available
            if self.agent and hasattr(self.agent, '_deduplication_stats'):
                dedup_stats = self.agent._deduplication_stats.get(self.session_id, {})
                if dedup_stats:
                    dedup_summary = []
                    total_duplicates = 0
                    total_tool_calls = 0
                    
                    for tool_key, stats in dedup_stats.items():
                        total_tool_calls += stats['total_calls']
                        if stats['duplicates'] > 0:
                            dedup_summary.append(f"{stats['tool_name']}: {stats['duplicates']}")
                            total_duplicates += stats['duplicates']
                    
                    if dedup_summary:
                        # Show per-tool breakdown
                        per_tool_breakdown = ", ".join(dedup_summary)
                        token_displays.append(format_translation("deduplication", self.language, 
                                                               duplicates=total_duplicates, 
                                                               breakdown=per_tool_breakdown))
                        print(f"🔍 DEBUG: Added deduplication stats: {total_duplicates} duplicates")
                    
                    # Add total tool calls count
                    if total_tool_calls > 0:
                        token_displays.append(format_translation("total_tool_calls", self.language, calls=total_tool_calls))
                        print(f"🔍 DEBUG: Added total tool calls: {total_tool_calls}")
            
            # Combine all token displays
            if token_displays:
                token_display = "\n\n" + "\n".join(token_displays)
                # Preserve existing content (including any error messages) and append token display
                current_content = working_history[-1].get("content", response_content)
                working_history[-1] = {"role": "assistant", "content": current_content + token_display}
                print(f"🔍 DEBUG: Added token display: {token_display}")
            
            # Add tool messages with metadata after the main response
            if tool_messages:
                # Insert tool messages before the final assistant response
                final_response = working_history[-1]
                working_history = working_history[:-1]  # Remove the final response temporarily
                
                # Add tool messages with metadata
                for tool_msg in tool_messages:
                    working_history.append(tool_msg)
                
                # Add back the final response
                working_history.append(final_response)
            
            # Stop processing state
            self.stop_processing()
            
            # Final yield with updated stats
            yield working_history, ""
            
        except Exception as e:
            self.debug_streamer.error(f"Error in stream chat: {e}")
            execution_time = time.time() - start_time
            error_msg = format_translation("error_streaming", self.language, error=str(e)) + f"\n\n{format_translation('execution_time', self.language, time=execution_time)}"
            # Preserve existing content and append error message
            current_content = working_history[-1].get("content", "")
            working_history[-1] = {"role": "assistant", "content": current_content + "\n\n" + error_msg}
            # Stop processing state on error
            self.stop_processing()
            yield working_history, ""
    
    def _create_event_handlers(self) -> Dict[str, Any]:
        """Create event handlers for all tabs"""
        handlers = {
            # Chat handlers (core functionality)
            "stream_message": self._stream_message_wrapper,
            "clear_chat": self.clear_conversation,
            
            # Status and monitoring handlers
            "update_status": self._update_status,
            "update_token_budget": self._update_token_budget,
            "refresh_logs": self._refresh_logs,
            "refresh_stats": self._refresh_stats,
            "update_all_ui": self.update_all_ui_components,
            "trigger_ui_update": self.trigger_ui_update,
            "get_progress_status": self.get_progress_status,
            "update_progress_display": self.update_progress_display,
            
            # LLM selection handlers - removed auto-refresh handler
            # LLM selection components update only when explicitly triggered
        }
        
        # Add language switch handler if this is the language detection app
        
        return handlers
    
    def _stream_message_wrapper(self, message: str, history: List[Dict[str, str]]):
        """Stream a message to the agent (synchronous wrapper)"""
        if not message.strip():
            yield history, ""
            return
        
        # Start processing state for icon rotation
        self.start_processing()
        
        # Use the existing event loop or create a new one
        try:
            loop = asyncio.get_running_loop()
            # We're in an event loop, need to run in a thread
            import concurrent.futures
            import threading
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    async def async_stream():
                        async for result in self.stream_chat_with_agent(message, history):
                            yield result
                    
                    # Convert async generator to regular generator
                    async_gen = async_stream()
                    results = []
                    while True:
                        try:
                            result = new_loop.run_until_complete(async_gen.__anext__())
                            results.append(result)
                        except StopAsyncIteration:
                            break
                    return results
                finally:
                    new_loop.close()
            
            # Run in thread and collect all results
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                results = future.result()
                
            # Yield all results
            for result in results:
                yield result
                
        except RuntimeError:
            # No event loop running, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def async_stream():
                    async for result in self.stream_chat_with_agent(message, history):
                        yield result
                
                # Convert async generator to regular generator
                async_gen = async_stream()
                while True:
                    try:
                        result = loop.run_until_complete(async_gen.__anext__())
                        yield result
                    except StopAsyncIteration:
                        break
                
            finally:
                loop.close()
        
        # Refresh UI after streaming completes (EVENT-DRIVEN)
        self._refresh_ui_after_message()
    
    def _update_status(self) -> str:
        """Update status display - delegates to stats tab"""
        stats_tab = self.tab_instances.get('stats')
        if stats_tab and hasattr(stats_tab, 'format_stats_display'):
            return stats_tab.format_stats_display()
        
        # Fallback status
        if self.agent and self.agent.is_ready():
            return get_translation_key("agent_ready", self.language)
        else:
            return get_translation_key("agent_initializing", self.language)
    
    def _update_token_budget(self) -> str:
        """Update token budget display - delegates to chat tab"""
        chat_tab = self.tab_instances.get('chat')
        if chat_tab and hasattr(chat_tab, 'format_token_budget_display'):
            return chat_tab.format_token_budget_display()
        
        # Fallback token budget
        return get_translation_key("token_budget_initializing", self.language)
    
    
    def _refresh_logs(self) -> str:
        """Refresh logs display - delegates to logs tab"""
        logs_tab = self.tab_instances.get('logs')
        if logs_tab and hasattr(logs_tab, 'get_initialization_logs'):
            return logs_tab.get_initialization_logs()
        
        # Fallback logs
        return "\n".join(self.initialization_logs) if self.initialization_logs else "No logs available"
    
    def _refresh_stats(self) -> str:
        """Refresh stats display - delegates to stats tab"""
        stats_tab = self.tab_instances.get('stats')
        if stats_tab and hasattr(stats_tab, 'format_stats_display'):
            return stats_tab.format_stats_display()
        
        # Fallback stats
        if self.agent and self.agent.is_ready():
            return get_translation_key("agent_ready", self.language)
        else:
            return get_translation_key("agent_initializing", self.language)
    
    def _trigger_ui_update(self):
        """Trigger UI update after agent initialization or message processing"""
        try:
            print("🔍 DEBUG: Triggering UI update...")
            # Store the update trigger - the UI will check this
            self._ui_update_needed = True
        except Exception as e:
            print(f"🔍 DEBUG: Error triggering UI update: {e}")
    
    def check_and_clear_ui_update(self) -> bool:
        """Check if UI update is needed and clear the flag"""
        if self._ui_update_needed:
            self._ui_update_needed = False
            return True
        return False
    
    def update_all_ui_components(self) -> Tuple[str, str, str]:
        """Update all UI components and return their values"""
        status = self._update_status()
        stats = self._refresh_stats()
        logs = self._refresh_logs()
        return status, stats, logs
    
    def trigger_ui_update(self):
        """Trigger UI update after agent initialization or message processing"""
        try:
            print("🔍 DEBUG: Triggering UI update...")
            # This will be called when the agent is ready or after messages
            # The actual UI update will happen through Gradio's event system
            self._ui_update_needed = True
        except Exception as e:
            print(f"🔍 DEBUG: Error triggering UI update: {e}")
    
    def _refresh_ui_after_message(self):
        """Refresh all UI components after a message is processed (EVENT-DRIVEN)"""
        try:
            # Trigger UI update after message processing
            self.trigger_ui_update()
            
            print("🔍 DEBUG: UI refreshed after message completion (EVENT-DRIVEN)")
            
        except Exception as e:
            print(f"🔍 DEBUG: Error refreshing UI after message: {e}")
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface using UI Manager and modular tabs"""
        # Validate UI Manager
        if not self.ui_manager:
            raise RuntimeError("UI Manager not available - cannot create interface")
        
        # Create event handlers
        event_handlers = self._create_event_handlers()
        
        # Create tab modules with error handling
        tab_modules = []
        try:
            if ChatTab:
                chat_tab = ChatTab(event_handlers, language=self.language, i18n_instance=self.i18n)
                chat_tab.set_main_app(self)  # Set reference to main app
                tab_modules.append(chat_tab)
                self.tab_instances['chat'] = chat_tab
            else:
                print("⚠️ ChatTab not available")
                
            if LogsTab:
                logs_tab = LogsTab(event_handlers, language=self.language, i18n_instance=self.i18n)
                logs_tab.set_main_app(self)  # Pass main app reference
                tab_modules.append(logs_tab)
                self.tab_instances['logs'] = logs_tab
            else:
                print("⚠️ LogsTab not available")
                
            if StatsTab:
                stats_tab = StatsTab(event_handlers, language=self.language, i18n_instance=self.i18n)
                stats_tab.set_agent(self.agent)  # Pass agent reference
                tab_modules.append(stats_tab)
                self.tab_instances['stats'] = stats_tab
            else:
                print("⚠️ StatsTab not available")
        except Exception as e:
            print(f"❌ Error creating tab modules: {e}")
            raise
        
        # Use UI Manager to create interface
        try:
            demo = self.ui_manager.create_interface(tab_modules, event_handlers)
        except Exception as e:
            print(f"❌ Error creating interface: {e}")
            raise
        
        # Consolidate all components from UI Manager (single source of truth)
        self.components = self.ui_manager.get_components()
        
        # Set agent reference on all tabs that support it
        if self.agent:
            self.ui_manager.set_agent(self.agent)
        
        return demo


class NextGenAppWithLanguageDetection(NextGenApp):
    """LangChain-native Gradio application with dynamic language detection and switching"""
    
    def __init__(self, language: str = "en"):
        super().__init__(language)
        self.current_language = language
        self.supported_languages = ["en", "ru"]
    
    def detect_language_from_url(self):
        """Detect language from URL parameters using environment variables or sys.argv"""
        try:
            # Check if we're running with Gradio and can access URL parameters
            import os
            import sys
            
            # Check for language parameter in command line arguments
            for i, arg in enumerate(sys.argv):
                if arg == '--lang' and i + 1 < len(sys.argv):
                    lang = sys.argv[i + 1].lower()
                    if lang in self.supported_languages:
                        print(f"🌐 Language detected from command line: {lang}")
                        return lang
                elif arg.startswith('--lang='):
                    lang = arg.split('=')[1].lower()
                    if lang in self.supported_languages:
                        print(f"🌐 Language detected from command line: {lang}")
                        return lang
            
            # Check environment variable (can be set by Gradio)
            lang_env = os.environ.get('GRADIO_LANG', '').lower()
            if lang_env in self.supported_languages:
                print(f"🌐 Language detected from environment: {lang_env}")
                return lang_env
            
            # Check for URL parameter in environment (Gradio might set this)
            url_lang = os.environ.get('LANG_PARAM', '').lower()
            if url_lang in self.supported_languages:
                print(f"🌐 Language detected from URL parameter: {url_lang}")
                return url_lang
            
            return None
            
        except Exception as e:
            print(f"⚠️ URL language detection failed: {e}")
            return None
    
    def detect_language(self, request=None):
        """Detect language from URL parameters, headers, or browser settings"""
        # Default to Russian
        detected_lang = "ru"
        
        try:
            # Check URL parameters first
            if request and hasattr(request, 'quick_params'):
                lang_param = request.quick_params.get('lang', '').lower()
                if lang_param in self.supported_languages:
                    detected_lang = lang_param
                    print(f"🌐 Language detected from URL parameter: {detected_lang}")
                    return detected_lang
            
            # Check Accept-Language header
            if request and hasattr(request, 'headers'):
                accept_lang = request.headers.get('Accept-Language', '')
                if 'ru' in accept_lang.lower():
                    detected_lang = "ru"
                    print(f"🌐 Language detected from browser: {detected_lang}")
                    return detected_lang
            
            # Fallback to default
            print(f"🌐 Using default language: {detected_lang}")
            return detected_lang
            
        except Exception as e:
            print(f"⚠️ Language detection failed: {e}, using default: {detected_lang}")
            return detected_lang
    
    


# Global demo variable for single port architecture
demo = None

def get_demo_with_language_detection():
    """Get or create the demo interface with language detection support"""
    global demo
    
    if demo is None:
        try:
            # Create app with language detection capability
            app = NextGenAppWithLanguageDetection()
            demo = app.create_interface()
            # Ensure the demo has the required attributes for Gradio reloading
            if not hasattr(demo, '_queue'):
                demo._queue = None
        except Exception as e:
            print(f"❌ Error creating demo: {e}")
            # Create a minimal working demo with required attributes
            with gr.Blocks() as fallback_demo:
                gr.Markdown("# CMW Platform Agent")
                gr.Markdown("Application is initializing...")
            # Ensure required attributes exist
            if not hasattr(fallback_demo, '_queue'):
                fallback_demo._queue = None
            demo = fallback_demo
    
    return demo

def get_demo(language: str = "en"):
    """Get or create the demo interface for the specified language (legacy support)"""
    # For backward compatibility, create language-specific demo
    try:
        app = NextGenApp(language=language)
        return app.create_interface()
    except Exception as e:
        print(f"❌ Error creating demo for language {language}: {e}")
        # Create a minimal working demo to prevent KeyError
        with gr.Blocks() as demo:
            gr.Markdown("# CMW Platform Agent")
            gr.Markdown("Application is initializing...")
        return demo

# Create a safe demo instance for Gradio reloading
# This prevents the _queue attribute error by ensuring demo is always valid
def create_safe_demo():
    """Create a safe demo instance that won't cause reload errors"""
    try:
        demo_instance = get_demo("en")  # Default to English
        # Ensure the demo has the required attributes for Gradio reloading
        if not hasattr(demo_instance, '_queue'):
            demo_instance._queue = None
        return demo_instance
    except Exception as e:
        print(f"❌ Error creating safe demo: {e}")
        # Create a minimal working demo with required attributes
        with gr.Blocks() as demo:
            gr.Markdown("# CMW Platform Agent")
            gr.Markdown("Application is initializing...")
        # Ensure required attributes exist
        if not hasattr(demo, '_queue'):
            demo._queue = None
        return demo

# Initialize demo for Gradio reloading - use language detection
demo = get_demo_with_language_detection()

def reload_demo():
    """Reload the demo for Gradio hot reloading"""
    global demo
    try:
        print("🔄 Reloading demo...")
        demo = get_demo_with_language_detection()
        return demo
    except Exception as e:
        print(f"❌ Error reloading demo: {e}")
        # Return current demo or create minimal one
        if demo is None:
            with gr.Blocks() as fallback_demo:
                gr.Markdown("# CMW Platform Agent")
                gr.Markdown("Error during reload - please restart the application")
            return fallback_demo
        return demo

def find_available_port(start_port=7860, max_attempts=10):
    """Find an available port starting from start_port"""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    return None

def main():
    """Main function to run the application with single port and language detection"""
    import argparse
    import sys
    import os
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CMW Platform Agent')
    parser.add_argument('-en', '--english', action='store_true', help='Start in English')
    parser.add_argument('-ru', '--russian', action='store_true', help='Start in Russian')
    parser.add_argument('-p', '--port', type=int, default=None, help='Port to run on (overrides config)')
    parser.add_argument('--auto-port', action='store_true', help='Automatically find an available port')
    parser.add_argument('--config', action='store_true', help='Show current configuration')
    args = parser.parse_args()
    
    # Show configuration if requested
    if args.config:
        config.print_config()
        return
    
    # Get language settings from central config
    language_settings = get_language_settings()
    port_settings = get_port_settings()
    
    # Determine language from command line, environment variable, or config default
    # Priority: Command line > Environment variable > Config default
    language = language_settings['default_language']
    
    if args.russian:
        language = "ru"
    elif args.english:
        language = "en"
    
    # Determine port
    default_port = args.port if args.port is not None else port_settings['default_port']
    
    if args.auto_port:
        port = find_available_port(default_port, port_settings['auto_port_range'])
        if port is None:
            print(f"❌ Could not find an available port starting from {default_port}")
            sys.exit(1)
    else:
        port = default_port
    
    print(f"🚀 Starting LangChain-Native LLM Agent App with language detection...")
    print(f"🌐 Language: {language.upper()}")
    print(f"🔌 Port: {port}")
    
    # Create app with specified language
    app = NextGenAppWithLanguageDetection(language=language)
    demo = app.create_interface()
    
    print(f"🌐 Launching Gradio interface on port {port} with language switching...")
    demo.launch(
        debug=True,
        share=False,
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )


if __name__ == "__main__":
    main()
