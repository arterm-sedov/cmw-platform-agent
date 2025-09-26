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
import logging
import gradio as gr
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
import json
import time
from dataclasses import asdict

# Initialize logging early (idempotent)
try:
    from agent_ng.logging_config import setup_logging
    setup_logging()
    _logger = logging.getLogger(__name__)
except Exception:
    _logger = logging.getLogger(__name__)
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
    from agent_ng.langchain_agent import CmwAgent as NextGenAgent, ChatMessage
    from agent_ng.llm_manager import get_llm_manager
    from agent_ng.debug_streamer import get_debug_streamer, get_log_handler, LogLevel, LogCategory
    # from agent_ng.streaming_chat import get_chat_interface  # Module moved to .unused
    from agent_ng.tabs import ChatTab, LogsTab, StatsTab
    from agent_ng.ui_manager import get_ui_manager
    from agent_ng.utils import safe_string
    from agent_ng.i18n_translations import create_i18n_instance, get_translation_key, format_translation
    from langsmith import traceable
    _logger.info("Successfully imported all modules using absolute imports")
except ImportError as e1:
    _logger.warning("Absolute imports failed: %s", e1)
    # Fallback to relative imports (when running as module)
    try:
        from .langchain_agent import CmwAgent as NextGenAgent, ChatMessage
        from .llm_manager import get_llm_manager
        from .debug_streamer import get_debug_streamer, get_log_handler, LogLevel, LogCategory
        # from .streaming_chat import get_chat_interface  # Module moved to .unused
        from .tabs import ChatTab, LogsTab, StatsTab
        from .ui_manager import get_ui_manager
        from .i18n_translations import create_i18n_instance, get_translation_key, format_translation
        from langsmith import traceable
        _logger.info("Successfully imported all modules using relative imports")
    except ImportError as e2:
        _logger.critical("Both absolute and relative imports failed. Absolute: %s | Relative: %s", e1, e2)
        _logger.error("Please check: 1) requirements_ng.txt installed 2) PYTHONPATH 3) circular imports 4) modules exist 5) working directory")
        raise ImportError(f"Failed to import required modules. Absolute: {e1}, Relative: {e2}")


class NextGenApp:
    """LangChain-native Gradio application with modular tab architecture and i18n support"""
    
    def __init__(self, language: str = "en"):
        # No global agent - only session-specific agents
        self.llm_manager = get_llm_manager()
        self.initialization_logs = []
        self.is_initializing = False
        self.initialization_complete = False
        # REMOVED: self.session_id = "default"  # This was causing data leakage between users!
        self._ui_update_needed = False
        self.language = language
        # Session-aware turn snapshots for analytics/logs (non-persistent)
        self.session_turn_snapshots = {}  # session_id -> turn_snapshot
        
        # Initialize concurrency management
        try:
            from .concurrency_config import get_concurrency_config
            from .queue_manager import create_queue_manager
            self.concurrency_config = get_concurrency_config()
            self.queue_manager = create_queue_manager(self.concurrency_config)
        except ImportError:
            # Fallback for when running as script
            from concurrency_config import get_concurrency_config
            from queue_manager import create_queue_manager
            self.concurrency_config = get_concurrency_config()
            self.queue_manager = create_queue_manager(self.concurrency_config)
        
        # Session Management - Clean modular approach
        try:
            from .session_manager import SessionManager
        except ImportError:
            # Fallback for when running as script
            from session_manager import SessionManager
        self.session_manager = SessionManager(language)
        
        # Create i18n instance for the specified language
        self.i18n = create_i18n_instance(language)
        
        # Initialize debug system - use a default session that will be accessible
        self.debug_streamer = get_debug_streamer("default")
        self.log_handler = get_log_handler("default")
        
        # Initialize session-aware logging
        try:
            from .debug_streamer import set_session_context
        except ImportError:
            # Fallback for when running as script
            from debug_streamer import set_session_context
        self.set_session_context = set_session_context
        # self.chat_interface = get_chat_interface("app_ng")  # Dead code - never used
        
        # Initialize UI manager with i18n support
        try:
            self.ui_manager = get_ui_manager(language=language, i18n_instance=self.i18n)
            if not self.ui_manager:
                raise ValueError("UI Manager not available")
        except Exception as e:
            _logger.exception("Failed to initialize UI Manager: %s", e)
            self.ui_manager = None
        
        # Initialize tab modules
        self.tabs = {}
        self.tab_instances = {}  # Store tab instances for event handlers
        self.components = {}
        
        # Progress status storage with translation
        # Progress status is now managed per-session through session manager
        self.is_processing = False
        self.current_global_progress = get_translation_key("progress_ready", language)  # Global progress for timer updates
        self.last_progress_display = ""  # Cache last display to avoid unnecessary updates
        
        # Persistent background asyncio loop for streaming to avoid closing gRPC/Gemini loop between turns
        self._stream_loop = None
        self._stream_loop_thread = None
        self._stream_queue_timeout_s = 0.1
        self._ensure_stream_loop()
        
        # Initialize synchronously first, then start async initialization
        self._start_async_initialization()
    
    def get_user_session_id(self, request: gr.Request = None) -> str:
        """Get session ID using clean session manager"""
        return self.session_manager.get_session_id(request)
    
    def get_user_agent(self, session_id: str) -> NextGenAgent:
        """Get agent instance using clean session manager"""
        return self.session_manager.get_agent(session_id)
    
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
    
    def _ensure_stream_loop(self):
        """Ensure a persistent background event loop exists and is running."""
        try:
            if self._stream_loop is not None and self._stream_loop.is_running():
                return
        except Exception:
            # Recreate loop if previous loop errored out
            self._stream_loop = None
            self._stream_loop_thread = None
        
        import threading
        import asyncio
        self._stream_loop = asyncio.new_event_loop()
        
        def _run_loop_forever():
            asyncio.set_event_loop(self._stream_loop)
            self._stream_loop.run_forever()
        
        self._stream_loop_thread = threading.Thread(target=_run_loop_forever, daemon=True)
        self._stream_loop_thread.start()
    
    def _submit_stream_task(self, message, history, request, out_queue):
        """Submit the async streaming producer to the background loop and feed results into out_queue."""
        import asyncio
        
        async def _producer():
            try:
                async for result in self.stream_chat_with_agent(message, history, request):
                    out_queue.put(result)
            except Exception as e:
                # Propagate errors to the consumer
                out_queue.put({"type": "error", "content": f"{e}", "metadata": {"error": str(e)}})
            finally:
                # Sentinel to indicate completion
                out_queue.put(StopIteration)
        
        return asyncio.run_coroutine_threadsafe(_producer(), self._stream_loop)
    
    async def _initialize_agent(self):
        """Initialize the session manager (no global agent needed)"""
        self.is_initializing = True
        self.debug_streamer.info("Starting session manager initialization", LogCategory.INIT)
        self.initialization_logs.append("🚀 " + get_translation_key("logs_initializing", self.language))
        
        try:
            # Initialize session manager (creates agents on-demand per session)
            self.debug_streamer.info("Session manager ready", LogCategory.INIT)
            
            self.initialization_logs.append("✅ " + get_translation_key("session_manager_ready", self.language))
            self.initialization_complete = True
            
            
            # Trigger UI update after initialization
            self._trigger_ui_update()
                
        except Exception as e:
            try:
                self.debug_streamer.error(f"Initialization failed: {str(e)}", LogCategory.INIT)
            except:
                # If debug streamer fails, just continue
                pass
            self.initialization_logs.append(format_translation("error_initialization_failed", self.language, error=str(e)))
        
        self.is_initializing = False
    
    def get_conversation_summary(self, session_id: str, markdown: bool = False) -> str:
        """
        Generate unified conversation summary for logs and download frontmatter.
        
        Args:
            session_id: Session ID to get summary for
            markdown: If True, format for markdown (download). If False, format for plain text (logs).
            
        Returns:
            Formatted conversation summary string
        """
        try:
            from datetime import datetime
            
            # Get session-specific agent and stats
            agent = self.session_manager.get_session_agent(session_id)
            if not agent:
                self.debug_streamer.debug(f"No agent found for session: {session_id}")
                return ""
            
            # Get conversation stats
            stats = agent.get_stats()
            conversation_stats = stats.get("conversation_stats", {})
            llm_info = stats.get("llm_info", {})
            
            # Get tool information from session-aware turn snapshot
            tool_names = set()
            tool_calls_count = 0
            snapshot = self.session_turn_snapshots.get(session_id)
            if snapshot and isinstance(snapshot, list):
                for msg in snapshot:
                    if msg.get('role') == 'tool' and msg.get('name'):
                        tool_names.add(msg['name'])
                    elif msg.get('role') == 'assistant' and msg.get('tool_calls'):
                        tool_calls_count += len(msg['tool_calls'])
            
            # Get roles sequence from snapshot
            roles_seq = []
            if snapshot:
                roles_seq = [m.get("role", "?") for m in snapshot]
            
            # Get provider/model info
            provider = llm_info.get('provider', 'unknown')
            model = llm_info.get('model', 'unknown')
            
            # Format timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Format tool information with total count
            if tool_names:
                tool_counts = {}
                for msg in snapshot:
                    if msg.get('role') == 'tool' and msg.get('name'):
                        tool_name = msg['name']
                        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                tools_text = ", ".join([f"{name} ({count})" for name, count in sorted(tool_counts.items())])
                total_tools = len(tool_counts)
            else:
                tools_text = "None"
                total_tools = 0
            
            # Get all used provider/model combinations from conversation history
            provider_models = set()
            if agent and hasattr(agent, "memory_manager"):
                memory = agent.memory_manager.get_memory(session_id)
                if memory and hasattr(memory, "chat_memory") and hasattr(memory.chat_memory, "messages"):
                    messages = memory.chat_memory.messages
                    for msg in messages:
                        if hasattr(msg, '__class__') and msg.__class__.__name__ == 'AIMessage':
                            # Check if this message has LLM info in metadata
                            if hasattr(msg, 'additional_kwargs') and 'llm_provider' in msg.additional_kwargs:
                                provider = msg.additional_kwargs.get('llm_provider', 'unknown')
                                model = msg.additional_kwargs.get('llm_model', 'unknown')
                                provider_models.add(f"{provider} / {model}")
            
            # If no provider/model info found in messages, use current
            if not provider_models:
                provider_models.add(f"{provider} / {model}")
            
            # Format provider/model information
            if len(provider_models) == 1:
                providers_text = list(provider_models)[0]
                total_models = 1
            else:
                providers_text = ", ".join(sorted(provider_models))
                total_models = len(provider_models)
            
            # Build summary based on format
            if markdown:
                # Markdown format for download
                summary = f"## {self._get_translation('conversation_summary')} ({timestamp})\n\n"
                summary += f"**{self._get_translation('total_messages_label')}:** {conversation_stats.get('message_count', 0)} ({conversation_stats.get('user_messages', 0)} user, {conversation_stats.get('assistant_messages', 0)} assistant)\n\n"
                summary += f"**{self._get_translation('roles_sequence')}:** {' → '.join(roles_seq)}\n\n"
                summary += f"**{self._get_translation('tools_used_total')} ({total_tools}):** {tools_text}\n\n"
                summary += f"**{self._get_translation('providers_models_total')} ({total_models}):** {providers_text}\n\n"
            else:
                # Plain text format for logs
                summary = f"--- {self._get_translation('conversation_summary')} ({timestamp}) ---\n"
                summary += f"Сеанс: {session_id}\n"
                summary += f"{self._get_translation('total_messages_label')}: {conversation_stats.get('message_count', 0)} ({conversation_stats.get('user_messages', 0)} user, {conversation_stats.get('assistant_messages', 0)} assistant)\n"
                summary += f"{self._get_translation('roles_sequence')}: {' → '.join(roles_seq)}\n"
                summary += f"{self._get_translation('tools_used_total')} ({total_tools}): {tools_text}\n"
                summary += f"{self._get_translation('providers_models_total')} ({total_models}): {providers_text}\n"
                summary += f"Total conversation tokens: {total_tokens:,}\n"
                summary += f"Average tokens per conversation: {avg_tokens:,.0f}\n"
            
            # Add newline for logs format
            if not markdown:
                summary += "\n"
            
            return summary
            
        except Exception as e:
            self.debug_streamer.warning(f"Failed to generate conversation summary: {e}")
            return ""
    
    def _get_translation(self, key: str) -> str:
        """Get translation for a key using the current language"""
        from .i18n_translations import get_translation_key
        return get_translation_key(key, self.language)
    
    def is_ready(self) -> bool:
        """Check if the app is ready (session-based pattern)"""
        return self.initialization_complete and self.session_manager is not None
    
    def clear_conversation(self, request: gr.Request = None) -> Tuple[List[Dict[str, str]], str]:
        """Clear the conversation history (LangChain-native pattern) - now properly session-aware"""
        if request:
            # Use clean session manager for session-aware clearing
            session_id = self.session_manager.get_session_id(request)
            self.session_manager.clear_conversation(session_id)
            self.debug_streamer.info(f"Conversation cleared for session: {session_id}")
        else:
            # Fallback for non-session requests - use default session
            session_id = "default"
            if hasattr(self, 'session_manager'):
                session_data = self.session_manager.get_session_data(session_id)
                if session_data and session_data.agent:
                    session_data.agent.clear_conversation(session_id)
                    if hasattr(session_data.agent, 'token_tracker'):
                        session_data.agent.token_tracker.start_new_conversation()
                    self.debug_streamer.info("Conversation cleared (default session)")
        
        # Reset progress status for all sessions
        # Progress status is now managed per-session through session manager
        return [], ""
    
    def get_progress_status(self, request: gr.Request = None) -> str:
        """Get the current progress status for the UI - now properly session-aware"""
        if request:
            session_id = self.session_manager.get_session_id(request)
            return self.session_manager.get_status(session_id)
        # Fallback to default status if no request available
        return get_translation_key("progress_ready", self.language)
    
    def update_progress_display(self, request: gr.Request = None) -> str:
        """Update progress display - session-aware, minimal with caching"""
        if not request:
            # Use global progress status for timer-based updates
            if self.is_processing:
                icon = self.session_manager.get_clock_icon()
                result = f"{icon} {self.current_global_progress}"
            else:
                result = self.current_global_progress
            
            # Only update if content changed to reduce UI blocking
            if result != self.last_progress_display:
                self.last_progress_display = result
                return result
            return self.last_progress_display
        
        session_id = self.session_manager.get_session_id(request)
        status = self.session_manager.get_status(session_id)
        
        # Add rotating clock icon if processing (UI handles rotation)
        if self.is_processing:
            icon = self.session_manager.get_clock_icon()
            result = f"{icon} {status}"
        else:
            result = status
        
        # Only update if content changed to reduce UI blocking
        if result != self.last_progress_display:
            self.last_progress_display = result
            return result
        return self.last_progress_display
    
    
    def start_processing(self):
        """Mark that processing has started"""
        self.is_processing = True
        # Update global progress to show processing started
        self.current_global_progress = get_translation_key("progress_processing", self.language)
        # Reset cache to force update
        self.last_progress_display = ""
    
    def stop_processing(self):
        """Mark that processing has stopped"""
        self.is_processing = False
        # Update global progress to show ready state
        self.current_global_progress = get_translation_key("progress_ready", self.language)
        # Reset cache to force update
        self.last_progress_display = ""
    
    def get_conversation_history(self, session_id: str = "default") -> List[BaseMessage]:
        """Get the current conversation history (session-based pattern)"""
        if hasattr(self, 'session_manager'):
            session_data = self.session_manager.get_session_data(session_id)
            if session_data and session_data.agent:
                return session_data.agent.get_conversation_history(session_id)
        return []
    
    
    async def stream_chat_with_agent(self, message: str, history: List[Dict[str, str]], request: gr.Request = None) -> AsyncGenerator[Tuple[List[Dict[str, str]], str], None]:
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
            # Extract session ID from Gradio request for user isolation
            session_id = self.get_user_session_id(request)
            user_agent = self.get_user_agent(session_id)
            
            # Set session context for logging
            self.set_session_context(session_id)
            
            # Debug: Check which LLM instance is being used
            if user_agent and hasattr(user_agent, 'llm_instance') and user_agent.llm_instance:
                llm_info = user_agent.get_llm_info()
                print(f"🔍 DEBUG: Using session agent with LLM: {llm_info.get('provider', 'unknown')}/{llm_info.get('model_name', 'unknown')}")
            else:
                print(f"❌ DEBUG: Session agent has no LLM instance!")
            
            self.debug_streamer.info(f"Streaming message for session {session_id}: {message[:50]}...")
            
            # Initialize response
            
            # Add user message to history
            working_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
            
            # Get prompt token count for user message (will be displayed below assistant response)
            prompt_tokens = None
            if user_agent:
                try:
                    prompt_tokens = user_agent.count_prompt_tokens_for_chat(history, message)
                except Exception as e:
                    self.debug_streamer.warning(f"Failed to get prompt token count: {e}")
            
            yield working_history, ""
            
            # Stream response using simple streaming
            response_content = ""  # Initialize as empty string to prevent None concatenation
            tool_usage = ""
            assistant_message_index = -1  # Track the index of the assistant message in working_history
            # Tool messages are now added immediately to working_history during streaming
            streaming_error_handled = False  # Flag to track if streaming error was handled
            
            # print(f"🔍 DEBUG: Starting streaming for session {session_id}")
            # print(f"🔍 DEBUG: Working history length before streaming: {len(working_history)}")
            
            try:
                async for event in user_agent.stream_message(message, session_id):
                    # Safety check for None event
                    if event is None:
                        # print("🔍 DEBUG: Received None event, skipping...")
                        continue
                    
                    try:
                        event_type = event.get("type", "unknown")
                        content = event.get("content", "")
                        metadata = event.get("metadata", {})
                        # print(f"🔍 DEBUG: Processing event - type: {event_type}, content length: {len(str(content))}")
                    except Exception as e:
                        import traceback
                        # print(f"🔍 DEBUG: Error processing event: {e}")
                        # print(f"🔍 DEBUG: Event data: {event}")
                        # print(f"🔍 DEBUG: Event type: {type(event)}")
                        # print(f"🔍 DEBUG: Event traceback:")
                        # traceback.print_exc()
                        streaming_error_handled = True
                        # print(f"🔍 DEBUG: Set streaming_error_handled to True in event processing")
                        continue
                    
                    if event_type == "thinking":
                        # Agent is thinking - update or create assistant message
                        if assistant_message_index >= 0 and assistant_message_index < len(working_history):
                            # Update existing assistant message
                            working_history[assistant_message_index] = {"role": "assistant", "content": content}
                        else:
                            # Create new assistant message
                            working_history.append({"role": "assistant", "content": content})
                            assistant_message_index = len(working_history) - 1
                        yield working_history, ""
                        
                    elif event_type == "iteration_progress":
                        # Iteration progress - update progress display in sidebar
                        # Store progress status for UI update - session-specific
                        self.session_manager.set_status(session_id, content)
                        # Also update global progress for timer-based updates
                        self.current_global_progress = content
                        # Reset cache to force update
                        self.last_progress_display = ""
                        yield working_history, ""
                        
                    elif event_type == "completion":
                        # Final completion message - update progress display
                        self.session_manager.set_status(session_id, content)
                        # Also update global progress for timer-based updates
                        self.current_global_progress = content
                        # Reset cache to force update
                        self.last_progress_display = ""
                        yield working_history, ""

                    elif event_type == "turn_complete":
                        # Store session-aware turn snapshot for analytics/logs (non-persistent)
                        ordered = metadata.get("ordered_messages") if metadata else None
                        conversation_id = metadata.get("conversation_id") if metadata else session_id
                        if ordered and conversation_id:
                            self.session_turn_snapshots[conversation_id] = ordered
                            self.debug_streamer.info(f"Turn snapshot stored for session {conversation_id} with {len(ordered)} messages")
                        yield working_history, ""
                        
                    elif event_type == "tool_start":
                        # Tool is starting - immediately add to working history
                        tool_name = metadata.get("tool_name", "unknown") if metadata else "unknown"
                        tool_title = metadata.get("title", format_translation("tool_called", self.language, tool_name=tool_name)) if metadata else format_translation("tool_called", self.language, tool_name="unknown")
                        
                        # Create tool message and immediately add to working history
                        tool_message = {
                            "role": "assistant", 
                            "content": content,
                            "metadata": {"title": tool_title}
                        }
                        working_history.append(tool_message)
                        yield working_history, ""
                        
                    elif event_type == "tool_end":
                        # Tool completed - immediately add to working history
                        tool_name = metadata.get("tool_name", "unknown") if metadata else "unknown"
                        tool_title = metadata.get("title", format_translation("tool_called", self.language, tool_name=tool_name)) if metadata else format_translation("tool_called", self.language, tool_name="unknown")
                        
                        # Create tool message and immediately add to working history
                        tool_message = {
                            "role": "assistant", 
                            "content": content,
                            "metadata": {"title": tool_title}
                        }
                        working_history.append(tool_message)
                        yield working_history, ""
                        
                    elif event_type == "content":
                        # Stream content from response - ensure content is not None
                        content_to_add = safe_string(content)
                        # print(f"🔍 DEBUG: Content event - content: '{content_to_add}', length: {len(content_to_add)}")
                        
                        # Only add line break when LLM starts answering after tool messages
                        if assistant_message_index < 0 and response_content == "":
                            # This is the very first content, check if there are recent tool messages
                            has_recent_tool_messages = False
                            for i in range(max(0, len(working_history) - 3), len(working_history)):
                                if i >= 0 and working_history[i] and working_history[i].get("metadata", {}).get("title"):
                                    has_recent_tool_messages = True
                                    break
                            
                            if has_recent_tool_messages:
                                # Add lean line break only when LLM starts answering after tools
                                content_to_add = "\n" + content_to_add
                        
                        response_content += content_to_add
                        
                        # Update or create assistant message
                        if assistant_message_index >= 0 and assistant_message_index < len(working_history):
                            # Update existing assistant message
                            working_history[assistant_message_index] = {"role": "assistant", "content": response_content}
                            # print(f"🔍 DEBUG: Updated existing assistant message at index {assistant_message_index}")
                        else:
                            # Create new assistant message
                            working_history.append({"role": "assistant", "content": response_content})
                            assistant_message_index = len(working_history) - 1
                            # print(f"🔍 DEBUG: Created new assistant message at index {assistant_message_index}")
                        
                        yield working_history, ""
                        
                    elif event_type == "error":
                        # Add error message to response content so users can see what went wrong
                        response_content += content + "\n\n"
                        
                        # Update assistant message with error
                        if assistant_message_index >= 0 and assistant_message_index < len(working_history):
                            working_history[assistant_message_index] = {"role": "assistant", "content": response_content}
                        else:
                            working_history.append({"role": "assistant", "content": response_content})
                            assistant_message_index = len(working_history) - 1
                        
                        streaming_error_handled = True
                        yield working_history, ""
            
            except Exception as e:
                import traceback
                _logger.error("Error streaming message: %s", e, exc_info=True)
                streaming_error_handled = True
                # print(f"🔍 DEBUG: Set streaming_error_handled to True in streaming loop")
                # Continue with the rest of the processing even if streaming fails
                pass
            
            # Add API token count to final response
            # Add token counts below assistant response
            token_displays = []
            
            # Add prompt tokens if available
            if prompt_tokens:
                token_displays.append(format_translation("prompt_tokens", self.language, tokens=prompt_tokens.formatted))
            
            # Add API tokens if available from session-specific agent
            if user_agent:
                try:
                    # print(f"🔍 DEBUG: Getting last API tokens from session agent")
                    last_api_tokens = user_agent.get_last_api_tokens()
                    # print(f"🔍 DEBUG: Last API tokens: {last_api_tokens}")
                    if last_api_tokens:
                        token_displays.append(format_translation("api_tokens", self.language, tokens=last_api_tokens.formatted))
                        # print(f"🔍 DEBUG: Added API token display")
                    else:
                        # print("🔍 DEBUG: No API tokens available")
                        pass
                except Exception as e:
                    # print(f"🔍 DEBUG: API token error: {e}")
                    self.debug_streamer.warning(f"Failed to get API token count: {e}")
            
            # Add provider/model information if available - use session-specific agent
            if user_agent and hasattr(user_agent, 'get_llm_info'):
                try:
                    llm_info = user_agent.get_llm_info()
                    if llm_info and 'provider' in llm_info and 'model_name' in llm_info:
                        provider = llm_info.get('provider', 'Unknown')
                        model = llm_info.get('model_name', 'Unknown')
                        token_displays.append(format_translation("provider_model", self.language, 
                                                               provider=provider, model=model))
                        # print(f"🔍 DEBUG: Added provider/model display: {provider} / {model}")
                except Exception as e:
                    # print(f"🔍 DEBUG: Provider/model display error: {e}")
                    self.debug_streamer.warning(f"Failed to get provider/model info: {e}")
            
            # Calculate execution time for the entire response
            execution_time = time.time() - start_time
            
            # Add deduplication stats if available from session-specific agent
            if user_agent and hasattr(user_agent, '_deduplication_stats'):
                dedup_stats = user_agent._deduplication_stats.get(session_id, {})
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
                        # print(f"🔍 DEBUG: Added deduplication stats: {total_duplicates} duplicates")
                    
                    # Add total tool calls count
                    if total_tool_calls > 0:
                        token_displays.append(format_translation("total_tool_calls", self.language, calls=total_tool_calls))
                        # print(f"🔍 DEBUG: Added total tool calls: {total_tool_calls}")
            
            # Add token statistics as a separate metadata block
            if token_displays:
                # Add execution time to the token display
                token_displays.append(format_translation("execution_time", self.language, time=execution_time))
                token_display = "\n".join(token_displays)
                # Create a separate metadata block for token statistics
                token_metadata_message = {
                    "role": "assistant", 
                    "content": token_display,
                    "metadata": {"title": format_translation("token_statistics_title", self.language)}
                }
                working_history.append(token_metadata_message)
                # print(f"🔍 DEBUG: Added token metadata block: {token_display}")
            
            # Tool messages are now added immediately during streaming, no need to add them here
            # Ensure tool messages are preserved and not overwritten
            # print(f"🔍 DEBUG: Final working history length: {len(working_history)}")
            for i, msg in enumerate(working_history):
                # Safety check for None first - before any method calls
                if msg is None:
                    # print(f"🔍 DEBUG: Message {i} is None, skipping...")
                    continue
                if not isinstance(msg, dict):
                    # print(f"🔍 DEBUG: Message {i} is not a dict (type: {type(msg)}), skipping...")
                    continue
                # Additional safety for metadata access
                metadata = msg.get("metadata") if msg else None
                if metadata and metadata.get("title"):
                    # print(f"🔍 DEBUG: Tool message {i}: {metadata.get('title', 'No title')}")
                    pass
                elif msg and msg.get("role") == "assistant":
                    # print(f"🔍 DEBUG: Assistant message {i}: {len(msg.get('content', ''))} chars")
                    pass
            
            # Stop processing state
            self.stop_processing()
            
            # Final yield with updated stats
            # print(f"🔍 DEBUG: Final yield - working_history length: {len(working_history)}")
            # print(f"🔍 DEBUG: Final yield - response_content length: {len(response_content)}")
            yield working_history, ""
            
        except Exception as e:
            # Log error to terminal but don't add to chat response
            try:
                self.debug_streamer.error(f"Error in stream chat: {e}")
            except:
                # If debug streamer fails, just continue
                pass
            _logger.error("Streaming error (terminal log only): %s", e, exc_info=True)
            # Stop processing state on error
            # self.stop_processing()
            # Continue gracefully - don't stop processing for streaming errors
            # The response might still be valid even with streaming issues
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
    
    def _stream_message_wrapper(self, message: str, history: List[Dict[str, str]], request: gr.Request = None):
        """Stream a message to the agent (synchronous wrapper)"""
        if not message.strip():
            yield history, ""
            return
        
        # Start processing state for icon rotation
        self.start_processing()
        
        # Always use the persistent background loop to avoid closing gRPC/Gemini resources
        self._ensure_stream_loop()
        import queue
        out_queue = queue.Queue()
        future = self._submit_stream_task(message, history, request, out_queue)
        
        try:
            while True:
                item = out_queue.get()
                if item is StopIteration:
                    break
                yield item
        finally:
            # Best-effort cancel if still running
            try:
                if not future.done():
                    future.cancel()
            except Exception:
                pass
        
        # Refresh UI after streaming completes (EVENT-DRIVEN)
        self._refresh_ui_after_message()
    
    def _update_status(self, request: gr.Request = None) -> str:
        """Update status display - always session-aware"""
        # Use stats tab for proper formatting (now always session-aware)
        stats_tab = self.tab_instances.get('stats')
        if stats_tab and hasattr(stats_tab, 'format_stats_display'):
            return stats_tab.format_stats_display(request)
        
        # Final fallback
        if self.is_ready():
            return get_translation_key("agent_ready", self.language)
        else:
            return get_translation_key("agent_initializing", self.language)
    
    def _update_token_budget(self, request: gr.Request = None) -> str:
        """Update token budget display - delegates to chat tab with session awareness"""
        chat_tab = self.tab_instances.get('chat')
        if chat_tab and hasattr(chat_tab, 'format_token_budget_display'):
            return chat_tab.format_token_budget_display(request)
        
        # Fallback token budget
        return get_translation_key("token_budget_initializing", self.language)
    
    
    def _refresh_logs(self, request: gr.Request = None) -> str:
        """Refresh logs display - delegates to logs tab with session awareness"""
        logs_tab = self.tab_instances.get('logs')
        if logs_tab and hasattr(logs_tab, 'get_initialization_logs'):
            return logs_tab.get_initialization_logs(request)
        
        # Fallback logs
        return "\n".join(self.initialization_logs) if self.initialization_logs else "No logs available"
    
    def _refresh_stats(self, request: gr.Request = None) -> str:
        """Refresh stats display - delegates to stats tab with session awareness"""
        stats_tab = self.tab_instances.get('stats')
        if stats_tab and hasattr(stats_tab, 'format_stats_display'):
            return stats_tab.format_stats_display(request)
        
        # Fallback stats
        if self.is_ready():
            return get_translation_key("agent_ready", self.language)
        else:
            return get_translation_key("agent_initializing", self.language)
    
    def _trigger_ui_update(self):
        """Trigger UI update after agent initialization or message processing"""
        try:
            # print("🔍 DEBUG: Triggering UI update...")
            # Store the update trigger - the UI will check this
            self._ui_update_needed = True
        except Exception as e:
            # print(f"🔍 DEBUG: Error triggering UI update: {e}")
            pass
    
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
            # print("🔍 DEBUG: Triggering UI update...")
            # This will be called when the agent is ready or after messages
            # The actual UI update will happen through Gradio's event system
            self._ui_update_needed = True
        except Exception as e:
            # print(f"🔍 DEBUG: Error triggering UI update: {e}")
            pass
    
    def _refresh_ui_after_message(self):
        """Refresh all UI components after a message is processed (EVENT-DRIVEN)"""
        try:
            # Trigger UI update after message processing
            self.trigger_ui_update()
            
            # print("🔍 DEBUG: UI refreshed after message completion (EVENT-DRIVEN)")
            
        except Exception as e:
            # print(f"🔍 DEBUG: Error refreshing UI after message: {e}")
            pass
    
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
                _logger.warning("ChatTab not available")
                
            if LogsTab:
                logs_tab = LogsTab(event_handlers, language=self.language, i18n_instance=self.i18n)
                logs_tab.set_main_app(self)  # Pass main app reference
                tab_modules.append(logs_tab)
                self.tab_instances['logs'] = logs_tab
            else:
                _logger.warning("LogsTab not available")
                
            if StatsTab:
                stats_tab = StatsTab(event_handlers, language=self.language, i18n_instance=self.i18n)
                stats_tab.set_main_app(self)  # Set reference to main app for session management
                tab_modules.append(stats_tab)
                self.tab_instances['stats'] = stats_tab
            else:
                _logger.warning("StatsTab not available")
        except Exception as e:
            _logger.exception("Error creating tab modules: %s", e)
            raise
        
        # Use UI Manager to create interface
        try:
            demo = self.ui_manager.create_interface(tab_modules, event_handlers)
        except Exception as e:
            _logger.exception("Error creating interface: %s", e)
            raise
        
        # Configure concurrency and queuing
        self.queue_manager.configure_queue(demo)
        
        # Consolidate all components from UI Manager (single source of truth)
        self.components = self.ui_manager.get_components()
        
        # No global agent - tabs will get session-specific agents as needed
        
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
            _logger.exception("Error creating demo: %s", e)
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
        _logger.exception("Error creating demo for language %s: %s", language, e)
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
        _logger.exception("Error creating safe demo: %s", e)
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
        _logger.info("Reloading demo...")
        demo = get_demo_with_language_detection()
        return demo
    except Exception as e:
        _logger.exception("Error reloading demo: %s", e)
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
    
    # Setup LangSmith environment first
    from agent_ng.langsmith_config import setup_langsmith_environment
    setup_langsmith_environment()
    
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
            _logger.error("Could not find an available port starting from %s", default_port)
            sys.exit(1)
    else:
        port = default_port
    
    _logger.info("Starting LangChain-Native LLM Agent App with language detection...")
    _logger.info("Language: %s", language.upper())
    _logger.info("Port: %s", port)
    
    # Create app with specified language
    app = NextGenAppWithLanguageDetection(language=language)
    demo = app.create_interface()
    
    _logger.info("Launching Gradio interface on port %s with language switching...", port)
    demo.launch(
        debug=True,
        share=False,
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )


if __name__ == "__main__":
    main()
