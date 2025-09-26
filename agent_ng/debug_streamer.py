"""
Lean Debug Streamer
==================

A highly efficient, modular debug streaming system for real-time logging
and thinking transparency in LLM agents.

Key Features:
- Real-time log streaming to Gradio interface
- Thinking transparency with collapsible sections
- Minimal overhead and clean separation of concerns
- Thread-safe logging with queue-based streaming
- Support for different log levels and categories
- Integration with Gradio ChatMessage metadata

Inspired by the sophisticated logging in agent.py but designed to be
lean, efficient, and modular.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
from queue import Empty, Queue
import threading
import time
from typing import Any, Dict, List, Optional, Union


class LogLevel(Enum):
    """Log levels for different types of messages"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    LLM_STREAM = "llm_stream"
    SUCCESS = "success"


class LogCategory(Enum):
    """Categories for organizing logs"""
    INIT = "initialization"
    LLM = "llm_call"
    TOOL = "tool_execution"
    STREAM = "streaming"
    ERROR = "error_handling"
    THINKING = "thinking_process"
    SYSTEM = "system"


@dataclass
class LogEntry:
    """A single log entry with metadata"""
    timestamp: float
    level: LogLevel
    category: LogCategory
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    thread_id: str = ""
    session_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "category": self.category.value,
            "message": self.message,
            "metadata": self.metadata,
            "thread_id": self.thread_id,
            "session_id": self.session_id,
            "formatted_time": datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S.%f")[:-3]
        }


class DebugStreamer:
    """
    Lean, efficient debug streamer for real-time logging.
    
    Features:
    - Thread-safe queue-based logging
    - Real-time streaming to Gradio interface
    - Minimal overhead with clean separation
    - Support for different log levels and categories
    - Integration with Gradio ChatMessage metadata
    """

    def __init__(self, session_id: str = "default", max_queue_size: int = 1000):
        self.session_id = session_id
        self.max_queue_size = max_queue_size
        self.log_queue = Queue(maxsize=max_queue_size)
        self.subscribers: list[Callable[[LogEntry], None]] = []
        self.is_running = False
        self.worker_thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Start the worker thread
        self.start()

    def start(self):
        """Start the debug streamer worker thread"""
        if self.is_running:
            return

        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """Stop the debug streamer"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

    def subscribe(self, callback: Callable[[LogEntry], None]):
        """Subscribe to log entries"""
        with self._lock:
            self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[LogEntry], None]):
        """Unsubscribe from log entries"""
        with self._lock:
            if callback in self.subscribers:
                self.subscribers.remove(callback)

    def _worker_loop(self):
        """Worker loop that processes log entries"""
        while self.is_running:
            try:
                # Get log entry with timeout
                entry = self.log_queue.get(timeout=0.1)

                # Notify all subscribers
                with self._lock:
                    for callback in self.subscribers:
                        try:
                            callback(entry)
                        except Exception as e:
                            print(f"Error in log subscriber: {e}")

                self.log_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                print(f"Error in debug streamer worker: {e}")

    def log(self, level: LogLevel, category: LogCategory, message: str,
            metadata: dict[str, Any] | None = None, session_id: str | None = None):
        """Log a message with the specified level and category"""
        if metadata is None:
            metadata = {}

        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            category=category,
            message=message,
            metadata=metadata,
            thread_id=threading.get_ident(),
            session_id=session_id or self.session_id
        )

        try:
            self.log_queue.put_nowait(entry)
        except:
            # Queue is full, drop the oldest entry
            try:
                self.log_queue.get_nowait()
                self.log_queue.put_nowait(entry)
            except Empty:
                pass

    # Convenience methods for different log levels
    def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM,
              metadata: dict[str, Any] | None = None):
        """Log a debug message"""
        self.log(LogLevel.DEBUG, category, message, metadata)

    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM,
             metadata: dict[str, Any] | None = None):
        """Log an info message"""
        self.log(LogLevel.INFO, category, message, metadata)

    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM,
                metadata: dict[str, Any] | None = None):
        """Log a warning message"""
        self.log(LogLevel.WARNING, category, message, metadata)

    def error(self, message: str, category: LogCategory = LogCategory.ERROR,
              metadata: dict[str, Any] | None = None):
        """Log an error message"""
        self.log(LogLevel.ERROR, category, message, metadata)

    def thinking(self, message: str, metadata: dict[str, Any] | None = None):
        """Log a thinking process message"""
        self.log(LogLevel.THINKING, LogCategory.THINKING, message, metadata)

    def tool_use(self, tool_name: str, tool_args: dict[str, Any],
                 result: str | None = None, metadata: dict[str, Any] | None = None):
        """Log a tool usage"""
        tool_metadata = {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "result": result,
            **(metadata or {})
        }
        self.log(LogLevel.TOOL_USE, LogCategory.TOOL, f"Using tool: {tool_name}", tool_metadata)

    def llm_stream(self, content: str, metadata: dict[str, Any] | None = None):
        """Log LLM streaming content"""
        self.log(LogLevel.LLM_STREAM, LogCategory.LLM, content, metadata)

    def success(self, message: str, category: LogCategory = LogCategory.SYSTEM,
                metadata: dict[str, Any] | None = None):
        """Log a success message"""
        self.log(LogLevel.SUCCESS, category, message, metadata)

    def get_recent_logs(self, count: int = 50) -> list[LogEntry]:
        """Get recent log entries (for debugging)"""
        # This is a simple implementation - in production you might want to use a proper log store
        return []


class SessionAwareLogHandler(logging.Handler):
    """
    Session-aware log handler that routes logs to appropriate session handlers.
    
    This handler determines the current session context and routes log records
    to the appropriate session-specific log handler.
    """

    def __init__(self):
        super().__init__()
        self._session_context = threading.local()
        self._default_session_id = "default"
        self._global_session_context = {}  # Global session context storage
        self._context_lock = threading.Lock()

    def set_session_context(self, session_id: str):
        """Set the current session context for this thread"""
        self._session_context.session_id = session_id
        # Also store in global context for cross-thread access
        with self._context_lock:
            self._global_session_context[threading.get_ident()] = session_id

    def get_current_session_id(self) -> str:
        """Get the current session ID for this thread"""
        # First try thread-local storage
        thread_local_id = getattr(self._session_context, 'session_id', None)
        if thread_local_id:
            return thread_local_id
        
        # Fallback to global context
        with self._context_lock:
            return self._global_session_context.get(threading.get_ident(), self._default_session_id)

    def emit(self, record: logging.LogRecord) -> None:
        """Route log records to the appropriate session handler - no fallback to default"""
        try:
            # Get current session ID
            session_id = self.get_current_session_id()
            
            # Try to extract session ID from the log record if available
            if hasattr(record, 'session_id') and record.session_id:
                session_id = record.session_id
            elif hasattr(record, 'args') and record.args:
                # Look for session ID in the log message args
                for arg in record.args:
                    if isinstance(arg, str) and 'session' in arg.lower():
                        # Try to extract session ID from the message
                        import re
                        session_match = re.search(r'session[_\s]*([a-zA-Z0-9_]+)', arg, re.IGNORECASE)
                        if session_match:
                            session_id = session_match.group(1)
                            break
            
            # Try to extract session ID from the log message itself - improved pattern matching
            if session_id == self._default_session_id:
                import re
                message = record.getMessage()
                
                # Look for patterns like "gradio_abc123" or "session_123" in the message
                # This is the most common pattern in our logs
                session_match = re.search(r'(?:gradio[_\s]*|session[_\s]*)([a-zA-Z0-9_]+)', message, re.IGNORECASE)
                if session_match:
                    session_id = session_match.group(1)
                else:
                    # Also try to find any session-like pattern that looks like a Gradio session hash
                    # Gradio session hashes are typically 10+ characters long
                    session_match = re.search(r'([a-zA-Z0-9_]{10,})', message)
                    if session_match and ('gradio' in message.lower() or 'session' in message.lower()):
                        session_id = session_match.group(1)
                    else:
                        # Try to find any pattern that looks like a session ID
                        # Look for patterns like "gradio_abc123" or "session_123" anywhere in the message
                        session_match = re.search(r'(gradio[_\s]*[a-zA-Z0-9_]+|session[_\s]*[a-zA-Z0-9_]+)', message, re.IGNORECASE)
                        if session_match:
                            # Extract just the ID part
                            id_match = re.search(r'([a-zA-Z0-9_]+)$', session_match.group(1))
                            if id_match:
                                session_id = id_match.group(1)
            
            # Only route to session-specific handler if we found a valid session ID
            if session_id and session_id != self._default_session_id:
                session_handler = get_log_handler(session_id)
                if session_handler:
                    # Route to session-specific handler
                    session_handler.emit(record)
            # No fallback to default session - logs without session ID are ignored
        except Exception:
            # No fallback - just ignore logs that can't be routed
            pass


class GradioLogHandler(logging.Handler):
    """
    Handler for streaming logs to Gradio interface.
    
    This class handles the conversion of log entries to Gradio-compatible
    formats and manages the streaming to the Logs tab.
    """

    def __init__(self, debug_streamer: DebugStreamer):
        super().__init__()
        self.debug_streamer = debug_streamer
        self.log_buffer: list[str] = []
        self.max_buffer_size = 1000
        self.current_logs_display = ""

        # Subscribe to log entries
        self.debug_streamer.subscribe(self._handle_log_entry)

    def emit(self, record: logging.LogRecord) -> None:
        """Bridge Python logging records into the DebugStreamer."""
        try:
            # Map logging level to LogLevel
            if record.levelno >= logging.ERROR:
                level = LogLevel.ERROR
            elif record.levelno >= logging.WARNING:
                level = LogLevel.WARNING
            elif record.levelno >= logging.INFO:
                level = LogLevel.INFO
            else:
                level = LogLevel.DEBUG

            message = record.getMessage()
            metadata = {
                "logger": record.name,
                "module": record.module,
                "file": record.filename,
                "line": record.lineno,
            }
            self.debug_streamer.log(level, LogCategory.SYSTEM, message, metadata)
        except Exception:
            # Never raise from logging handler
            pass

    def _handle_log_entry(self, entry: LogEntry):
        """Handle a new log entry"""
        # Format the log entry for display
        formatted_log = self._format_log_entry(entry)

        # Add to buffer
        self.log_buffer.append(formatted_log)

        # Trim buffer if too large
        if len(self.log_buffer) > self.max_buffer_size:
            self.log_buffer = self.log_buffer[-self.max_buffer_size:]

        # Update current display
        self.current_logs_display = "\n".join(self.log_buffer[-50:])  # Show last 50 entries

    def _format_log_entry(self, entry: LogEntry) -> str:
        """Format a log entry for display"""
        timestamp = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S.%f")[:-3]

        # Choose emoji based on level
        emoji_map = {
            LogLevel.DEBUG: "🔍",
            LogLevel.INFO: "ℹ️",
            LogLevel.WARNING: "⚠️",
            LogLevel.ERROR: "❌",
            LogLevel.THINKING: "💭",
            LogLevel.TOOL_USE: "🔧",
            LogLevel.LLM_STREAM: "📡",
            LogLevel.SUCCESS: "✅"
        }

        emoji = emoji_map.get(entry.level, "📝")

        # Format the message with better spacing
        formatted = f"{emoji} [{timestamp}] {entry.message}"

        # Add metadata if present
        if entry.metadata:
            metadata_str = json.dumps(entry.metadata, indent=2)
            formatted += f"\n   📋 {metadata_str}"

        # Add new line for better readability
        formatted += "\n"

        return formatted

    def get_current_logs(self) -> str:
        """Get the current logs display"""
        return self.current_logs_display or "No logs available yet."

    def clear_logs(self):
        """Clear the log buffer"""
        self.log_buffer.clear()
        self.current_logs_display = ""


class ThinkingTransparency:
    """
    Handler for thinking transparency in Gradio ChatMessage.
    
    This class manages the creation of thinking sections that can be
    displayed in collapsible accordions in the Gradio chat interface.
    """

    def __init__(self, debug_streamer: DebugStreamer):
        self.debug_streamer = debug_streamer
        self.current_thinking = ""
        self.thinking_metadata = {}

    def start_thinking(self, title: str = "🧠 Thinking", metadata: dict[str, Any] | None = None):
        """Start a thinking process"""
        self.current_thinking = ""
        self.thinking_metadata = {
            "title": title,
            "status": "pending",
            **(metadata or {})
        }
        self.debug_streamer.thinking(f"Starting thinking process: {title}")

    def add_thinking(self, content: str):
        """Add content to the current thinking process"""
        self.current_thinking += content
        self.debug_streamer.thinking(content)

    def complete_thinking(self, final_content: str | None = None):
        """Complete the thinking process"""
        if final_content:
            self.current_thinking = final_content

        self.thinking_metadata["status"] = "done"
        self.debug_streamer.thinking("Thinking process completed")

        return self._create_thinking_message()

    def _create_thinking_message(self) -> dict[str, Any]:
        """Create a ChatMessage-compatible thinking message"""
        return {
            "role": "assistant",
            "content": self.current_thinking,
            "metadata": self.thinking_metadata
        }

    def create_tool_usage_message(self, tool_name: str, tool_args: dict[str, Any],
                                 result: str) -> dict[str, Any]:
        """Create a tool usage message with metadata"""
        return {
            "role": "assistant",
            "content": f"Used tool: {tool_name}\n\n**Arguments:**\n{json.dumps(tool_args, indent=2)}\n\n**Result:**\n{result}",
            "metadata": {
                "title": f"🔧 {tool_name}",
                "status": "done",
                "tool_name": tool_name,
                "tool_args": tool_args
            }
        }


# Session-specific debug streamer instances
_debug_streamers: dict[str, DebugStreamer] = {}
_log_handlers: dict[str, GradioLogHandler] = {}
_thinking_transparencies: dict[str, ThinkingTransparency] = {}
_session_aware_handler: SessionAwareLogHandler | None = None


def get_debug_streamer(session_id: str = "default") -> DebugStreamer:
    """Get session-specific debug streamer instance"""
    global _debug_streamers
    if session_id not in _debug_streamers:
        _debug_streamers[session_id] = DebugStreamer(session_id)
    return _debug_streamers[session_id]


def get_log_handler(session_id: str = "default") -> GradioLogHandler:
    """Get session-specific log handler instance"""
    global _log_handlers
    if session_id not in _log_handlers:
        debug_streamer = get_debug_streamer(session_id)
        _log_handlers[session_id] = GradioLogHandler(debug_streamer)
    
    # Set session context for this session to ensure proper routing
    try:
        session_aware_handler = get_session_aware_handler()
        session_aware_handler.set_session_context(session_id)
    except Exception:
        # If setting context fails, continue anyway
        pass
    
    return _log_handlers[session_id]


def get_thinking_transparency(session_id: str = "default") -> ThinkingTransparency:
    """Get session-specific thinking transparency instance"""
    global _thinking_transparencies
    if session_id not in _thinking_transparencies:
        debug_streamer = get_debug_streamer(session_id)
        _thinking_transparencies[session_id] = ThinkingTransparency(debug_streamer)
    return _thinking_transparencies[session_id]


def get_session_aware_handler() -> SessionAwareLogHandler:
    """Get the global session-aware log handler"""
    global _session_aware_handler
    if _session_aware_handler is None:
        _session_aware_handler = SessionAwareLogHandler()
    return _session_aware_handler


def set_session_context(session_id: str):
    """Set the current session context for logging"""
    handler = get_session_aware_handler()
    handler.set_session_context(session_id)


