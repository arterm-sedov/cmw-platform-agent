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
import threading
import time
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
import json
from datetime import datetime


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
    metadata: Dict[str, Any] = field(default_factory=dict)
    thread_id: str = ""
    session_id: str = "default"
    
    def to_dict(self) -> Dict[str, Any]:
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
        self.subscribers: List[Callable[[LogEntry], None]] = []
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
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
            metadata: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None):
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
              metadata: Optional[Dict[str, Any]] = None):
        """Log a debug message"""
        self.log(LogLevel.DEBUG, category, message, metadata)
    
    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
             metadata: Optional[Dict[str, Any]] = None):
        """Log an info message"""
        self.log(LogLevel.INFO, category, message, metadata)
    
    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
                metadata: Optional[Dict[str, Any]] = None):
        """Log a warning message"""
        self.log(LogLevel.WARNING, category, message, metadata)
    
    def error(self, message: str, category: LogCategory = LogCategory.ERROR, 
              metadata: Optional[Dict[str, Any]] = None):
        """Log an error message"""
        self.log(LogLevel.ERROR, category, message, metadata)
    
    def thinking(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a thinking process message"""
        self.log(LogLevel.THINKING, LogCategory.THINKING, message, metadata)
    
    def tool_use(self, tool_name: str, tool_args: Dict[str, Any], 
                 result: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Log a tool usage"""
        tool_metadata = {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "result": result,
            **(metadata or {})
        }
        self.log(LogLevel.TOOL_USE, LogCategory.TOOL, f"Using tool: {tool_name}", tool_metadata)
    
    def llm_stream(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Log LLM streaming content"""
        self.log(LogLevel.LLM_STREAM, LogCategory.LLM, content, metadata)
    
    def success(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
                metadata: Optional[Dict[str, Any]] = None):
        """Log a success message"""
        self.log(LogLevel.SUCCESS, category, message, metadata)
    
    def get_recent_logs(self, count: int = 50) -> List[LogEntry]:
        """Get recent log entries (for debugging)"""
        # This is a simple implementation - in production you might want to use a proper log store
        return []


class GradioLogHandler:
    """
    Handler for streaming logs to Gradio interface.
    
    This class handles the conversion of log entries to Gradio-compatible
    formats and manages the streaming to the Logs tab.
    """
    
    def __init__(self, debug_streamer: DebugStreamer):
        self.debug_streamer = debug_streamer
        self.log_buffer: List[str] = []
        self.max_buffer_size = 1000
        self.current_logs_display = ""
        
        # Subscribe to log entries
        self.debug_streamer.subscribe(self._handle_log_entry)
    
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
    
    def start_thinking(self, title: str = "🧠 Thinking", metadata: Optional[Dict[str, Any]] = None):
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
    
    def complete_thinking(self, final_content: Optional[str] = None):
        """Complete the thinking process"""
        if final_content:
            self.current_thinking = final_content
        
        self.thinking_metadata["status"] = "done"
        self.debug_streamer.thinking("Thinking process completed")
        
        return self._create_thinking_message()
    
    def _create_thinking_message(self) -> Dict[str, Any]:
        """Create a ChatMessage-compatible thinking message"""
        return {
            "role": "assistant",
            "content": self.current_thinking,
            "metadata": self.thinking_metadata
        }
    
    def create_tool_usage_message(self, tool_name: str, tool_args: Dict[str, Any], 
                                 result: str) -> Dict[str, Any]:
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
_debug_streamers: Dict[str, DebugStreamer] = {}
_log_handlers: Dict[str, GradioLogHandler] = {}
_thinking_transparencies: Dict[str, ThinkingTransparency] = {}


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
    return _log_handlers[session_id]


def get_thinking_transparency(session_id: str = "default") -> ThinkingTransparency:
    """Get session-specific thinking transparency instance"""
    global _thinking_transparencies
    if session_id not in _thinking_transparencies:
        debug_streamer = get_debug_streamer(session_id)
        _thinking_transparencies[session_id] = ThinkingTransparency(debug_streamer)
    return _thinking_transparencies[session_id]


def cleanup_debug_system():
    """Cleanup the debug system"""
    global _debug_streamers, _log_handlers, _thinking_transparencies
    
    # Stop all debug streamers
    for streamer in _debug_streamers.values():
        streamer.stop()
    
    # Clear all instances
    _debug_streamers.clear()
    _log_handlers.clear()
    _thinking_transparencies.clear()
