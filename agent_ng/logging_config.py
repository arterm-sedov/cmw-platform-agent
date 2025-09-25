"""
Lean, idempotent logging configuration for the project.

Reads environment via dotenv and configures the root logger with a console
handler and optional rotating file handler. Supports simple/verbose or JSON
formatting. Designed to be called early in application startup and safe to
call multiple times.

Environment variables (all optional):

- LOG_LEVEL: Global minimum level for logs.
  - Values: DEBUG | INFO | WARNING | ERROR | CRITICAL
  - Default: INFO
  - Example: LOG_LEVEL=DEBUG

- LOG_FORMAT: Human-readable text format preset.
  - Values: simple | verbose
  - Default: simple
  - simple: "%(asctime)s %(levelname)s %(name)s: %(message)s"
  - verbose: adds process/thread, filename and line number (see below toggles)

- LOG_JSON: Emit JSON lines instead of text format.
  - Values: true | false
  - Default: false
  - Example: LOG_JSON=true

- LOG_FILE: Path to a file to also write logs.
  - Empty or unset: console only
  - Example: LOG_FILE=logs/app.log

- LOG_ROTATE: Use rotating file handler when LOG_FILE is set.
  - Values: true | false
  - Default: true (when LOG_FILE provided)
  - Example: LOG_ROTATE=false

- LOG_MAX_BYTES: Max size per log file before rotation (bytes).
  - Default: 1048576 (1MB)
  - Example: LOG_MAX_BYTES=5242880  # 5MB

- LOG_BACKUP_COUNT: Number of rotated files to keep.
  - Default: 5
  - Example: LOG_BACKUP_COUNT=10

- LOG_PROPAGATE: Whether root messages propagate to ancestor loggers.
  - Values: true | false
  - Default: true

- LOG_INCLUDE_PROCESS: Include process id in verbose format.
  - Values: true | false
  - Default: false

- LOG_INCLUDE_THREAD: Include thread name in verbose format.
  - Values: true | false
  - Default: false

- LOG_FORCE: Force reconfiguration (removes existing handlers and re-applies).
  - Values: true | false
  - Default: false
  - Use when reloading or changing logging at runtime

- LOG_CONSOLE_MAX_LENGTH: Maximum length for console log entries before truncation.
  - Values: integer
  - Default: 400
  - Example: LOG_CONSOLE_MAX_LENGTH=500
"""

from __future__ import annotations

import json
import logging
from logging import Handler, Logger
from logging.handlers import RotatingFileHandler
import os
from typing import Optional

from dotenv import load_dotenv

_INITIALIZED = False


class _JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for structured logs.

    Emits one JSON object per line with core fields and exception info when present.
    """

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "time": self.formatTime(record, datefmt="%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info:
            exc_type = record.exc_info[0].__name__ if record.exc_info[0] else None
            base.update(
                {
                    "exc_type": exc_type,
                    "exc_message": str(record.exc_info[1]) if record.exc_info[1] else None,
                    "traceback": self.formatException(record.exc_info),
                }
            )
        return json.dumps(base, ensure_ascii=False)


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def setup_logging(force: bool | None = None) -> Logger:
    """Configure root logging using environment variables.

    Environment variables:
    - LOG_LEVEL (default: INFO)
    - LOG_FORMAT: simple | verbose (default: simple)
    - LOG_JSON: true|false (default: false)
    - LOG_FILE: path to log file (optional)
    - LOG_ROTATE: true|false (default: true if LOG_FILE set)
    - LOG_MAX_BYTES: int (default: 1048576)
    - LOG_BACKUP_COUNT: int (default: 5)
    - LOG_PROPAGATE: true|false (default: true)
    - LOG_INCLUDE_PROCESS: true|false (default: false)
    - LOG_INCLUDE_THREAD: true|false (default: false)
    - LOG_FORCE: true|false (default: false)
    """

    global _INITIALIZED

    # Load env only once; safe if called many times
    load_dotenv()

    if force is None:
        force = _parse_bool(os.getenv("LOG_FORCE"), False)

    if _INITIALIZED and not force:
        return logging.getLogger()

    root = logging.getLogger()

    # Reset handlers if forcing or first-time init
    if force or not _INITIALIZED:
        for h in list(root.handlers):
            try:
                root.removeHandler(h)
                h.close()
            except Exception:
                # Best-effort cleanup; never raise from logging setup
                pass

    # Level (global minimum)
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    try:
        level = getattr(logging, level_name, logging.INFO)
    except Exception:
        level = logging.INFO
    root.setLevel(level)

    # Formatter selection: text (simple/verbose) or JSON
    use_json = _parse_bool(os.getenv("LOG_JSON"), False)
    fmt_style = os.getenv("LOG_FORMAT", "simple").lower()
    include_process = _parse_bool(os.getenv("LOG_INCLUDE_PROCESS"), False)
    include_thread = _parse_bool(os.getenv("LOG_INCLUDE_THREAD"), False)

    if use_json:
        formatter: logging.Formatter = _JsonFormatter()
    else:
        base = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        if fmt_style == "verbose":
            parts = ["%(asctime)s", "%(levelname)s"]
            if include_process:
                parts.append("%(process)d")
            if include_thread:
                parts.append("%(threadName)s")
            parts.extend(["%(name)s", "%(filename)s:%(lineno)d", ":", "%(message)s"])
            base = " ".join(parts)
        formatter = logging.Formatter(fmt=base, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler with truncation
    console = logging.StreamHandler()
    console.setLevel(level)
    
    # Create a custom formatter that truncates long messages
    class TruncatingFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None, max_length=400):
            super().__init__(fmt, datefmt)
            self.max_length = max_length
        
        def format(self, record):
            # Get the original formatted message
            formatted = super().format(record)
            
            # Truncate if too long
            if len(formatted) > self.max_length:
                truncated = formatted[:self.max_length-3] + "..."
                return truncated
            
            return formatted
    
    # Use the truncating formatter for console
    console_max_length = int(os.getenv("LOG_CONSOLE_MAX_LENGTH", "400"))
    console_formatter = TruncatingFormatter(
        fmt=formatter._fmt if hasattr(formatter, '_fmt') else formatter.format,
        datefmt=formatter.datefmt if hasattr(formatter, 'datefmt') else None,
        max_length=console_max_length
    )
    console.setFormatter(console_formatter)
    root.addHandler(console)

    # Optional file handler (with rotation)
    log_file = os.getenv("LOG_FILE", "").strip()
    if log_file:
        rotate_default = True
        rotate = _parse_bool(os.getenv("LOG_ROTATE"), rotate_default)
        max_bytes = int(os.getenv("LOG_MAX_BYTES", "1048576") or 1048576)
        backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5") or 5)
        try:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        except Exception:
            pass
        file_handler: Handler
        if rotate:
            file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        else:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Propagation
    root.propagate = _parse_bool(os.getenv("LOG_PROPAGATE"), True)

    # Attach internal debug handler if available (keeps Logs tab working)
    try:
        # Try absolute import first
        try:
            from agent_ng.debug_streamer import get_log_handler
        except ImportError:
            # Fallback to relative import
            from .debug_streamer import get_log_handler

        # Create handlers for common session IDs that might be used
        session_ids = ["default", "gradio_default", "app_ng"]
        for session_id in session_ids:
            try:
                dbg = get_log_handler(session_id)
                if dbg and isinstance(dbg, Handler) and dbg not in root.handlers:
                    # Don't set formatter - let debug_streamer handle its own formatting
                    dbg.setLevel(level)
                    root.addHandler(dbg)
            except Exception:
                # Continue with other session IDs
                continue
    except Exception as e:
        # Optional integration; ignore on failure
        print(f"Warning: Could not attach debug_streamer handler: {e}")
        pass

    _INITIALIZED = True
    return root


