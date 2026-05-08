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
  - Logs get YYYYMMDD date suffix automatically
  - Example: LOG_FILE=logs/app.log

- LOG_MAX_BYTES: Max size per log file in bytes before rotation to .1, .2, etc.
  - Default: 1048576 (1MB)
  - Example: LOG_MAX_BYTES=5242880

- LOG_MAX_SIZE_MB: Max total size in MB. Oldest files deleted when exceeded.
  - Default: 100 (MB)
  - Example: LOG_MAX_SIZE_MB=50

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

from contextlib import suppress
import json
import logging
from logging import Logger
from logging.handlers import RotatingFileHandler
import os
import time

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
                    "exc_message": str(record.exc_info[1])
                    if record.exc_info[1]
                    else None,
                    "traceback": self.formatException(record.exc_info),
                }
            )
        return json.dumps(base, ensure_ascii=False)


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _enforce_size_limit(log_dir: str, base_name: str, max_total_bytes: int) -> None:
    if max_total_bytes <= 0 or not os.path.isdir(log_dir):
        return
    with suppress(OSError):
        files = []
        for f in os.listdir(log_dir):
            if f.startswith(base_name):
                fpath = os.path.join(log_dir, f)
                if os.path.isfile(fpath):
                    files.append(
                        (fpath, os.path.getmtime(fpath), os.path.getsize(fpath))
                    )
        if not files:
            return
        files.sort(key=lambda x: x[1])
        total = sum(f[2] for f in files)
        while total > max_total_bytes and files:
            oldest = files.pop(0)
            total -= oldest[2]
            with suppress(OSError):
                os.remove(oldest[0])


def _date_suffix(log_file: str) -> str:
    base, ext = os.path.splitext(log_file)
    return f"{base}-{time.strftime('%Y%m%d')}{ext}"


def _make_file_handler(log_file: str) -> RotatingFileHandler:
    dated = _date_suffix(log_file)
    log_dir = os.path.dirname(log_file) or "."
    base_name = os.path.basename(os.path.splitext(log_file)[0])
    os.makedirs(log_dir, exist_ok=True)
    max_bytes = int(os.getenv("LOG_MAX_BYTES", "1048576") or 1048576)
    max_mb = int(os.getenv("LOG_MAX_SIZE_MB", "100") or 100)
    _enforce_size_limit(log_dir, base_name, max_mb * 1024 * 1024)
    backup_count = (max_mb * 1024 * 1024) // max_bytes
    return RotatingFileHandler(
        dated, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )


def setup_openapi_debug_log() -> RotatingFileHandler:
    load_dotenv()
    log_dir = os.path.dirname(os.getenv("LOG_FILE", ".")) or "."
    path = os.path.join(log_dir, "openai_compat_io_debug.jsonl")
    handler = _make_file_handler(path)
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


def setup_logging(force: bool | None = None) -> Logger:
    """Configure root logging using environment variables.

    Environment variables:
    - LOG_LEVEL (default: INFO)
    - LOG_FORMAT: simple | verbose (default: simple)
    - LOG_JSON: true|false (default: false)
    - LOG_FILE: path to log file (optional)
    - LOG_MAX_BYTES: int (default: 1048576)
    - LOG_MAX_SIZE_MB: int (default: 100)
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
            root.removeHandler(h)
            h.close()

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
                truncated = formatted[: self.max_length - 3] + "..."
                return truncated

            return formatted

    # Use the truncating formatter for console
    console_max_length = int(os.getenv("LOG_CONSOLE_MAX_LENGTH", "400"))
    console_formatter = TruncatingFormatter(
        fmt=formatter._fmt if hasattr(formatter, "_fmt") else formatter.format,
        datefmt=formatter.datefmt if hasattr(formatter, "datefmt") else None,
        max_length=console_max_length,
    )
    console.setFormatter(console_formatter)
    root.addHandler(console)

    # Optional file handler (rotation, date suffix, size cap)
    log_file = os.getenv("LOG_FILE", "").strip()
    if log_file:
        file_handler = _make_file_handler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Propagation
    root.propagate = _parse_bool(os.getenv("LOG_PROPAGATE"), True)

    # Attach session-aware debug handlers (keeps Logs tab working)
    try:
        # Try absolute import first
        try:
            from agent_ng.debug_streamer import SessionAwareLogHandler, get_log_handler
        except ImportError:
            # Fallback to relative import
            from .debug_streamer import SessionAwareLogHandler, get_log_handler

        # Create a session-aware handler that routes logs to appropriate session handlers
        session_aware_handler = SessionAwareLogHandler()
        session_aware_handler.setLevel(level)
        root.addHandler(session_aware_handler)

        # Session-aware handler will create session handlers on-demand
        # No need to pre-create handlers for non-existent sessions
    except Exception as e:
        # Optional integration; ignore on failure
        print(f"Warning: Could not attach debug_streamer handler: {e}")

    _INITIALIZED = True
    return root
