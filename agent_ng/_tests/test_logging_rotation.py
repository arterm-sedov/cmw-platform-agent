"""Tests for logging rotation on Windows and POSIX.

Verifies that the logging handler used by setup_logging() can rotate
without PermissionError on Windows (the core bug this fix addresses).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


class TestLoggingRotation:
    """Contract: handler rotates without PermissionError and writes logs."""

    def test_concurrent_rotating_file_handler_importable(self) -> None:
        """ConcurrentRotatingFileHandler is importable (dependency installed)."""
        from concurrent_log_handler import ConcurrentRotatingFileHandler

        assert ConcurrentRotatingFileHandler is not None

    def test_handler_rotates_without_error(self, tmp_path: Path) -> None:
        """Handler rotates when maxBytes exceeded — no PermissionError."""
        from concurrent_log_handler import ConcurrentRotatingFileHandler

        log_file = str(tmp_path / "test.log")
        handler = ConcurrentRotatingFileHandler(
            log_file,
            maxBytes=100,  # very small to force rotation
            backupCount=3,
            encoding="utf-8",
        )
        logger = logging.getLogger("test_rotation")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        # Write enough data to trigger rotation
        for i in range(50):
            logger.info("Message %d: %s", i, "x" * 50)

        handler.close()
        logger.removeHandler(handler)

        # Verify log file exists and has content
        assert os.path.exists(log_file)
        assert os.path.getsize(log_file) > 0

    def test_handler_writes_logs_correctly(self, tmp_path: Path) -> None:
        """Handler writes log records to file correctly."""
        from concurrent_log_handler import ConcurrentRotatingFileHandler

        log_file = str(tmp_path / "test_write.log")
        handler = ConcurrentRotatingFileHandler(
            log_file,
            maxBytes=1024 * 1024,
            backupCount=1,
            encoding="utf-8",
        )
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)

        logger = logging.getLogger("test_write")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("Hello, World!")
        handler.close()
        logger.removeHandler(handler)

        content = Path(log_file).read_text(encoding="utf-8")
        assert "Hello, World!" in content

    def test_make_file_handler_returns_concurrent_handler(
        self, tmp_path: Path
    ) -> None:
        """_make_file_handler returns ConcurrentRotatingFileHandler."""
        from concurrent_log_handler import ConcurrentRotatingFileHandler

        from agent_ng.logging_config import _make_file_handler

        log_file = str(tmp_path / "test_make.log")
        handler = _make_file_handler(log_file)
        assert isinstance(handler, ConcurrentRotatingFileHandler)
        handler.close()

    def test_setup_logging_uses_concurrent_handler(
        self, tmp_path: Path
    ) -> None:
        """setup_logging() attaches a ConcurrentRotatingFileHandler."""
        from concurrent_log_handler import ConcurrentRotatingFileHandler

        from agent_ng.logging_config import setup_logging

        log_file = str(tmp_path / "test_setup.log")
        os.environ["LOG_FILE"] = log_file
        os.environ["LOG_FORCE"] = "true"
        try:
            root = setup_logging(force=True)
            # At least one handler should be ConcurrentRotatingFileHandler
            concurrent_handlers = [
                h
                for h in root.handlers
                if isinstance(h, ConcurrentRotatingFileHandler)
            ]
            assert len(concurrent_handlers) >= 1
        finally:
            os.environ.pop("LOG_FILE", None)
            os.environ.pop("LOG_FORCE", None)
            for h in root.handlers[:]:
                if isinstance(h, logging.FileHandler):
                    h.close()
                    root.removeHandler(h)
