"""
Browser Session Management for CMW Platform
===========================================

Manages browser automation sessions using Playwright for CMW Platform interactions.
Provides session lifecycle management, state persistence, and error handling.

Key Features:
- Session creation and cleanup
- State persistence (cookies, storage)
- Context isolation
- Error handling and recovery
- Async/await support

Based on Playwright Python best practices.
"""

import asyncio
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Optional

try:
    from playwright.async_api import Browser, BrowserContext, Page, async_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Browser = Any
    BrowserContext = Any
    Page = Any

logger = logging.getLogger(__name__)

# Session state directory
STATE_DIR = Path(".browser-states")
STATE_DIR.mkdir(exist_ok=True)


class BrowserSession:
    """Manages a single browser session with state persistence."""

    def __init__(self, session_name: str = "default") -> None:
        """
        Initialize browser session.

        Args:
            session_name: Unique identifier for this session
        """
        if not PLAYWRIGHT_AVAILABLE:
            msg = (
                "Playwright not available. Install with: pip install playwright && "
                "playwright install chromium"
            )
            raise ImportError(msg)

        self.session_name = session_name
        self.state_file = STATE_DIR / f"cmw-{session_name}.json"
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def start(self) -> Page:
        """
        Start browser session and return page.

        Returns:
            Page object for browser automation
        """
        logger.info("Starting browser session: %s", self.session_name)

        # Launch Playwright
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"]
        )

        # Create context with state if available
        context_options = {
            "viewport": {"width": 1920, "height": 1080},
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            ),
        }

        if self.state_file.exists():
            logger.info("Loading session state from %s", self.state_file)
            context_options["storage_state"] = str(self.state_file)

        self._context = await self._browser.new_context(**context_options)
        self._page = await self._context.new_page()

        logger.info("Browser session started successfully")
        return self._page

    async def save_state(self) -> bool:
        """
        Save current session state (cookies, storage).

        Returns:
            True if state saved successfully
        """
        if not self._context:
            logger.warning("No context to save state from")
            return False

        try:
            await self._context.storage_state(path=str(self.state_file))
            size = self.state_file.stat().st_size
            size_str = f"{size:,}"
            logger.info("Session state saved: %s (%s bytes)", self.state_file, size_str)
            return True
        except Exception:
            logger.exception("Failed to save session state")
            return False

    async def close(self):
        """Close browser session and cleanup resources."""
        logger.info("Closing browser session: %s", self.session_name)

        if self._page:
            await self._page.close()
            self._page = None

        if self._context:
            await self._context.close()
            self._context = None

        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        logger.info("Browser session closed")

    @property
    def page(self) -> Page | None:
        """Get current page object."""
        return self._page

    @property
    def context(self) -> BrowserContext | None:
        """Get current browser context."""
        return self._context


class BrowserSessionManager:
    """Manages multiple browser sessions."""

    def __init__(self) -> None:
        """Initialize session manager."""
        self._sessions: dict[str, BrowserSession] = {}

    async def get_session(self, session_name: str = "default") -> BrowserSession:
        """
        Get or create browser session.

        Args:
            session_name: Session identifier

        Returns:
            BrowserSession instance
        """
        if session_name not in self._sessions:
            session = BrowserSession(session_name)
            await session.start()
            self._sessions[session_name] = session
            logger.info("Created new session: %s", session_name)
        else:
            logger.info("Reusing existing session: %s", session_name)

        return self._sessions[session_name]

    async def close_session(self, session_name: str) -> None:
        """
        Close and remove session.

        Args:
            session_name: Session to close
        """
        if session_name in self._sessions:
            await self._sessions[session_name].close()
            del self._sessions[session_name]
            logger.info("Closed session: %s", session_name)

    async def close_all(self):
        """Close all active sessions."""
        for session_name in list(self._sessions.keys()):
            await self.close_session(session_name)
        logger.info("All sessions closed")

    def list_sessions(self) -> list[str]:
        """List active session names."""
        return list(self._sessions.keys())


# Global session manager instance (module-level singleton)
_session_manager: BrowserSessionManager | None = None


def get_session_manager() -> BrowserSessionManager:
    """Get global session manager instance."""
    global _session_manager  # noqa: PLW0603 - intentional module-level singleton
    if _session_manager is None:
        _session_manager = BrowserSessionManager()
    return _session_manager
