"""
Tests for Browser Automation
============================

Tests browser session management and automation tools.

Key test areas:
- Session lifecycle management
- State persistence
- Tool invocation
- Error handling
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Test browser session management
try:
    from agent_ng.browser_session import (
        BrowserSession,
        BrowserSessionManager,
        get_session_manager,
        STATE_DIR,
    )
    BROWSER_SESSION_AVAILABLE = True
except ImportError:
    BROWSER_SESSION_AVAILABLE = False

# Test browser tools
try:
    from tools.browser_tools import (
        navigate_to_page,
        click_element,
        fill_form_field,
        take_screenshot,
        login_to_platform,
    )
    BROWSER_TOOLS_AVAILABLE = True
except ImportError:
    BROWSER_TOOLS_AVAILABLE = False


@pytest.mark.skipif(not BROWSER_SESSION_AVAILABLE, reason="Browser session not available")
class TestBrowserSession:
    """Test browser session management."""

    @pytest.mark.asyncio
    async def test_session_creation(self):
        """Test creating a browser session."""
        with patch('agent_ng.browser_session.async_playwright') as mock_playwright:
            # Mock Playwright components
            mock_page = AsyncMock()
            mock_context = AsyncMock()
            mock_browser = AsyncMock()
            mock_pw = AsyncMock()

            mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=mock_page)

            mock_playwright.return_value.start = AsyncMock(return_value=mock_pw)

            session = BrowserSession("test_session")
            page = await session.start()

            assert page is not None
            assert session.page == mock_page
            assert session.context == mock_context

            await session.close()

    @pytest.mark.asyncio
    async def test_session_state_persistence(self):
        """Test saving and loading session state."""
        with patch('agent_ng.browser_session.async_playwright') as mock_playwright:
            # Mock Playwright components
            mock_page = AsyncMock()
            mock_context = AsyncMock()
            mock_browser = AsyncMock()
            mock_pw = AsyncMock()

            mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=mock_page)
            mock_context.storage_state = AsyncMock()

            mock_playwright.return_value.start = AsyncMock(return_value=mock_pw)

            session = BrowserSession("test_state")
            await session.start()

            # Save state
            result = await session.save_state()
            assert result is True
            mock_context.storage_state.assert_called_once()

            await session.close()

    @pytest.mark.asyncio
    async def test_session_context_manager(self):
        """Test using session as async context manager."""
        with patch('agent_ng.browser_session.async_playwright') as mock_playwright:
            # Mock Playwright components
            mock_page = AsyncMock()
            mock_context = AsyncMock()
            mock_browser = AsyncMock()
            mock_pw = AsyncMock()

            mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=mock_page)

            mock_playwright.return_value.start = AsyncMock(return_value=mock_pw)

            async with BrowserSession("test_context") as session:
                assert session.page is not None


@pytest.mark.skipif(not BROWSER_SESSION_AVAILABLE, reason="Browser session not available")
class TestBrowserSessionManager:
    """Test browser session manager."""

    @pytest.mark.asyncio
    async def test_get_or_create_session(self):
        """Test getting or creating sessions."""
        with patch('agent_ng.browser_session.async_playwright') as mock_playwright:
            # Mock Playwright components
            mock_page = AsyncMock()
            mock_context = AsyncMock()
            mock_browser = AsyncMock()
            mock_pw = AsyncMock()

            mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=mock_page)

            mock_playwright.return_value.start = AsyncMock(return_value=mock_pw)

            manager = BrowserSessionManager()

            # Get new session
            session1 = await manager.get_session("test1")
            assert session1 is not None

            # Get same session again
            session2 = await manager.get_session("test1")
            assert session1 is session2

            # Get different session
            session3 = await manager.get_session("test2")
            assert session3 is not session1

            await manager.close_all()

    @pytest.mark.asyncio
    async def test_close_session(self):
        """Test closing specific session."""
        with patch('agent_ng.browser_session.async_playwright') as mock_playwright:
            # Mock Playwright components
            mock_page = AsyncMock()
            mock_context = AsyncMock()
            mock_browser = AsyncMock()
            mock_pw = AsyncMock()

            mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=mock_page)

            mock_playwright.return_value.start = AsyncMock(return_value=mock_pw)

            manager = BrowserSessionManager()

            await manager.get_session("test1")
            await manager.get_session("test2")

            assert len(manager.list_sessions()) == 2

            await manager.close_session("test1")

            assert len(manager.list_sessions()) == 1
            assert "test2" in manager.list_sessions()

            await manager.close_all()


@pytest.mark.skipif(not BROWSER_TOOLS_AVAILABLE, reason="Browser tools not available")
class TestBrowserTools:
    """Test browser automation tools."""

    @pytest.mark.asyncio
    async def test_navigate_to_page(self):
        """Test navigate_to_page tool."""
        with patch('tools.browser_tools.get_session_manager') as mock_get_manager:
            # Mock session and page
            mock_page = AsyncMock()
            mock_page.goto = AsyncMock()
            mock_page.title = AsyncMock(return_value="Test Page")

            mock_session = Mock()
            mock_session.page = mock_page

            mock_manager = AsyncMock()
            mock_manager.get_session = AsyncMock(return_value=mock_session)
            mock_get_manager.return_value = mock_manager

            result = await navigate_to_page.ainvoke({
                "url": "https://example.com",
                "session_name": "test"
            })

            assert "Successfully navigated" in result
            assert "Test Page" in result
            mock_page.goto.assert_called_once()

    @pytest.mark.asyncio
    async def test_click_element(self):
        """Test click_element tool."""
        with patch('tools.browser_tools.get_session_manager') as mock_get_manager:
            # Mock session and page
            mock_page = AsyncMock()
            mock_page.click = AsyncMock()

            mock_session = Mock()
            mock_session.page = mock_page

            mock_manager = AsyncMock()
            mock_manager.get_session = AsyncMock(return_value=mock_session)
            mock_get_manager.return_value = mock_manager

            result = await click_element.ainvoke({
                "selector": "button.submit",
                "session_name": "test"
            })

            assert "Successfully clicked" in result
            mock_page.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_fill_form_field(self):
        """Test fill_form_field tool."""
        with patch('tools.browser_tools.get_session_manager') as mock_get_manager:
            # Mock session and page
            mock_page = AsyncMock()
            mock_page.fill = AsyncMock()

            mock_session = Mock()
            mock_session.page = mock_page

            mock_manager = AsyncMock()
            mock_manager.get_session = AsyncMock(return_value=mock_session)
            mock_get_manager.return_value = mock_manager

            result = await fill_form_field.ainvoke({
                "selector": "input[name='username']",
                "value": "testuser",
                "session_name": "test"
            })

            assert "Successfully filled" in result
            mock_page.fill.assert_called_once()

    @pytest.mark.asyncio
    async def test_take_screenshot(self):
        """Test take_screenshot tool."""
        with patch('tools.browser_tools.get_session_manager') as mock_get_manager:
            # Mock session and page
            mock_page = AsyncMock()
            mock_page.screenshot = AsyncMock()

            mock_session = Mock()
            mock_session.page = mock_page

            mock_manager = AsyncMock()
            mock_manager.get_session = AsyncMock(return_value=mock_session)
            mock_get_manager.return_value = mock_manager

            result = await take_screenshot.ainvoke({
                "filename": "test.png",
                "session_name": "test"
            })

            assert "Screenshot saved" in result
            mock_page.screenshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_login_to_platform(self):
        """Test login_to_platform tool."""
        with patch('tools.browser_tools.get_session_manager') as mock_get_manager:
            # Mock session and page
            mock_page = AsyncMock()
            mock_page.goto = AsyncMock()
            mock_page.fill = AsyncMock()
            mock_page.click = AsyncMock()
            mock_page.wait_for_load_state = AsyncMock()

            mock_session = AsyncMock()
            mock_session.page = mock_page
            mock_session.save_state = AsyncMock(return_value=True)

            mock_manager = AsyncMock()
            mock_manager.get_session = AsyncMock(return_value=mock_session)
            mock_get_manager.return_value = mock_manager

            result = await login_to_platform.ainvoke({
                "base_url": "https://platform.example.com",
                "username": "testuser",
                "password": "testpass",
                "session_name": "test"
            })

            assert "Successfully logged in" in result
            mock_page.goto.assert_called_once()
            assert mock_page.fill.call_count == 2  # username and password
            mock_page.click.assert_called_once()
            mock_session.save_state.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
