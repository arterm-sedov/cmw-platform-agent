"""
Browser Automation Tools for CMW Platform
=========================================

LangChain tools for browser automation using Playwright.
Enables UI-only features and visual verification.

Key Features:
- Navigate to CMW Platform pages
- Login and authentication
- Click buttons and interact with UI
- Take screenshots for verification
- Extract data from UI elements

Based on LangChain tool patterns and Playwright best practices.
"""

import asyncio
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

try:
    from agent_ng.browser_session import BrowserSession, get_session_manager

    BROWSER_AVAILABLE = True
except ImportError:
    BROWSER_AVAILABLE = False

logger = logging.getLogger(__name__)


class NavigateInput(BaseModel):
    """Input for navigate_to_page tool."""

    url: str = Field(description="URL to navigate to")
    session_name: str = Field(
        default="default", description="Browser session name for isolation"
    )


class ClickElementInput(BaseModel):
    """Input for click_element tool."""

    selector: str = Field(description="CSS selector or text to click")
    session_name: str = Field(default="default", description="Browser session name")


class FillFormInput(BaseModel):
    """Input for fill_form_field tool."""

    selector: str = Field(description="CSS selector for input field")
    value: str = Field(description="Value to fill in")
    session_name: str = Field(default="default", description="Browser session name")


class ScreenshotInput(BaseModel):
    """Input for take_screenshot tool."""

    filename: str | None = Field(
        default=None,
        description="Filename to save screenshot (auto-generated if not provided)",
    )
    session_name: str = Field(default="default", description="Browser session name")


class LoginInput(BaseModel):
    """Input for login_to_platform tool."""

    base_url: str = Field(
        description="CMW Platform base URL (e.g., https://platform.example.com)"
    )
    username: str = Field(description="Username for login")
    password: str = Field(description="Password for login")
    session_name: str = Field(default="default", description="Browser session name")


@tool(args_schema=NavigateInput)
async def navigate_to_page(url: str, session_name: str = "default") -> str:
    """
    Navigate browser to a URL.

    Use this to open CMW Platform pages or any web page.

    Args:
        url: Full URL to navigate to
        session_name: Browser session identifier

    Returns:
        Success message with page title
    """
    if not BROWSER_AVAILABLE:
        return "Error: Browser automation not available. Install playwright."

    try:
        manager = get_session_manager()
        session = await manager.get_session(session_name)
        page = session.page

        if not page:
            return f"Error: No page available in session {session_name}"

        await page.goto(url, wait_until="networkidle", timeout=30000)
        title = await page.title()

        logger.info("Navigated to %s - Title: %s", url, title)
        return f"Successfully navigated to {url}\nPage title: {title}"

    except Exception:
        logger.exception("Navigation failed")
        return f"Error navigating to {url}"


@tool(args_schema=ClickElementInput)
async def click_element(selector: str, session_name: str = "default") -> str:
    """
    Click an element on the page.

    Use this to click buttons, links, or any clickable element.

    Args:
        selector: CSS selector or text content to click
        session_name: Browser session identifier

    Returns:
        Success or error message
    """
    if not BROWSER_AVAILABLE:
        return "Error: Browser automation not available."

    try:
        manager = get_session_manager()
        session = await manager.get_session(session_name)
        page = session.page

        if not page:
            return f"Error: No page available in session {session_name}"

        # Try CSS selector first, then text content
        try:
            await page.click(selector, timeout=10000)
        except Exception:
            # Try as text content
            await page.get_by_text(selector).click(timeout=10000)

        logger.info("Clicked element: %s", selector)
        return f"Successfully clicked: {selector}"

    except Exception:
        logger.exception("Click failed")
        return f"Error clicking {selector}"


@tool(args_schema=FillFormInput)
async def fill_form_field(
    selector: str, value: str, session_name: str = "default"
) -> str:
    """
    Fill a form field with a value.

    Use this to enter text into input fields, textareas, etc.

    Args:
        selector: CSS selector for the input field
        value: Text value to enter
        session_name: Browser session identifier

    Returns:
        Success or error message
    """
    if not BROWSER_AVAILABLE:
        return "Error: Browser automation not available."

    try:
        manager = get_session_manager()
        session = await manager.get_session(session_name)
        page = session.page

        if not page:
            return f"Error: No page available in session {session_name}"

        await page.fill(selector, value, timeout=10000)

        logger.info("Filled field %s with value", selector)
        return f"Successfully filled field: {selector}"

    except Exception:
        logger.exception("Fill failed")
        return f"Error filling {selector}"


@tool(args_schema=ScreenshotInput)
async def take_screenshot(
    filename: str | None = None, session_name: str = "default"
) -> str:
    """
    Take a screenshot of the current page.

    Use this for visual verification or debugging.

    Args:
        filename: Optional filename (auto-generated if not provided)
        session_name: Browser session identifier

    Returns:
        Path to saved screenshot
    """
    if not BROWSER_AVAILABLE:
        return "Error: Browser automation not available."

    try:
        manager = get_session_manager()
        session = await manager.get_session(session_name)
        page = session.page

        if not page:
            return f"Error: No page available in session {session_name}"

        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{session_name}_{timestamp}.png"

        # Ensure screenshots directory exists
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)

        filepath = screenshots_dir / filename
        await page.screenshot(path=str(filepath), full_page=True)

        logger.info("Screenshot saved: %s", filepath)
        return f"Screenshot saved: {filepath}"

    except Exception:
        logger.exception("Screenshot failed")
        return "Error taking screenshot"


@tool(args_schema=LoginInput)
async def login_to_platform(
    base_url: str, username: str, password: str, session_name: str = "default"
) -> str:
    """
    Login to CMW Platform.

    Use this to authenticate before accessing platform features.

    Args:
        base_url: Platform base URL
        username: Login username
        password: Login password
        session_name: Browser session identifier

    Returns:
        Success or error message
    """
    if not BROWSER_AVAILABLE:
        return "Error: Browser automation not available."

    try:
        manager = get_session_manager()
        session = await manager.get_session(session_name)
        page = session.page

        if not page:
            return f"Error: No page available in session {session_name}"

        # Navigate to login page
        login_url = f"{base_url.rstrip('/')}/Account/LogOn"
        await page.goto(login_url, wait_until="networkidle", timeout=30000)

        # Fill login form
        await page.fill('input[name="UserName"]', username, timeout=10000)
        await page.fill('input[name="Password"]', password, timeout=10000)

        # Submit form
        await page.click('button[type="submit"]', timeout=10000)

        # Wait for navigation
        await page.wait_for_load_state("networkidle", timeout=30000)

        # Save session state for reuse
        await session.save_state()

        logger.info("Logged in to %s as %s", base_url, username)
        return f"Successfully logged in to {base_url} as {username}"

    except Exception:
        logger.exception("Login failed")
        return "Error logging in"


# Export all browser tools
__all__ = [
    "click_element",
    "fill_form_field",
    "login_to_platform",
    "navigate_to_page",
    "take_screenshot",
]
