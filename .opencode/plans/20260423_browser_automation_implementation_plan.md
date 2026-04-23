# Browser Automation Implementation Plan

> **Created:** 2026-04-23  
> **Status:** Draft - Awaiting Approval  
> **Goal:** Add browser automation capabilities to CMW Platform agent using agent-browser  
> **Estimated Time:** 10-13 days (3 phases)

---

## Executive Summary

Add browser automation tools to complement existing API-based tools, enabling access to UI-only platform features and providing visual verification capabilities.

**Key Decision:** Use **agent-browser** (not Playwright) for AI-native design and token efficiency.

**Approach:** Hybrid authentication - login via browser, extract cookies for API calls, use browser only for UI-only operations.

---

## Background

### Problem Statement

CMW Platform's OpenAPI specifications don't cover all platform features. Some operations are only accessible via web GUI:
- Visual workflow designers
- Complex admin panels
- Certain configuration screens
- UI-specific operations

### POC Results (2026-04-22)

✅ **FEASIBLE** - Browser automation tested successfully:
- Login automation works flawlessly
- Session management robust
- Element targeting reliable
- Screenshot capture functional
- Performance acceptable (10-15s for login + navigate)

**Key Finding:** CMW Platform is a Single Page Application (SPA) requiring custom wait strategies for dynamic content loading.

### URL Structure Discovery (2026-04-23)

✅ **COMPLETED** - Actual URL patterns documented:
- Settings/Administration: `#Settings/Administration`
- Global Security: `#Settings/globalSecurity`
- Groups: `#Settings/Groups`
- Solutions: `#solutions`
- Dashboard: `#desktop/`

**Pattern:** Hash-based routing with `#Settings/{Section}` for admin pages.

---

## Architecture Design

### Tool Organization

```
tools/
├── browser_tools/                    # New module
│   ├── __init__.py                   # Export all browser tools
│   ├── browser_session_manager.py   # Per-user browser context management
│   ├── tool_browser_login.py        # Authenticate via web UI
│   ├── tool_browser_navigate.py     # Navigate to CMW pages (SPA-aware)
│   ├── tool_browser_interact.py     # Click, fill, submit forms
│   ├── tool_browser_extract.py      # Extract data from UI tables/grids
│   ├── tool_browser_screenshot.py   # Visual verification
│   └── tool_browser_execute.py      # Execute arbitrary browser commands
```

### Integration Points

1. **Session Manager Integration**
   - Extend `agent_ng/session_manager.py` for browser contexts
   - Per-user isolation (existing pattern)
   - Automatic cleanup on session end

2. **Tool Registration**
   - Add to `agent_ng/langchain_agent.py` tool registry
   - Follow existing `@tool` decorator pattern
   - Pydantic schemas for input validation

3. **HTTP Session Integration**
   - Extract cookies from browser after login
   - Inject into `tools/requests_.py` HTTP session
   - Enable hybrid API + browser operations

---

## Implementation Phases

### Phase 1: Foundation (3-4 days)

**Goal:** Working browser authentication + session management

**Tasks:**
1. Create `tools/browser_tools/` module structure
2. Implement `BrowserSessionManager` class
   - Per-user browser context isolation
   - Session state persistence (JSON files)
   - Automatic cleanup on timeout
3. Build `tool_browser_login` with Pydantic schema
   - Navigate to login page
   - Fill credentials
   - Extract session cookies
   - Inject into HTTP session
4. Add browser session cleanup to `session_manager.py`
5. Write integration tests
6. Document authentication patterns

**Deliverables:**
- `tools/browser_tools/browser_session_manager.py`
- `tools/browser_tools/tool_browser_login.py`
- `agent_ng/_tests/test_browser_login.py`
- Documentation in skill references

**Success Criteria:**
- Agent can login via browser UI
- Cookies extracted and usable for API calls
- Session isolation verified (multiple users)
- Tests passing

---

### Phase 2: Core Operations (4-5 days)

**Goal:** Full browser automation capabilities

**Tasks:**
1. Implement `tool_browser_navigate`
   - SPA-aware navigation with custom waits
   - Content polling for dynamic loading
   - URL validation and error handling
2. Build `tool_browser_interact`
   - Click, fill, select operations
   - Element ref validation
   - Screenshot on error
3. Create `tool_browser_extract`
   - Extract data from UI tables/grids
   - Parse accessibility tree
   - Return structured data (JSON/CSV)
4. Implement `tool_browser_screenshot`
   - Full page and element screenshots
   - Annotated screenshots with refs
   - Save to session-specific directory
5. Build `tool_browser_execute`
   - Execute arbitrary agent-browser commands
   - Command validation and sanitization
   - Error handling with context
6. Register all tools in `langchain_agent.py`
7. Add comprehensive error handling
8. Write integration tests for each tool

**Deliverables:**
- 5 additional browser tools
- Integration with LangChain agent
- Error handling and logging
- Test suite with >80% coverage

**Success Criteria:**
- Agent can navigate CMW Platform UI
- Agent can interact with forms and buttons
- Agent can extract data from UI tables
- Screenshots captured for debugging
- All tests passing

---

### Phase 3: Advanced Features (3-4 days)

**Goal:** Production-ready optimizations

**Tasks:**
1. Implement hybrid authentication
   - Browser login → cookie extraction
   - Cookie injection into HTTP session
   - Automatic session refresh on expiry
2. Add session state persistence
   - Save browser state to `.browser-states/`
   - Restore state on agent restart
   - Cleanup old state files (>7 days)
3. Build UI-specific operation helpers
   - Workflow designer navigation
   - Visual editor interactions
   - Complex form handling
4. Optimize performance
   - Session reuse (avoid repeated logins)
   - Parallel operations (multiple tabs)
   - Snapshot caching
5. Implement visual verification
   - Screenshot comparison (baseline vs current)
   - Diff highlighting
   - Regression detection
6. Add monitoring and metrics
   - Operation timing
   - Success/failure rates
   - Session lifecycle tracking

**Deliverables:**
- Hybrid authentication working
- Performance optimizations
- UI-specific operation tools
- Monitoring dashboard integration

**Success Criteria:**
- Browser login enables API calls (cookie sharing)
- Session reuse reduces login overhead by 80%
- UI-only features accessible
- Performance metrics tracked

---

## Technical Specifications

### 1. Browser Session Manager

```python
# tools/browser_tools/browser_session_manager.py

from typing import Dict, Optional
import subprocess
import json
from pathlib import Path

class BrowserSession:
    """Represents a single browser session for a user."""
    
    def __init__(self, session_name: str, state_file: Path):
        self.session_name = session_name
        self.state_file = state_file
        self.is_active = False
    
    def execute(self, command: str) -> dict:
        """Execute agent-browser command."""
        full_cmd = f"agent-browser --session-name {self.session_name} {command}"
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def save_state(self):
        """Save browser state to file."""
        self.execute(f'state save "{self.state_file}"')
    
    def load_state(self):
        """Load browser state from file."""
        if self.state_file.exists():
            self.execute(f'state load "{self.state_file}"')
    
    def close(self):
        """Close browser session."""
        self.execute("close")
        self.is_active = False


class BrowserSessionManager:
    """Manages isolated browser contexts per user session."""
    
    _sessions: Dict[str, BrowserSession] = {}
    _state_dir = Path(".browser-states")
    
    @classmethod
    def get_or_create_session(cls, session_id: str) -> BrowserSession:
        """Get existing browser session or create new one."""
        if session_id not in cls._sessions:
            cls._state_dir.mkdir(exist_ok=True)
            state_file = cls._state_dir / f"cmw-{session_id}.json"
            
            session = BrowserSession(
                session_name=f"cmw-{session_id}",
                state_file=state_file
            )
            
            # Try to restore previous state
            session.load_state()
            session.is_active = True
            
            cls._sessions[session_id] = session
        
        return cls._sessions[session_id]
    
    @classmethod
    def cleanup_session(cls, session_id: str):
        """Close browser and cleanup session."""
        if session_id in cls._sessions:
            session = cls._sessions[session_id]
            session.save_state()
            session.close()
            del cls._sessions[session_id]
    
    @classmethod
    def cleanup_all(cls):
        """Close all browser sessions."""
        for session_id in list(cls._sessions.keys()):
            cls.cleanup_session(session_id)
```

### 2. Browser Login Tool

```python
# tools/browser_tools/tool_browser_login.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
from .browser_session_manager import BrowserSessionManager

class BrowserLoginInput(BaseModel):
    """Input schema for browser_login tool."""
    base_url: str = Field(
        description="CMW Platform base URL (e.g., https://platform.example.com/)"
    )
    username: str = Field(
        description="Login username"
    )
    password: str = Field(
        description="Login password"
    )
    session_id: str = Field(
        description="User session ID for isolation"
    )

@tool
def browser_login(
    base_url: str,
    username: str,
    password: str,
    session_id: str
) -> dict:
    """
    Authenticate to CMW Platform via browser UI.
    Extracts session cookies for subsequent API calls.
    
    Credentials are read from:
    - .env file (CMW_LOGIN, CMW_PASSWORD, CMW_BASE_URL) when CMW_USE_DOTENV=true
    - Browser state/session config when CMW_USE_DOTENV=false
    
    This follows the same pattern as existing API tools in tools/requests_.py
    
    Use this when:
    - Initial authentication to platform
    - Session has expired
    - Need to access UI-only features
    
    Returns:
        dict: {
            "success": bool,
            "message": str,
            "cookies_extracted": int,
            "screenshot": str (path to verification screenshot)
        }
    """
    try:
        # Get credentials using existing pattern from tools/requests_.py
        from tools.requests_ import _load_server_config
        config = _load_server_config()
        
        # Use provided credentials or fall back to config
        base_url = base_url or config.get("base_url")
        username = username or config.get("username")
        password = password or config.get("password")
        
        # Get or create browser session
        browser = BrowserSessionManager.get_or_create_session(session_id)
        
        # Navigate to platform (auto-redirects to login)
        result = browser.execute(f'open "{base_url}"')
        if not result["success"]:
            return {
                "success": False,
                "error": f"Failed to open platform: {result['stderr']}"
            }
        
        # Wait for page load
        browser.execute('wait --load networkidle')
        
        # Take snapshot to identify form elements
        snapshot_result = browser.execute('snapshot -i')
        
        # Fill username (typically @e8 based on POC)
        # Note: In production, parse snapshot to find correct refs
        browser.execute(f'fill @e8 "{username}"')
        
        # Fill password (typically @e9)
        browser.execute(f'fill @e9 "{password}"')
        
        # Take screenshot before submit
        screenshot_dir = Path(f".browser-states/screenshots/{session_id}")
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = screenshot_dir / "login_before_submit.png"
        browser.execute(f'screenshot "{screenshot_path}"')
        
        # Click submit button (typically @e2)
        browser.execute('click @e2')
        
        # Wait for navigation
        browser.execute('wait --load networkidle')
        browser.execute('wait 2000')  # Buffer for SPA
        
        # Verify login success
        url_result = browser.execute('get url')
        current_url = url_result["stdout"].strip()
        
        if "Login" in current_url or "LogOn" in current_url:
            # Still on login page - authentication failed
            error_screenshot = screenshot_dir / "login_failed.png"
            browser.execute(f'screenshot "{error_screenshot}"')
            return {
                "success": False,
                "error": "Authentication failed - still on login page",
                "screenshot": str(error_screenshot)
            }
        
        # Take success screenshot
        success_screenshot = screenshot_dir / "login_success.png"
        browser.execute(f'screenshot "{success_screenshot}"')
        
        # Extract cookies for HTTP session
        # Note: Implement cookie extraction and injection
        cookies = _extract_cookies(browser)
        _inject_cookies_to_http_session(session_id, cookies)
        
        # Save browser state for reuse
        browser.save_state()
        
        return {
            "success": True,
            "message": "Authenticated successfully",
            "cookies_extracted": len(cookies),
            "screenshot": str(success_screenshot)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Browser login failed: {str(e)}"
        }


def _extract_cookies(browser: BrowserSession) -> list:
    """Extract cookies from browser session."""
    # Implementation: Parse browser state file or use agent-browser cookies command
    # For now, placeholder
    return []


def _inject_cookies_to_http_session(session_id: str, cookies: list):
    """Inject browser cookies into HTTP session for API calls."""
    # Implementation: Update tools/requests_.py session with cookies
    # For now, placeholder
    pass
```

### 3. SPA Navigation Helper

```python
# tools/browser_tools/tool_browser_navigate.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
from .browser_session_manager import BrowserSessionManager
from pathlib import Path
import time

class BrowserNavigateInput(BaseModel):
    """Input schema for browser_navigate tool."""
    url: str = Field(
        description="Full URL or hash fragment (e.g., #Settings/Administration, #solutions, #desktop/)"
    )
    wait_for_text: Optional[str] = Field(
        default=None,
        description="Text to wait for after navigation (for SPA content)"
    )
    session_id: str = Field(
        description="User session ID"
    )

@tool
def browser_navigate(
    url: str,
    wait_for_text: Optional[str],
    session_id: str
) -> dict:
    """
    Navigate to CMW Platform page with SPA-aware waiting.
    
    Supports hash-based routing patterns:
    - #Settings/Administration - Admin settings
    - #Settings/globalSecurity - Security settings
    - #Settings/Groups - Groups management
    - #solutions - Solutions/applications
    - #desktop/ - Main dashboard
    
    Use this when:
    - Navigating to admin pages
    - Accessing UI-only features
    - Need to wait for dynamic content
    
    Returns:
        dict: {
            "success": bool,
            "current_url": str,
            "snapshot": str (accessibility tree),
            "screenshot": str (path)
        }
    """
    try:
        browser = BrowserSessionManager.get_or_create_session(session_id)
        
        # Navigate
        result = browser.execute(f'open "{url}"')
        if not result["success"]:
            return {
                "success": False,
                "error": f"Navigation failed: {result['stderr']}"
            }
        
        # Wait for network idle
        browser.execute('wait --load networkidle')
        
        # Wait for specific content if provided
        if wait_for_text:
            browser.execute(f'wait --text "{wait_for_text}"')
        
        # Additional buffer for SPA rendering
        browser.execute('wait 2000')
        
        # Take snapshot
        snapshot_result = browser.execute('snapshot -i')
        
        # Get current URL
        url_result = browser.execute('get url')
        
        # Take screenshot
        screenshot_dir = Path(f".browser-states/screenshots/{session_id}")
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = screenshot_dir / f"navigate_{int(time.time())}.png"
        browser.execute(f'screenshot "{screenshot_path}"')
        
        return {
            "success": True,
            "current_url": url_result["stdout"].strip(),
            "snapshot": snapshot_result["stdout"],
            "screenshot": str(screenshot_path)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Navigation failed: {str(e)}"
        }
```

---

## Testing Strategy

### Unit Tests

```python
# agent_ng/_tests/test_browser_session_manager.py

def test_session_creation():
    """Test browser session creation."""
    manager = BrowserSessionManager()
    session = manager.get_or_create_session("test-user-1")
    assert session.session_name == "cmw-test-user-1"
    assert session.is_active

def test_session_isolation():
    """Test multiple users have isolated sessions."""
    manager = BrowserSessionManager()
    session1 = manager.get_or_create_session("user-1")
    session2 = manager.get_or_create_session("user-2")
    assert session1.session_name != session2.session_name

def test_session_cleanup():
    """Test session cleanup."""
    manager = BrowserSessionManager()
    session = manager.get_or_create_session("test-user")
    manager.cleanup_session("test-user")
    assert "test-user" not in manager._sessions
```

### Integration Tests

```python
# agent_ng/_tests/test_browser_login.py

def test_browser_login_success():
    """Test successful browser login."""
    # Note: Use test credentials, not production
    result = browser_login(
        base_url="https://test-platform.example.com/",
        username="test_user",
        password="test_password",
        session_id="test-session"
    )
    assert result["success"] == True
    assert result["cookies_extracted"] > 0

def test_browser_login_failure():
    """Test login with invalid credentials."""
    result = browser_login(
        base_url="https://test-platform.example.com/",
        username="invalid",
        password="invalid",
        session_id="test-session"
    )
    assert result["success"] == False
    assert "failed" in result["error"].lower()
```

---

## Dependencies

### New Dependencies

None! agent-browser is already installed (v0.25.4).

### Environment Variables

```bash
# .env additions (no sensitive data in plan)

# Browser automation settings
BROWSER_SESSION_TIMEOUT=3600          # Auto-cleanup after 1 hour (seconds)
BROWSER_DEFAULT_WAIT=30000            # Default timeout (milliseconds)
BROWSER_HEADLESS=true                 # Headless mode (true/false)
BROWSER_STATE_DIR=.browser-states     # Session state storage
BROWSER_SCREENSHOT_DIR=screenshots    # Screenshot storage (relative to state dir)

# Credentials (same as existing API tools)
# CMW_BASE_URL, CMW_LOGIN, CMW_PASSWORD already defined above
# CMW_USE_DOTENV controls whether to use .env or browser state config
```

---

## Risk Mitigation

| Risk | Severity | Mitigation |
|------|----------|------------|
| **SPA content timing** | Medium | Custom wait strategies, content polling, retry logic |
| **Session expiry** | Medium | Auto-refresh logic, timeout detection, re-login |
| **UI changes** | Low | Semantic selectors (roles, labels), not CSS classes |
| **Performance overhead** | Low | Use browser only for UI-only ops, cache sessions |
| **Debugging difficulty** | Medium | Medium | Screenshot on error, detailed logging |
| **State corruption** | Medium | JSON snapshots before/after, rollback capability |
| **Concurrent access** | Low | Per-user session isolation, unique session names |

---

## Success Metrics

### Functional Metrics
- ✅ Agent can authenticate via browser UI
- ✅ Agent can navigate to any CMW Platform page
- ✅ Agent can interact with forms, buttons, dropdowns
- ✅ Agent can extract data from UI tables
- ✅ Browser sessions isolated per user
- ✅ Cookies shared between browser and HTTP sessions

### Performance Metrics
- Browser login: <10 seconds
- Page navigation: <5 seconds
- Element interaction: <2 seconds
- Session reuse: <1 second (no re-login)

### Quality Metrics
- Test coverage: >80%
- Error handling: All failure modes covered
- Documentation: Complete with examples
- Code quality: Passes ruff/mypy checks

---

## Rollout Plan

### Phase 1 Rollout (Week 1)
1. Implement browser session manager
2. Implement browser login tool
3. Write tests
4. Internal testing with test instance
5. Documentation

### Phase 2 Rollout (Week 2)
1. Implement remaining browser tools
2. Integration with LangChain agent
3. Comprehensive testing
4. User acceptance testing
5. Documentation updates

### Phase 3 Rollout (Week 3)
1. Performance optimizations
2. Monitoring integration
3. Production deployment
4. User training

---

## Open Questions

1. **Cookie Extraction:** What's the best method to extract cookies from agent-browser?
   - Option A: Parse state JSON file
   - Option B: Use agent-browser cookies command
   - Option C: Intercept network requests
   - **Recommendation:** Option B (cleaner API)

2. **Session Timeout:** What's the actual CMW Platform session timeout?
   - POC observed ~15-20 minutes
   - Need to confirm with platform admin
   - **Recommendation:** Make configurable via env var

3. **URL Pattern Coverage:** Are there additional admin pages beyond discovered patterns?
   - Discovered: #Settings/Administration, #Settings/globalSecurity, #Settings/Groups, #solutions, #desktop/
   - May need to explore more sections during Phase 2
   - **Recommendation:** Document additional patterns as discovered

---

## Approval Checklist

- [ ] Architecture reviewed and approved
- [ ] Phase 1 scope confirmed
- [ ] Test instance access provided
- [ ] Environment variables configured
- [ ] Success metrics agreed upon
- [ ] Rollout plan approved

---

## Next Steps

1. **Review this plan** - Provide feedback and approvals
2. **Answer open questions** - Clarify technical decisions
3. **Approve Phase 1** - Green light to start implementation
4. **Provide test instance** - Access to CMW Platform for testing
5. **Start implementation** - Begin Phase 1 development

---

**Plan Status:** 📋 Draft - Awaiting Approval  
**Estimated Start:** Upon approval  
**Estimated Completion:** 10-13 days after start
