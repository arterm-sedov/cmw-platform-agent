# Browser Automation Reference

> **Created:** 2026-04-22  
> **Status:** Implementation Guide  
> **Related:** See `.opencode/plans/20260423_browser_automation_implementation_plan.md`

---

## Overview

Browser automation enables access to CMW Platform features not available via API, including visual workflow designers, complex admin panels, and UI-specific operations.

**Tool:** agent-browser v0.25.4 (AI-native, ref-based targeting)

---

## Quick Start

### 1. Login

```python
from tools.browser_tools.tool_browser_login import browser_login

result = browser_login.invoke({
    "base_url": "https://your-platform.example.com/",
    "username": "your_username",
    "password": "your_password",
    "session_id": "user-session-id"
})

# Result includes:
# - success: bool
# - cookies_extracted: int
# - screenshot: path to verification image
```

### 2. Navigate

```python
from tools.browser_tools.tool_browser_navigate import browser_navigate

result = browser_navigate.invoke({
    "url": "#Settings/Administration",
    "wait_for_text": "Administration",  # SPA content indicator
    "session_id": "user-session-id"
})

# Result includes:
# - current_url: str
# - snapshot: accessibility tree with refs
# - screenshot: path
```

### 3. Interact

```python
from tools.browser_tools.tool_browser_interact import browser_interact

# Click button
browser_interact.invoke({
    "action": "click",
    "element_ref": "@e5",
    "session_id": "user-session-id"
})

# Fill form
browser_interact.invoke({
    "action": "fill",
    "element_ref": "@e8",
    "value": "New Value",
    "session_id": "user-session-id"
})
```

---

## CMW Platform Specifics

### Login Flow

**Correct URL:** Platform auto-redirects to `/Home/Login/?returnUrl=/` when unauthenticated.

**Form Structure:**
- Username field: textbox "E-mail address or username"
- Password field: textbox "••••••••••" (password field)
- Submit button: button "Log in"

**Success Indicator:** 
- URL changes from `/Home/Login/?returnUrl=/` to `/#desktop/`
- Page title changes from "Comindware Platform" to app-specific title (e.g., "Рабочий стол")

**Credentials:** Always load from `.env` file:
```python
from dotenv import load_dotenv
import os
load_dotenv()
username = os.environ["CMW_LOGIN"]
password = os.environ["CMW_PASSWORD"]
```

### SPA Navigation

CMW Platform is a Single Page Application with hash-based routing.

**Hash Navigation via JavaScript:**
```javascript
// playwright-cli or agent-browser eval
window.location.hash = '#RecordType/oa.193/Administration';
```

**After hash change:** Always re-snapshot to get fresh element refs. Page title updates to reflect current context.

**Example title progression:**
- Login: "Comindware Platform"
- Dashboard: "Рабочий стол"
- Template admin: "Управление объектами недвижимости > Шаблоны > Планы техобслуживания > Свойства"
- Template operations: "Управление объектами недвижимости > Шаблоны > Планы техобслуживания > Кнопки"

| Page | URL | Wait Indicator |
|------|-----|----------------|
| Dashboard | `/#desktop/` | "Welcome" or "Рабочий стол" |
| Administration | `/#Settings/Administration` | "Administration" |
| Global Security | `/#Settings/globalSecurity` | "Security" |
| Groups | `/#Settings/Groups` | "Группы" |
| Solutions | `/#solutions` | "Solutions" |
| Solution Admin | `/#solutions/sln.23/Administration` | "Приложение" |
| Template Admin | `/#RecordType/oa.193/Administration` | Template name in breadcrumb |
| Template Operations | `/#RecordType/oa.193/Operations` | "Кнопки" |
| Templates List | `/#solutions/sln.23/templates/showall/...` | "Шаблоны" |

**Critical:** Always use `wait_for_text` parameter for SPA pages. Content loads dynamically after URL change.

**UI Localization:** Russian UI shows Russian labels (e.g., "Приложение", "Шаблоны", "Кнопки"). Use `ref=` from snapshot for reliable element targeting, not text labels.

**Navigation Sidebar:** Has collapsible sections with submenu items. After clicking navigation items, always re-snapshot.

### Navigation Pattern

```python
def navigate_to_admin_page(page_name: str, session_id: str):
    """Navigate to admin page with proper SPA handling."""
    
    pages = {
        "administration": {
            "url": "/#Settings/Administration",
            "wait_text": "Administration"
        },
        "security": {
            "url": "/#Settings/globalSecurity",
            "wait_text": "Security"
        },
        "groups": {
            "url": "/#Settings/Groups",
            "wait_text": "Группы"
        },
        "solutions": {
            "url": "/#solutions",
            "wait_text": "Solutions"
        }
    }
    
    if page_name not in pages:
        return {"success": False, "error": f"Unknown page: {page_name}"}
    
    page_info = pages[page_name]
    
    result = browser_navigate.invoke({
        "url": f"https://your-platform.example.com{page_info['url']}",
        "wait_for_text": page_info["wait_text"],
        "session_id": session_id
    })
    
    return result
```

---

## Element Identification

### Ref System

agent-browser uses refs (`@e1`, `@e2`, etc.) to identify elements:

```bash
# Take snapshot to get refs
agent-browser snapshot -i

# Output:
# - textbox "Email" [ref=e1]
# - textbox "Password" [ref=e2]
# - button "Submit" [ref=e3]

# Use refs in commands
agent-browser fill @e1 "user@example.com"
agent-browser fill @e2 "password"
agent-browser click @e3
```

### Ref Lifecycle

**Important:** Refs invalidate after page changes.

| Event | Refs Valid? | Action |
|-------|-------------|--------|
| Initial snapshot | ✅ Yes | Use refs |
| Click link/button | ❌ No | Re-snapshot |
| Form submission | ❌ No | Re-snapshot |
| SPA navigation | ❌ No | Re-snapshot |
| Dynamic content load | ⚠️ Maybe | Re-snapshot to be safe |

**Pattern:**
```python
# 1. Snapshot
snapshot = browser_navigate.invoke(...)

# 2. Interact
browser_interact.invoke({"action": "click", "element_ref": "@e5", ...})

# 3. Re-snapshot after navigation
snapshot = browser_navigate.invoke(...)

# 4. Use new refs
browser_interact.invoke({"action": "fill", "element_ref": "@e8", ...})
```

### Finding Elements

**From snapshot output:**
```
- link "Системные роли" [ref=e31]
- button "Создать" [ref=e42]
- textbox "Поиск" [ref=e21]
```

**Parse refs programmatically:**
```python
import re

def extract_refs(snapshot_text: str) -> dict:
    """Extract element refs from snapshot."""
    refs = {}
    pattern = r'- (\w+) "([^"]+)" \[ref=e(\d+)\]'
    
    for match in re.finditer(pattern, snapshot_text):
        element_type, label, ref_num = match.groups()
        refs[label] = f"@e{ref_num}"
    
    return refs

# Usage
snapshot = browser_navigate.invoke(...)["snapshot"]
refs = extract_refs(snapshot)
roles_link = refs.get("Системные роли")  # "@e31"
```

---

## Session Management

### Per-User Isolation

Each user gets isolated browser context:

```python
# User 1
browser_login.invoke({..., "session_id": "user-alice-123"})

# User 2
browser_login.invoke({..., "session_id": "user-bob-456"})

# Sessions are completely isolated
# - Separate cookies
# - Separate localStorage
# - Separate browser instances
```

### State Persistence

Browser state saved automatically:

**Location:** `.browser-states/cmw-{session_id}.json`

**Contents:**
- Cookies
- localStorage
- sessionStorage
- Current URL

**Restoration:** Automatic on next tool call with same `session_id`

### Session Lifecycle

```python
# 1. First use - creates new session
browser_login.invoke({..., "session_id": "user-123"})

# 2. Subsequent uses - restores session
browser_navigate.invoke({..., "session_id": "user-123"})

# 3. Auto-cleanup after timeout (default: 1 hour)
# Or manual cleanup:
from tools.browser_tools.browser_session_manager import BrowserSessionManager
BrowserSessionManager.cleanup_session("user-123")
```

---

## Hybrid Authentication

Browser login extracts cookies for API calls:

```python
# 1. Login via browser
result = browser_login.invoke({
    "base_url": "https://platform.example.com/",
    "username": "user",
    "password": "pass",
    "session_id": "user-123"
})

# 2. Cookies automatically injected into HTTP session
# 3. API calls now work without separate authentication

from tools.applications_tools.tool_list_applications import list_applications
apps = list_applications.invoke({})  # Uses browser cookies
```

**Benefits:**
- Single authentication point
- Handles complex auth (SSO, 2FA, etc.)
- API calls faster than browser operations
- Use browser only for UI-only features

---

## Performance

### Timing Benchmarks (from POC)

| Operation | Time | Notes |
|-----------|------|-------|
| Browser startup | 2-3s | First command only |
| Login flow | 5-8s | Including waits |
| Page navigation | 2-4s | SPA content loading |
| Snapshot capture | 0.5-1s | Fast |
| Screenshot | 1-2s | Depends on page size |
| Session save | 0.5s | Fast |

**Total login + navigate:** 10-15 seconds

### Optimization Strategies

1. **Session Reuse**
   - Login once, reuse session
   - Saves 5-8 seconds per operation
   - State persists across agent restarts

2. **Parallel Operations**
   - Multiple tabs in same session
   - Independent operations in parallel
   - Not yet implemented (Phase 3)

3. **Hybrid Approach**
   - Use API for data operations (fast)
   - Use browser only for UI-only features (slow)
   - Best of both worlds

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Element not found @e5" | Refs invalidated | Re-snapshot after page change |
| "Timeout waiting for networkidle" | Slow page load | Increase timeout or use specific wait |
| "Login failed - still on login page" | Invalid credentials | Check username/password |
| "Session expired" | Inactivity timeout | Re-login with browser_login |
| "Screenshot shows old content" | Snapshot too early | Add wait_for_text parameter |

### Error Recovery Pattern

```python
def safe_browser_operation(operation_func, max_retries=3):
    """Execute browser operation with retry logic."""
    for attempt in range(max_retries):
        try:
            result = operation_func()
            
            if result["success"]:
                return result
            
            # Check if session expired
            if "Login" in result.get("error", ""):
                # Re-login and retry
                browser_login.invoke({...})
                continue
            
            # Check if refs invalidated
            if "Element not found" in result.get("error", ""):
                # Re-snapshot and retry
                browser_navigate.invoke({...})
                continue
            
            return result
            
        except Exception as e:
            if attempt == max_retries - 1:
                return {"success": False, "error": str(e)}
            time.sleep(2 ** attempt)
    
    return {"success": False, "error": "Max retries exceeded"}
```

---

## Screenshots

### Types

1. **Full Page**
   ```python
   browser_screenshot.invoke({
       "session_id": "user-123"
   })
   ```

2. **Element**
   ```python
   browser_screenshot.invoke({
       "element_ref": "@e5",
       "session_id": "user-123"
   })
   ```

3. **Annotated** (shows all refs)
   ```python
   browser_screenshot.invoke({
       "annotate": True,
       "session_id": "user-123"
   })
   ```

### Storage

**Location:** `.browser-states/screenshots/{session_id}/`

**Naming:**
- `login_before_submit.png`
- `login_success.png`
- `navigate_{timestamp}.png`

**Retention:** Manual cleanup (not auto-deleted)

---

## Undo/Redo

### Automatic Snapshots

Every tool call creates before/after snapshots:

**Location:** `.opencode/backups/{session_id}/`

**Format:**
```
1745678901_before_browser_login.json
1745678902_after_browser_login.json
1745678903_before_browser_navigate.json
1745678904_after_browser_navigate.json
```

### Rollback

```python
from agent_ng.undo_manager import UndoManager

undo = UndoManager()
undo.start_session("user-123")

# List snapshots
snapshots = undo.list_snapshots()
# ['1745678901_before_browser_login.json', ...]

# Rollback to specific point
undo.rollback_to("1745678901_before_browser_login.json")
```

### Snapshot Contents

```json
{
  "timestamp": 1745678901,
  "tool_name": "browser_login",
  "context": {
    "base_url": "https://platform.example.com/",
    "username": "user",
    "session_id": "user-123"
  },
  "type": "before"
}
```

---

## Advanced Usage

### Raw Command Execution

```python
from tools.browser_tools.tool_browser_execute import browser_execute

result = browser_execute.invoke({
    "commands": [
        "open https://platform.example.com/#Settings/Administration",
        "wait --load networkidle",
        "wait --text 'Administration'",
        "wait 2000",
        "snapshot -i",
        "screenshot administration.png",
        "get text body"
    ],
    "session_id": "user-123"
})

for cmd_result in result["results"]:
    print(f"{cmd_result['command']}: {cmd_result['success']}")
    print(f"Output: {cmd_result['output']}")
```

### Data Extraction

```python
from tools.browser_tools.tool_browser_extract import browser_extract

# Extract table data
result = browser_extract.invoke({
    "element_ref": "@e10",
    "extraction_type": "table",
    "session_id": "user-123"
})

if result["success"]:
    data = result["data"]  # List of dicts
    for row in data:
        print(row)
```

---

## Best Practices

### 1. Always Save State Before Operations

```python
# Bad
browser_interact.invoke({"action": "click", ...})

# Good
from agent_ng.undo_manager import UndoManager
undo = UndoManager()
undo.save_before_tool_call("browser_interact", {...})
browser_interact.invoke({"action": "click", ...})
undo.save_after_tool_call("browser_interact", result)
```

### 2. Use API When Possible

```python
# Bad - slow browser operation
browser_navigate.invoke({"url": "#solutions", ...})
browser_extract.invoke({"extraction_type": "table", ...})

# Good - fast API call
from tools.applications_tools.tool_list_applications import list_applications
apps = list_applications.invoke({})
```

### 3. Handle SPA Navigation Properly

```python
# Bad - no wait for content
browser_navigate.invoke({"url": "#Settings/Administration", ...})
browser_interact.invoke({"action": "click", ...})  # May fail

# Good - wait for content
browser_navigate.invoke({
    "url": "#Settings/Administration",
    "wait_for_text": "Administration",  # Wait for SPA content
    ...
})
browser_interact.invoke({"action": "click", ...})
```

### 4. Re-snapshot After Navigation

```python
# Bad - stale refs
snapshot1 = browser_navigate.invoke({...})
browser_interact.invoke({"element_ref": "@e5", ...})  # Click link
browser_interact.invoke({"element_ref": "@e8", ...})  # May fail - refs invalid

# Good - fresh refs
snapshot1 = browser_navigate.invoke({...})
browser_interact.invoke({"element_ref": "@e5", ...})  # Click link
snapshot2 = browser_navigate.invoke({...})  # Re-snapshot
browser_interact.invoke({"element_ref": "@e8", ...})  # Use new refs
```

### 5. Take Screenshots on Error

```python
try:
    result = browser_interact.invoke({...})
    if not result["success"]:
        # Take screenshot for debugging
        browser_screenshot.invoke({
            "session_id": session_id
        })
except Exception as e:
    browser_screenshot.invoke({
        "session_id": session_id
    })
    raise
```

---

## Troubleshooting

### Debug Checklist

1. **Check session state**
   ```python
   # Verify session exists
   from tools.browser_tools.browser_session_manager import BrowserSessionManager
   session = BrowserSessionManager.get_or_create_session("user-123")
   print(f"Active: {session.is_active}")
   ```

2. **Verify authentication**
   ```python
   # Check current URL
   result = browser_execute.invoke({
       "commands": ["get url"],
       "session_id": "user-123"
   })
   if "Login" in result["results"][0]["output"]:
       print("Session expired - need to re-login")
   ```

3. **Inspect page state**
   ```python
   # Take full snapshot
   result = browser_execute.invoke({
       "commands": ["snapshot"],
       "session_id": "user-123"
   })
   print(result["results"][0]["output"])
   ```

4. **Check screenshots**
   ```python
   # Visual verification
   browser_screenshot.invoke({
       "annotate": True,  # Shows all refs
       "session_id": "user-123"
   })
   # Check .browser-states/screenshots/user-123/
   ```

---

## Environment Variables

```bash
# Browser automation settings
BROWSER_SESSION_TIMEOUT=3600          # Auto-cleanup after 1 hour (seconds)
BROWSER_DEFAULT_WAIT=30000            # Default timeout (milliseconds)
BROWSER_HEADLESS=true                 # Headless mode (true/false)
BROWSER_STATE_DIR=.browser-states     # Session state storage
BROWSER_SCREENSHOT_DIR=screenshots    # Screenshot storage

# Undo/redo settings
UNDO_ENABLED=true                     # Enable undo/redo snapshots
UNDO_BACKUP_DIR=.opencode/backups     # Backup directory
UNDO_RETENTION_DAYS=7                 # Keep backups for N days
```

---

## Related Documentation

- Implementation Plan: `.opencode/plans/20260423_browser_automation_implementation_plan.md`
- POC Findings: `browser-automation-poc/FINDINGS.md`
- agent-browser Skill: `~/.agents/skills/agent-browser/SKILL.md`

---

## MCP Tool Interface Reference

The MCP server exposes tool names prefixed with the server alias. All four options below are configured in `~/.config/opencode/opencode.json`.

### Option 1: agent-browser MCP

Tool prefix: `agent-browser_browser_*`

```
agent-browser_browser_new_session      — create isolated session
agent-browser_browser_navigate         — open a URL
agent-browser_browser_snapshot         — get accessibility tree with @e1, @e2 refs
agent-browser_browser_click            — click by selector or ref
agent-browser_browser_fill             — fill input
agent-browser_browser_type             — type char-by-char (triggers JS handlers)
agent-browser_browser_press            — press keyboard key
agent-browser_browser_screenshot       — capture PNG
agent-browser_browser_evaluate         — run JS in page context
agent-browser_browser_wait_for_selector / wait_for_navigation
agent-browser_browser_get_text / get_html / get_url / get_title
agent-browser_browser_set_cookies / get_cookies
agent-browser_browser_close_session
```

Typical agent flow:
```
1. browser_new_session { viewport: {width:1920, height:1080} }
2. browser_navigate { url: "https://platform/..." }
3. browser_snapshot  → returns refs @e1, @e2, ...
4. browser_click { selector: "@e5" }
5. browser_fill { selector: "@e8", value: "text" }
6. browser_screenshot { path: "cmw-platform-workspace/step1.png" }
7. browser_close_session { sessionId: "..." }
```

Headed mode: `$env:AGENT_BROWSER_HEADED = "1"` then restart opencode.

### Option 2: playwright MCP

Tool prefix: `playwright_browser_*` — headed by default.

```
playwright_browser_navigate            — open URL
playwright_browser_snapshot            — rich accessibility snapshot with [ref=eN] (use to act)
playwright_browser_click               — click by ref (e.g., ref="e21")
playwright_browser_fill_form           — fill MULTIPLE fields at once
playwright_browser_type                — type into single field
playwright_browser_press_key           — keyboard input
playwright_browser_select_option       — dropdowns
playwright_browser_take_screenshot     — PNG/JPEG, optionally full-page
playwright_browser_evaluate            — run JS on page or element
playwright_browser_wait_for            — wait for text / time
playwright_browser_tabs                — list/new/close/select tabs
playwright_browser_network_requests    — inspect network
playwright_browser_console_messages    — collect console logs
playwright_browser_handle_dialog       — accept/dismiss native dialogs
playwright_browser_navigate_back / resize / close
```

**Two-step interaction (mandatory):**
1. `playwright_browser_snapshot` → returns YAML tree with `[ref=eN]` identifiers
2. Use that `ref` in `click` / `type` / `fill_form`

```yaml
# Example snapshot excerpt:
- textbox "E-mail or username" [ref=e14]
- textbox "Password" [ref=e20]
- button "Log in" [ref=e21]
```

Then:
```
playwright_browser_fill_form {
  fields: [
    { name: "Username", ref: "e14", type: "textbox", value: "bobragent" },
    { name: "Password", ref: "e20", type: "textbox", value: "***" }
  ]
}
playwright_browser_click { ref: "e21", element: "Log in button" }
```

⚠️ **Refs invalidate** after any navigation or DOM change. Re-snapshot always.

### Snapshot vs Screenshot

- Use **snapshot** to act on elements — it returns refs like `[ref=e14]` or `@e14`.
- Use **screenshot** only for visual verification, debugging, or reporting.
- **Screenshots do not return refs** and cannot be used directly for click/fill actions.

Rule of thumb:
1. `snapshot` → identify element refs
2. act (`click` / `fill` / `type`)
3. `snapshot` again after DOM or navigation changes
4. `take_screenshot` only when you need visual evidence

### Option 3: agent-browser CLI

Available as `agent-browser` in the shell. Upstream project and bundled skill: [vercel-labs/agent-browser](https://github.com/vercel-labs/agent-browser) (`skills/agent-browser`).

```powershell
# Navigate, snapshot, fill, screenshot
agent-browser open "https://host/"
agent-browser wait --load networkidle
agent-browser snapshot -i
agent-browser fill @e14 "bobragent"
agent-browser fill @e20 "$env:CMW_PASSWORD"
agent-browser click @e21
agent-browser screenshot "cmw-platform-workspace/logged_in.png"
agent-browser close

# Headed mode
agent-browser --headed open "https://host/"

# Named persistent session
agent-browser --session-name cmw-admin open "https://host/"
# ... login once ...
agent-browser --session-name cmw-admin close   # state auto-saved
# Later:
agent-browser --session-name cmw-admin open "https://host/#Settings/Administration"

# Get CDP URL to connect Playwright to the same Chrome instance
agent-browser get cdp-url
```

Auth vault:
```powershell
$env:CMW_PASSWORD | agent-browser auth save cmw --url "https://host/" --username bobragent --password-stdin
agent-browser auth login cmw
```

### Option 4: playwright-cli

Available as `playwright-cli` in the shell. CLI reference: [Playwright CLI](https://playwright.dev/docs/cli) · [microsoft/playwright](https://github.com/microsoft/playwright).

```powershell
# Basic flow
playwright-cli open "https://host/"
playwright-cli snapshot
playwright-cli fill e14 "bobragent"
playwright-cli fill e20 "$env:CMW_PASSWORD"
playwright-cli click e21
playwright-cli screenshot page.png
playwright-cli close

# Headed mode
playwright-cli --headed open "https://host/"

# Cross-browser
playwright-cli --browser firefox open "https://host/"
playwright-cli --browser webkit open "https://host/"

# Advanced features
playwright-cli pdf --filename report.pdf
playwright-cli network-requests
playwright-cli console-messages
playwright-cli eval "document.title"
playwright-cli eval "el => el.textContent" e5

# Tabs
playwright-cli tab-list
playwright-cli tab-new "https://example.com"
playwright-cli tab-select 2
playwright-cli tab-close

# Wait
playwright-cli wait --text "Administration"
playwright-cli wait --time 2000
```

**Key differences from agent-browser CLI:**
- **No session persistence** — each invocation is isolated
- **Cross-browser** — `--browser chrome|firefox|webkit`
- **More features** — PDF, network inspection, console logs, tracing
- **Ref format** — `e1` not `@e1`

### Credentials: Loading from .env

Never hardcode. Load via `python-dotenv` in Python:

```python
from dotenv import load_dotenv
import os
load_dotenv()
base_url = os.environ["CMW_BASE_URL"]
username = os.environ["CMW_LOGIN"]
password = os.environ["CMW_PASSWORD"]
```

PowerShell:
```powershell
Get-Content .env | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
  $name, $value = $_ -split '=', 2
  [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), 'Process')
}
agent-browser open $env:CMW_BASE_URL
```

### Standalone Python Utilities (NOT agent tools)

`tools/browser_tools.py` and `agent_ng/browser_session.py` are **intentionally NOT bound to the agent's tool list** — for external scripts only.

```python
# Only for scripts in docs/progress_reports/** — never inside agent flow
from tools.browser_tools import navigate_to_page, click_element, take_screenshot
```

## Browser Script Utilities

### When to use this script

Use `browser_session_util.py` only for session management tasks:

- list saved browser sessions
- save a named session manually
- load a named session manually
- clean up stale sessions

For actual browser automation:

- use MCP browser tools inside agent workflows
- use `agent-browser` or `playwright-cli` directly for standalone scripts

Do **not** treat `browser_session_util.py` as the main automation interface.

### browser_session_util.py

Location: `.agents/skills/cmw-platform/scripts/browser/browser_session_util.py`

Purpose: manage saved browser sessions (`list`, `save`, `load`, `cleanup`).

**Usage:**

```bash
python browser_session_util.py list
python browser_session_util.py save --session-name my-session
python browser_session_util.py load --session-name my-session
python browser_session_util.py cleanup --session-name my-session
```

### Example workflow

```bash
# 1. Login using agent-browser directly
agent-browser --session-name cmw-test open "https://platform.example.com/"
agent-browser --session-name cmw-test wait --load networkidle
agent-browser --session-name cmw-test find label "Email" fill "user@example.com"
agent-browser --session-name cmw-test find label "Password" fill "password"
agent-browser --session-name cmw-test find role button click --name "Log in"

# 2. Navigate to admin page
agent-browser --session-name cmw-test open "#Settings/Administration"
agent-browser --session-name cmw-test wait --text "Administration"
agent-browser --session-name cmw-test snapshot -i

# 3. Session state is auto-saved with --session-name
# 4. Inspect sessions with the utility
python browser_session_util.py list

# 5. Cleanup when done
python browser_session_util.py cleanup --session-name cmw-test
```

### Direct agent-browser usage pattern

```bash
# Login pattern
agent-browser --session-name <session> open "<base_url>"
agent-browser --session-name <session> wait --load networkidle
agent-browser --session-name <session> snapshot -i
agent-browser --session-name <session> fill @e8 "<username>"
agent-browser --session-name <session> fill @e9 "<password>"
agent-browser --session-name <session> click @e2

# Navigation pattern (CMW Platform URLs)
agent-browser --session-name <session> open "#Settings/Administration"
agent-browser --session-name <session> wait --load networkidle
agent-browser --session-name <session> wait --text "Administration"
agent-browser --session-name <session> wait 2000
agent-browser --session-name <session> snapshot -i
agent-browser --session-name <session> screenshot
```

### Integration notes

Browser tools may invoke `agent-browser` via subprocess. Relevant code paths:
- `tools/browser_tools/tool_browser_login.py`
- `tools/browser_tools/tool_browser_navigate.py`
- `tools/browser_tools/browser_session_manager.py`

### Requirements

- `agent-browser` CLI installed (`npm i -g agent-browser`)
- Chrome/Chromium installed
- `.env` file with CMW Platform credentials for manual testing

### Notes

- Sessions are isolated by `--session-name`
- State is auto-saved/restored with `--session-name`
- Use `browser_session_util.py` for debugging and manual cleanup
- For actual automation, prefer `agent-browser` CLI directly or the MCP tools above

---

## CMW Platform URL Patterns (SPA Hash Routing)

CMW Platform is a Single Page Application. Always use `wait_for` text or `networkidle` after navigation. Use `resolve_entity` to convert any URL or entity ID to system names for tool calls.

### Entity ID Prefix Registry

| Prefix | Entity Type | API Term | Resolution Method |
|--------|------------|----------|-------------------|
| `oa.{N}` | Record template | RecordTemplate | `TemplateService/List` (Type: Record) |
| `pa.{N}` | Process template | ProcessTemplate | `TemplateService/List` (Type: Process) |
| `ra.{N}` | Role template | RoleTemplate | `TemplateService/List` (Type: Role) |
| `os.{N}` | Org unit template | OrgStructureTemplate | `TemplateService/List` (Type: OrgStructure) |
| `sln.{N}` | Application | Solution | Match `solution` field in TemplateService results |
| `event.{N}` | Button | UserCommand | `UserCommand/List` |
| `form.{N}` | Form | Form | `Form/List` |
| `card.{N}` | Card view | Card | `Form/List` |
| `tb.{N}` | Toolbar | Toolbar | `Toolbar/List` |
| `lst.{N}` / `ds.{N}` | Table | Dataset | `Dataset/List` |
| `diagram.{N}` | Process diagram | Diagram | `Process/DiagramService/ResolveDiagram` |
| `role.{N}` | Role | Role | `TemplateService/List` (Type: Role) |
| `workspace.{N}` | Navigation section | Workspace | Metadata only |
| Plain `{N}` | Record | Record | `GET webapi/Record/{recordId}` |
| Plain `{N}` (task page) | Task | Task | `POST TeamNetwork/UserTaskService/Get` |

⚠️ **Universal Resolution:** All entity IDs are resolved via `OntologyService/GetAxioms` → returns `cmw.alias` / `cmw.container.alias` (system name), `cmw.object.name` (display name).

### resolve_entity Tool

```python
from tools.platform_entity_resolver import resolve_entity

# Full URL with template + button
result = resolve_entity.invoke({
    "url_or_id": "https://host/#RecordType/oa.193/Operation/event.15199"
})

# Raw entity ID
result = resolve_entity.invoke({"url_or_id": "oa.193"})

# Process template with diagram
result = resolve_entity.invoke({
    "url_or_id": "#ProcessTemplate/pa.77/Designer/Revision/diagram.315"
})
```

Output → use `system_name` + `application_system_name` with other tools:
```json
{
  "success": true,
  "resolved": [
    {
      "entity_type": "Template",
      "id": "oa.193",
      "system_name": "ServiceRequests",
      "application_system_name": "CustomerPortal",
      "name": "Service Requests"
    },
    {
      "entity_type": "Button",
      "id": "event.15199",
      "system_name": "approve_request",
      "application_system_name": "CustomerPortal",
      "name": "Approve Request",
      "kind": "Trigger scenario"
    }
  ]
}
```

### Settings Pages

| Page | URL |
|------|-----|
| Applications | `#Settings/Applications` |
| Navigation Sections | `#Settings/NavigationSections` |
| Templates | `#Settings/Templates` |
| Diagrams | `#Settings/Diagrams` |
| Functions | `#Settings/Functions` |
| Data Transfer Paths | `#Settings/DataTransferPaths` |
| Accounts | `#Settings/Accounts` |
| Groups | `#Settings/Groups` |
| System Roles | `#Settings/Roles` |
| Permissions Audit | `#Settings/PermissionsAudit` |
| Monitoring | `#Settings/Monitoring` |
| Event Logs | `#Settings/EventLogs` |
| Licensing | `#Settings/Licensing` |
| Backup | `#Settings/Backup` |
| Connections | `#Settings/Connections` |
| Performance | `#Settings/Performance` |
| Global Configuration | `#Settings/GlobalConfiguration` |
| Authentication Keys | `#Settings/AuthenticationKeys` |
| Org Structure | `#Settings/OrgStructure` |
| Version Management | `#Settings/VersionManagement` |
| Administration (hub) | `#Settings/Administration` |
| Global Security | `#Settings/globalSecurity` |

**Note:** PascalCase throughout, with one exception: `#Settings/globalSecurity`.

### Solution & Entity Views

| Page | URL |
|------|-----|
| Solutions | `#solutions` |
| Solution Admin | `#solutions/sln.{N}/Administration` |
| Solution Diagrams | `#solutions/sln.{N}/DiagramList/showAll` |
| Solution Roles | `#solutions/sln.{N}/roles` |
| Solution Workspace | `#solutions/sln.{N}/Workspaces/workspace.{M}` |
| Dashboard | `#desktop/` |
| Global Security Role | `#Settings/globalSecurity/role.{N}` |
| Global Security Privileges | `#Settings/globalSecurity/role.{N}/privileges` |

### Record Template Views (`oa.{N}`)

| Page | URL |
|------|-----|
| Administration | `#RecordType/oa.{N}/Administration` |
| Context | `#RecordType/oa.{N}/Context` |
| Forms | `#RecordType/oa.{N}/Forms` |
| Form | `#RecordType/oa.{N}/Forms/form.{M}` |
| Operations (buttons) | `#RecordType/oa.{N}/Operations` |
| Operation | `#RecordType/oa.{N}/Operation/event.{M}` |
| Toolbar | `#RecordType/oa.{N}/Toolbar/` |
| Toolbar Settings | `#RecordType/oa.{N}/Toolbar/Settings/tb.{M}` |
| Card view | `#RecordType/oa.{N}/Card/` |
| Card Settings | `#RecordType/oa.{N}/Card/Settings/card.{M}` |
| Tables | `#RecordType/oa.{N}/Lists/` |
| Table | `#RecordType/oa.{N}/Lists/lst.{M}` |
| CSV Export | `#RecordType/oa.{N}/csv` |
| Security | `#RecordType/oa.{N}/Security` |
| Document Templates | `#RecordType/oa.{N}/DocumentsTemplates` |

### Other Template Views

| Template | Page | URL |
|----------|------|-----|
| Role (`ra.{N}`) | Administration | `#RecordType/ra.{N}/Administration` |
| Role | Toolbar | `#RecordType/ra.{N}/Toolbar/` |
| Org Unit (`os.{N}`) | Administration | `#RecordType/os.{N}/Administration` |
| Process (`pa.{N}`) | Designer Diagram | `#ProcessTemplate/pa.{N}/Designer/Revision/diagram.{M}` |
| Process | Tables | `#ProcessTemplate/pa.{N}/Lists/` |
| Process | Toolbar | `#ProcessTemplate/pa.{N}/Toolbar/tb.{M}` |
| Process | Operation | `#ProcessTemplate/pa.{N}/Operation/event.{M}` |

### Data & Form Views

| Page | URL |
|------|-----|
| Data view | `#data/oa.{N}/lst.{M}/...` |
| Form view | `#form/oa.{N}/form.{M}/{recordId}` |
| App data list | `#app/{App}/list/{Tpl}` |
| App record view | `#app/{App}/view/{Tpl}/{recordId}` |
| Task | `#task/{taskId}` |
| Entity resolver | `#Resolver/{id}` |

---

## Common Issues (MCP / CLI)

| Issue | Cause | Fix |
|-------|-------|-----|
| "Element not found" after click | Refs invalidated by navigation | Re-run `snapshot` before next action |
| Timeout on SPA load | Content loads after networkidle | Add `wait_for { text: "Expected text" }` |
| "Session expired" | Cookie TTL passed | Re-login; consider using saved state file |
| Screenshot shows old content | Action ran before render | `wait_for` text or `networkidle` |
| Modal dialog blocks clicks | Popup/notification overlay | `press Escape` or `handle_dialog` |
| Login form not detected | Localized page | Use `ref=` from snapshot, not text/placeholder |
| `agent-browser` "spawn ENOENT" | CLI not on PATH / Node not found | Use Windows PowerShell, or `npm i -g agent-browser` |

---

## Platform Testing Lessons

**Login flow:**
- Login URL: `https://host/Home/Login/?returnUrl=/`
- Fields: "E-mail address or username" (textbox), "Password" (textbox), "Log in" (button)
- After login: redirects to `#desktop/` (dashboard)
- Page title changes from "Comindware Platform" to app-specific title

**Navigation via hash:**
- `playwright-cli eval "() => { window.location.hash = '#path/here'; }"` works for SPA navigation
- After hash change: always `snapshot` to get fresh refs
- Page title reflects current context (e.g., "Управление объектами недвижимости > Шаблоны > ...")

**Admin panel structure:**
- Solution admin: `#solutions/sln.{N}/Administration`
- Template admin: `#RecordType/oa.{N}/Administration`
- Template operations (buttons): `#RecordType/oa.{N}/Operations`

**UI localization:**
- Russian UI shows Russian labels — use `ref=` from snapshot for reliable targeting
- Navigation sidebar has collapsible sections with submenu items

**Snapshot best practices:**
- After any navigation (hash change, click, form submit), always re-snapshot
- Refs invalidate immediately after DOM changes
- Use `playwright_browser_snapshot` / `playwright-cli snapshot` to get YAML tree with `[ref=eN]`

---

*Last Updated: 2026-05-02*
