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

*Last Updated: 2026-04-22*
