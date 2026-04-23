# Browser Automation Utility Scripts

Utility for managing browser sessions when working with CMW Platform browser automation.

## Why Only One Script?

The agent-browser CLI provides all necessary commands for login, navigation, and interaction. These utility scripts were redundant wrappers. We keep only the session management utility for debugging and manual cleanup.

## Script

### browser_session_util.py
Manage browser sessions (save/load/cleanup).

**Usage:**
```bash
# List all sessions
python browser_session_util.py list

# Save session state
python browser_session_util.py save --session-name my-session

# Load session state
python browser_session_util.py load --session-name my-session

# Cleanup session
python browser_session_util.py cleanup --session-name my-session
```

## Example Workflow

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

# 3. Session state is auto-saved with --session-name flag

# 4. List sessions using utility
python browser_session_util.py list

# 5. Cleanup when done
python browser_session_util.py cleanup --session-name cmw-test
```

## Direct agent-browser Usage

For actual implementation, use agent-browser CLI directly:

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
agent-browser --session-name <session> wait 2000  # SPA buffer
agent-browser --session-name <session> snapshot -i
agent-browser --session-name <session> screenshot

# Session state auto-saved/restored with --session-name flag
```

## Integration with Tools

The browser tools will use agent-browser CLI directly via subprocess:
- `tools/browser_tools/tool_browser_login.py` - Calls agent-browser commands
- `tools/browser_tools/tool_browser_navigate.py` - Calls agent-browser commands
- `tools/browser_tools/browser_session_manager.py` - Manages sessions

The tools will integrate with:
- LangChain `@tool` decorator
- Pydantic input validation
- Session manager for per-user isolation
- HTTP session for cookie injection

## CMW Platform URL Patterns

**Discovered patterns (hash-based routing):**
- `#Settings/Administration` - Admin settings
- `#Settings/globalSecurity` - Security settings
- `#Settings/Groups` - Groups management
- `#solutions` - Solutions/applications
- `#desktop/` - Main dashboard

## Requirements

- agent-browser CLI installed (`npm i -g agent-browser`)
- Chrome/Chromium installed
- `.env` file with CMW Platform credentials (for manual testing)

## Notes

- Sessions are isolated by `--session-name` flag
- State auto-saved/restored with `--session-name`
- Use `browser_session_util.py` for debugging and manual cleanup
- For actual automation, use agent-browser CLI directly or via LangChain tools
