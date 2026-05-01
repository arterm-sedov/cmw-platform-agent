# Browser Automation Utility Scripts

This directory is intentionally minimal.

The canonical documentation for browser automation now lives in:

- [`../../references/browser_automation.md`](../../references/browser_automation.md)

That reference contains:

- all 4 browser automation options (MCP + CLI)
- snapshot vs screenshot rules
- CMW Platform URL patterns
- troubleshooting
- platform testing lessons
- browser utility script guidance

## Remaining Script

### `browser_session_util.py`

Use this script only for session management tasks:

- list saved sessions
- save or load a named session manually
- clean up stale sessions

Do **not** use it as the main browser automation interface.

```bash
python browser_session_util.py list
python browser_session_util.py save --session-name my-session
python browser_session_util.py load --session-name my-session
python browser_session_util.py cleanup --session-name my-session
```

For real browser automation flows, use the MCP tools or `agent-browser` / `playwright-cli` directly.
