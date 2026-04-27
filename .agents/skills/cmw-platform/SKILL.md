---
name: cmw-platform
description: Use when working with Comindware Platform — connecting to platform, listing applications, exploring templates, managing records, querying data, creating or editing attributes, or any task requiring autonomous platform interaction. Triggers on platform operations, CMW queries, tenant management, rental lot operations, debt tracking, or record CRUD. Also triggers when user mentions working directly with API credentials, using manual HTTP approaches, or trying to bypass the tool layer. ALSO triggers when user mentions browser automation, UI-only features, visual verification, or accessing platform features not available via API.
---

# CMW Platform Skill

Enables autonomous interaction with Comindware Platform using tools from the agent's `tools/` directory.

**Browser Use:** Browser automation for UI-only features and visual verification.

**Four browser options available out-of-the-box:**
- **agent-browser MCP** — tool-based, token-efficient, Chrome-only. Best for agent workflows.
- **playwright MCP** — tool-based, cross-browser, rich snapshots with element refs. Best for complex UI and debugging (headed by default).
- **agent-browser CLI** — shell command, persistent sessions, Chrome-only. Best for standalone PowerShell scripts with session reuse.
- **playwright-cli** — shell command, cross-browser, advanced features (PDF, network, tracing). Best for cross-browser testing scripts.

All four are configured in `~/.config/opencode/opencode.json`. Full guidance + decision matrix in Section 2 below.

---

## 1. Core Concepts

### Platform Terminology

**Critical:** Use human-readable terms, never expose API terms to LLMs:

| API Term | Human-Friendly | Notes |
|----------|---------------|-------|
| alias | system name | Entity identifier |
| instance | record | Data entry |
| user command | button | UI action |
| container | template | Application/template |
| property | attribute | Field/column |
| dataset | table | List view. Also called список or list by users |
| list | table | Table view in template |
| solution | application | Business application |
| scheme | process diagram | Visual workflow |
| trigger | scenario | Automation logic |
| workspace | navigation section | Navigation section |
| card | card view | Card view for a table (not a form) |
| task | task | User task |

### Workflow

Always follow: `Intent → Plan → Validate → Execute → Result`

### Tool Usage Discipline

- Check for duplicate calls before invoking
- Transform results to human-readable format (never raw JSON)
- Validate required context before execution

### Browser Automation vs API

**When to use Browser Automation:**
- UI-only features (visual configuration, drag-and-drop)
- Visual verification (screenshots, layout validation)
- Features not available via API
- Complex multi-step UI workflows

**When to use API:**
- Data operations (CRUD on records)
- Bulk operations
- Programmatic access
- Performance-critical tasks

### Tool Invocation Pattern

```python
from tools.applications_tools.tool_list_applications import list_applications

result = list_applications.invoke({})
if not result["success"]:
    print(f"Error: {result['error']}")
    return
print(result["data"])
```

### Response Structure

```python
{
    "success": bool,      # True if operation succeeded
    "status_code": int,   # HTTP status code
    "data": list|dict,     # Response payload
    "error": str|dict      # Error details if success=False
}
```

### Save Before Edit

**ALWAYS save schemas and data BEFORE editing or deleting:**

```bash
# Save to cmw-platform-workspace/ immediately after fetching
cp Step1_Schema_GET.json cmw-platform-workspace/Step1_Schema_BEFORE.json
# ... make changes ...
cp Step2_Schema_GET_AFTER.json cmw-platform-workspace/Step2_Schema_AFTER.json
```

This is NOT optional. Violating this rule has caused data loss before.

→ See also: [references/tool_inventory.md](references/tool_inventory.md), [references/api_endpoints.md](references/api_endpoints.md)

---

## 2. Browser Automation

Browser automation enables UI-only features, visual verification, and workflows not exposed via API. The agent supports **four interchangeable browser options**: two MCPs and two CLIs. Pick the one that matches the task.

### Four Browser Options

| Option | Access | Best For | Headed Mode |
|--------|--------|----------|-------------|
| **agent-browser MCP** | Tool calls | AI agent workflows, token-efficient actions, Chrome-only | `AGENT_BROWSER_HEADED=1` env |
| **playwright MCP** | Tool calls | Cross-browser (Chrome/Firefox/WebKit), form filling, rich snapshots, debugging | `--headed` flag (default in MCP) |
| **agent-browser CLI** | Shell/PowerShell | Scripted batch automation, persistent sessions, local dev | `--headed` flag |
| **playwright-cli** | Shell/PowerShell | Cross-browser CLI, advanced features (PDF, network, tracing), no session persistence | `--headed` flag |

All four are configured in `~/.config/opencode/opencode.json` and available out-of-the-box.

### Decision Guide

**Use agent-browser MCP when:**
- Running inside an agent conversation and token budget matters (~5.7x more efficient than Playwright)
- Chrome-only is fine
- Workflow is action-driven (click → navigate → click)

**Use playwright MCP when:**
- Need Firefox/WebKit/mobile emulation
- Need rich accessibility snapshots with element refs (`e1`, `e2`, ...)
- Need advanced features: `fill_form`, multi-field forms, network interception, tracing
- Debugging UI issues visually (headed mode is default in the MCP)
- Working with complex SPAs where auto-waiting helps

**Use agent-browser CLI when:**
- Writing standalone scripts (PowerShell/Bash) outside the agent
- Need persistent named sessions across multiple shell invocations
- Batch scraping/automation pipelines
- Running in CI or scheduled jobs
- Chrome-only is acceptable

**Use playwright-cli when:**
- Writing standalone scripts that need cross-browser support
- Need advanced CLI features: PDF export, network inspection, console logs, tracing
- Don't need session persistence (each invocation is isolated)
- Want full Playwright feature set from the command line

→ See also: [browser-switch skill](file:///C:/Users/ased/.agents/skills/browser-switch/skills/browser-switch/SKILL.md), [agent-browser skill](file:///C:/Users/ased/.agents/skills/agent-browser/SKILL.md), [playwright-cli skill](file:///C:/Users/ased/.config/opencode/skills/playwright/SKILL.md)

### When to Use Browser vs API

| Operation | Use API | Use Browser |
|-----------|---------|-------------|
| List records | ✅ Fast, structured | ❌ Slow, parsing needed |
| Create/edit attributes | ✅ Direct, reliable | ❌ Complex UI navigation |
| Visual workflow designer | ❌ No API | ✅ UI-only feature |
| Admin panel configuration | ⚠️ Limited API | ✅ Full access |
| Verify UI changes | ❌ Can't see UI | ✅ Screenshots |
| Extract UI table data | ⚠️ If no API | ✅ Fallback option |

### Option 1: agent-browser MCP (preferred for agent workflows)

Configured as MCP `agent-browser` in `opencode.json`. Tool names exposed to the agent are prefixed `agent-browser_browser_*`:

```
agent-browser_browser_new_session    — create isolated session
agent-browser_browser_navigate       — open a URL
agent-browser_browser_snapshot       — get accessibility tree with @e1, @e2 refs
agent-browser_browser_click          — click by selector or ref
agent-browser_browser_fill           — fill input
agent-browser_browser_type           — type char-by-char (triggers JS handlers)
agent-browser_browser_press          — press keyboard key
agent-browser_browser_screenshot     — capture PNG
agent-browser_browser_evaluate       — run JS in page context
agent-browser_browser_wait_for_selector / wait_for_navigation
agent-browser_browser_get_text / get_html / get_url / get_title
agent-browser_browser_set_cookies / get_cookies
agent-browser_browser_close_session
```

Typical flow (agent calls these as tools, not Python):

```
1. browser_new_session { viewport: {width:1920, height:1080} }
2. browser_navigate { url: "https://platform/..." }
3. browser_snapshot  → returns refs @e1, @e2, ...
4. browser_click { selector: "@e5" }
5. browser_fill { selector: "@e8", value: "text" }
6. browser_screenshot { path: "cmw-platform-workspace/step1.png" }
7. browser_close_session { sessionId: "..." }
```

**Headed mode (user can watch / intervene):**
```powershell
$env:AGENT_BROWSER_HEADED = "1"   # then restart opencode
```

### Option 2: playwright MCP (best for visual work & cross-browser)

Configured as MCP `playwright` in `opencode.json`. Tool names are prefixed `playwright_browser_*`. This MCP runs **headed by default**, making it ideal when the user wants to watch the automation and intervene.

Key tools (use these names in tool calls):

```
playwright_browser_navigate          — open URL
playwright_browser_snapshot          — rich accessibility snapshot with refs (use THIS, not screenshot, to act)
playwright_browser_click             — click by ref (e.g., ref="e21")
playwright_browser_fill_form         — fill MULTIPLE fields at once (powerful!)
playwright_browser_type              — type into a single field
playwright_browser_press_key         — keyboard input
playwright_browser_select_option     — dropdowns
playwright_browser_take_screenshot   — PNG/JPEG, optionally full-page
playwright_browser_evaluate          — run JS on page or element
playwright_browser_wait_for          — wait for text / time
playwright_browser_tabs              — list/new/close/select tabs
playwright_browser_network_requests  — inspect network
playwright_browser_console_messages  — collect console logs
playwright_browser_handle_dialog     — accept/dismiss native dialogs
playwright_browser_navigate_back / resize / close
```

**Two-step interaction pattern (mandatory):**
1. `playwright_browser_snapshot` → returns YAML tree with `[ref=eN]` identifiers
2. Use that `ref` in `click`/`type`/`fill_form`

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

**⚠️ Ref lifecycle:** Refs invalidate after any navigation or DOM change. Always re-snapshot after clicking a link, submitting a form, or opening a modal.

**Screenshot vs Snapshot:** Use `snapshot` to act on elements (gives refs). Use `take_screenshot` only for visual verification or reporting — it does NOT return refs.

### Option 3: agent-browser CLI (for standalone scripts)

The CLI is available in the shell as `agent-browser`. Full command reference is in the [agent-browser skill](file:///C:/Users/ased/.agents/skills/agent-browser/SKILL.md).

**PowerShell examples** (this project uses PowerShell, not bash):

```powershell
# Basic navigation + snapshot + action
agent-browser open "https://bububu.bau.cbap.ru/"
agent-browser wait --load networkidle
agent-browser snapshot -i
agent-browser fill @e14 "bobragent"
agent-browser fill @e20 "$env:CMW_PASSWORD"
agent-browser click @e21
agent-browser screenshot "cmw-platform-workspace/logged_in.png"
agent-browser close

# Headed mode (user can watch)
agent-browser --headed open "https://bububu.bau.cbap.ru/"

# Named persistent session (survives across shell invocations)
agent-browser --session-name cmw-admin open "https://bububu.bau.cbap.ru/"
# ... login once ...
agent-browser --session-name cmw-admin close   # state auto-saved
# Later:
agent-browser --session-name cmw-admin open "https://bububu.bau.cbap.ru/#Settings/Administration"

# Chain commands (& not && in PowerShell pipelines; use ; for simple chaining)
agent-browser open "https://example.com"; agent-browser wait --load networkidle; agent-browser screenshot out.png

# Get cdp url to connect playwright to the same Chrome
agent-browser get cdp-url
```

**Auth vault (credentials stay encrypted, never in shell history):**
```powershell
$env:CMW_PASSWORD | agent-browser auth save cmw --url "https://bububu.bau.cbap.ru/" --username bobragent --password-stdin
agent-browser auth login cmw
```

### Credentials: Always from .env

**Never hardcode credentials in scripts.** Load them from `.env` the same way `agent_ng` does:

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
# python-dotenv reads .env automatically in Python scripts.
# For PowerShell-only scripts, load .env lines into env vars:
Get-Content .env | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
  $name, $value = $_ -split '=', 2
  [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), 'Process')
}
agent-browser open $env:CMW_BASE_URL
```

### Option 4: playwright-cli (for cross-browser scripts)

The CLI is available in the shell as `playwright-cli`. Full command reference is in the [playwright-cli skill](file:///C:/Users/ased/.config/opencode/skills/playwright/SKILL.md).

**PowerShell examples** (this project uses PowerShell, not bash):

```powershell
# Basic navigation + snapshot + action
playwright-cli open "https://bububu.bau.cbap.ru/"
playwright-cli snapshot
playwright-cli click e14
playwright-cli fill e14 "bobragent"
playwright-cli fill e20 "$env:CMW_PASSWORD"
playwright-cli click e21
playwright-cli screenshot page.png
playwright-cli close

# Headed mode (user can watch)
playwright-cli --headed open "https://bububu.bau.cbap.ru/"

# Cross-browser testing
playwright-cli --browser firefox open "https://bububu.bau.cbap.ru/"
playwright-cli --browser webkit open "https://bububu.bau.cbap.ru/"

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

# Wait patterns
playwright-cli wait --text "Administration"
playwright-cli wait --time 2000
```

**Key differences from agent-browser CLI:**
- **No session persistence** — each invocation is isolated (no `--session-name`)
- **Cross-browser** — `--browser chrome|firefox|webkit`
- **More features** — PDF export, network inspection, console logs, tracing
- **Ref format** — uses `e1`, `e2` (no `@` prefix)

### Standalone Python browser utilities (NOT agent tools)

The repo also ships standalone Playwright wrappers in `tools/browser_tools.py` and `agent_ng/browser_session.py`. **These are intentionally NOT bound to the agent's tool list** — they are for external scripts only. The agent should use the MCPs above.

```python
# Only for scripts in docs/progress_reports/**, never inside agent flow
from tools.browser_tools import navigate_to_page, click_element, take_screenshot
```

### CMW Platform URL Patterns (SPA hash routing)

CMW Platform is a Single Page Application. Always use `wait_for` text or `networkidle` after navigation.

**Entity ID Prefix Registry:**

| Prefix | Entity Type (Human-Friendly) | API Term | Resolution Method |
|--------|---------------------------|----------|-------------------|
| `oa.{N}` | Record template | RecordTemplate | `TemplateService/List` (Type: Record) |
| `pa.{N}` | Process template | ProcessTemplate | `TemplateService/List` (Type: Process) |
| `ra.{N}` | Role template | RoleTemplate | `TemplateService/List` (Type: Role) |
| `os.{N}` | Organizational unit template | OrgStructureTemplate | `TemplateService/List` (Type: OrgStructure) |
| `sln.{N}` | Application | Solution | Match `solution` field in TemplateService results |
| `event.{N}` | Button | UserCommand | List candidates via `UserCommand/List` |
| `form.{N}` | Form | Form | List candidates via `Form/List` |
| `card.{N}` | Card view | Card | List candidates via `Form/List` |
| `tb.{N}` | Toolbar | Toolbar | List candidates via `Toolbar/List` |
| `lst.{N}` / `ds.{N}` | Table | Dataset | List candidates via `Dataset/List` |
| `diagram.{N}` | Process diagram | Diagram | `Process/DiagramService/ResolveDiagram` |
| `role.{N}` | Role | Role | `TemplateService/List` (Type: Role) |
| `workspace.{N}` | Navigation section | Workspace | Metadata only (no known API) |
| Plain `{N}` | Record | Record | `GET webapi/Record/{recordId}` |
| Plain `{N}` (task page) | Task | Task | `POST TeamNetwork/UserTaskService/Get` |

**⚠️ Critical API Finding:** TemplateService/List is the ONLY endpoint that maps internal IDs (`oa.*`, `pa.*`, etc.) to system names + application context. For buttons/forms/toolbars/datasets, internal IDs are NOT exposed by any API — use `resolve_entity` tool to list candidates and match by name/context.

**URL Resolution Tool:** Use `resolve_entity` to parse any CMW Platform URL and get API-ready entity objects:

```python
from tools.platform_entity_resolver import resolve_entity

# Full URL with template + button
result = resolve_entity.invoke({
    "url_or_id": "https://host/#RecordType/oa.193/Operation/event.15199",
    "fetch_full": True,
})

# Raw entity ID
result = resolve_entity.invoke({"url_or_id": "oa.193"})

# Process template with diagram
result = resolve_entity.invoke({
    "url_or_id": "#ProcessTemplate/pa.77/Designer/Revision/diagram.315",
})
```

**Output structure:**
```json
{
    "success": True,
    "url_parsed": {
        "original": "...",
        "page_type": "RecordType",
        "entities_found": [
            {"type": "Template", "id": "oa.193"},
            {"type": "Button", "id": "event.15199"}
        ]
    },
    "resolved": [
        {
            "entity_type": "Template",
            "internal_id": "oa.193",
            "system_name": "MaintenancePlans",
            "application_system_name": "FacilityManagement",
            "application_id": "sln.23",
            "api_endpoint": "webapi/RecordTemplate/FacilityManagement/MaintenancePlans",
            "full_data": { ... }
        },
        {
            "entity_type": "Button",
            "internal_id": "event.15199",
            "candidates": [
                {"system_name": "schedule_maintenance", "name": "...", "kind": "UserEvent", ...}
            ],
            "note": "Internal entity IDs not exposed by API. Match by name/context."
        }
    ]
}
```

**Architecture — Settings Pages:**
| Page | URL |
|------|-----|
| Applications | `#Settings/Applications` |
| Navigation Sections | `#Settings/NavigationSections` |
| Templates | `#Settings/Templates` |
| Diagrams | `#Settings/Diagrams` |
| Functions | `#Settings/Functions` |
| Data Transfer Paths | `#Settings/DataTransferPaths` |

**Account Administration:**
| Page | URL |
|------|-----|
| Accounts | `#Settings/Accounts` |
| Groups | `#Settings/Groups` |
| System Roles | `#Settings/Roles` |
| Permissions Audit | `#Settings/PermissionsAudit` |
| Substitutions | `#Settings/Substitutions` |
| Registration and Login | `#Settings/RegistrationAndLogin` |

**Infrastructure:**
| Page | URL |
|------|-----|
| Monitoring | `#Settings/Monitoring` |
| Event Logs | `#Settings/EventLogs` |
| Licensing | `#Settings/Licensing` |
| Backup | `#Settings/Backup` |
| Connections | `#Settings/Connections` |
| Performance | `#Settings/Performance` |
| Logging Configuration | `#Settings/LoggingConfiguration` |
| Global Configuration | `#Settings/GlobalConfiguration` |
| Adapters | `#Settings/Adapters` |
| Authentication Keys | `#Settings/AuthenticationKeys` |

**Corporate Architecture:**
| Page | URL |
|------|-----|
| Org Structure | `#Settings/OrgStructure` |
| Processes | `#Settings/Processes` |
| Version Management | `#Settings/VersionManagement` |

**Solutions & Entity Views:**
| Page | URL | Entities |
|------|-----|----------|
| Administration (hub) | `#Settings/Administration` | None |
| Global Security | `#Settings/globalSecurity` | None |
| Global Security Role | `#Settings/globalSecurity/role.{N}` | Role |
| Global Security Privileges | `#Settings/globalSecurity/role.{N}/privileges` | Role |
| Solutions | `#solutions` | None |
| Solution Admin | `#solutions/sln.{N}/Administration` | Solution |
| Solution Diagrams | `#solutions/sln.{N}/DiagramList/showAll` | Solution |
| Solution Roles | `#solutions/sln.{N}/roles` | Solution |
| Solution Role | `#solutions/sln.{N}/roles/role.{M}` | Solution + Role |
| Solution Workspaces | `#solutions/Workspaces` | None |
| Solution Workspace | `#solutions/sln.{N}/Workspaces/workspace.{M}` | Solution + Workspace |
| Dashboard | `#desktop/` | None |

**Record Template Views (`oa.{N}`):**
| Page | URL | Entities |
|------|-----|----------|
| Administration | `#RecordType/oa.{N}/Administration` | Record template |
| Context | `#RecordType/oa.{N}/Context` | Record template |
| Forms | `#RecordType/oa.{N}/Forms` | Record template |
| Form | `#RecordType/oa.{N}/Forms/form.{M}` | Record template + Form |
| Operations | `#RecordType/oa.{N}/Operations` | Record template |
| Operation | `#RecordType/oa.{N}/Operation/event.{M}` | Record template + Button |
| Toolbar | `#RecordType/oa.{N}/Toolbar/` | Record template |
| Toolbar Settings | `#RecordType/oa.{N}/Toolbar/Settings/tb.{M}` | Record template + Toolbar |
| Card view | `#RecordType/oa.{N}/Card/` | Record template |
| Card Settings | `#RecordType/oa.{N}/Card/Settings/card.{M}` | Record template + Card view |
| Tables | `#RecordType/oa.{N}/Lists/` | Record template |
| Table | `#RecordType/oa.{N}/Lists/lst.{M}` | Record template + Table |
| CSV Export | `#RecordType/oa.{N}/csv` | Record template |
| Security | `#RecordType/oa.{N}/Security` | Record template |
| Document Templates | `#RecordType/oa.{N}/DocumentsTemplates` | Record template |

**Role Template Views (`ra.{N}`):**
| Page | URL | Entities |
|------|-----|----------|
| Administration | `#RecordType/ra.{N}/Administration` | Role template |
| Context | `#RecordType/ra.{N}/Context` | Role template |
| Toolbar | `#RecordType/ra.{N}/Toolbar/` | Role template |

**Organizational Unit Template Views (`os.{N}`):**
| Page | URL | Entities |
|------|-----|----------|
| Administration | `#RecordType/os.{N}/Administration` | Organizational unit template |
| Context | `#RecordType/os.{N}/Context` | Organizational unit template |
| Toolbar | `#RecordType/os.{N}/Toolbar/` | Organizational unit template |

**Process Template Views (`pa.{N}`):**
| Page | URL | Entities |
|------|-----|----------|
| Designer Diagram | `#ProcessTemplate/pa.{N}/Designer/Revision/diagram.{M}` | Process template + Process diagram |
| Tables | `#ProcessTemplate/pa.{N}/Lists/` | Process template |
| Table | `#ProcessTemplate/pa.{N}/Lists/lst.{M}` | Process template + Table |
| Toolbar | `#ProcessTemplate/pa.{N}/Toolbar/tb.{M}` | Process template + Toolbar |
| Operation | `#ProcessTemplate/pa.{N}/Operation/event.{M}` | Process template + Button |

**Data & Form Views:**
| Page | URL | Entities |
|------|-----|----------|
| Data view | `#data/oa.{N}/lst.{M}/...` | Record template + Table (+ query params) |
| Form view | `#form/oa.{N}/form.{M}/{recordId}` | Record template + Form + Record |
| App data list | `#app/{App}/list/{Tpl}` | Application + Record template |
| App record view | `#app/{App}/view/{Tpl}/{recordId}` | Application + Record template + Record |
| Entity resolver | `#Resolver/{id}` | Single entity |

**Task Views:**
| Page | URL | Entities |
|------|-----|----------|
| Task | `#task/{taskId}` | Task (plain numeric ID) |
| My Tasks | `#myTasks/...` | Tasks list page (no specific entity IDs) |

**Naming convention:** PascalCase (e.g. `#Settings/Applications`), with one camelCase exception: `#Settings/globalSecurity`.

### Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| "Element not found" after click | Refs invalidated by navigation | Re-run `snapshot` before next action |
| Timeout on SPA load | Content loads after networkidle | Add `wait_for { text: "Expected text" }` |
| "Session expired" | Cookie TTL passed | Re-login; consider using saved state file |
| Screenshot shows old content | Action ran before render completed | `wait_for` text or use `networkidle` |
| Modal dialog blocks clicks | Push-notification / permissions popup | `press Escape` or `handle_dialog` |
| Login form not detected | Login page is localized | Use `ref=` from snapshot, not text/placeholder |
| `agent-browser` "spawn ENOENT" | CLI not on PATH / Node not found in WSL | Use Windows PowerShell, or `npm i -g agent-browser` |

### Browser Automation Lessons from Platform Testing

**Login flow:**
- Login page URL: `https://host/Home/Login/?returnUrl=/`
- Fields: "E-mail address or username" (textbox), "Password" (textbox), "Log in" (button)
- After login: redirects to `#desktop/` (dashboard)
- Page title changes from "Comindware Platform" to app-specific title

**Navigation via hash:**
- `playwright-cli eval "() => { window.location.hash = '#path/here'; }"` works for SPA navigation
- After hash change: always `snapshot` to get fresh refs
- Page title updates to reflect current context (e.g., "Управление объектами недвижимости > Шаблоны > Планы техобслуживания > Кнопки")

**Admin panel structure:**
- Solution admin: `#solutions/sln.{N}/Administration`
- Template admin: `#RecordType/oa.{N}/Administration`
- Template operations (buttons): `#RecordType/oa.{N}/Operations`
- Templates list: `#solutions/sln.{N}/templates/showall/...`

**UI localization:**
- Russian UI shows Russian labels (e.g., "Приложение", "Шаблоны", "Кнопки")
- Use `ref=` from snapshot for reliable element targeting, not text labels
- Navigation sidebar has collapsible sections with submenu items

**Snapshot best practices:**
- After any navigation (hash change, click), always re-snapshot
- Refs are invalidated immediately after DOM changes
- Use `playwright-cli snapshot` to get the YAML tree with `[ref=eN]` identifiers

### Best Practices

1. **Snapshot → Act → Re-snapshot** after any DOM-changing action
2. **Save before edit** — always screenshot or export current UI state before making destructive changes
3. **Session isolation** — use unique `session_name` / `sessionId` per workflow to avoid cross-contamination
4. **Headed for debugging, headless for automation** — flip the mode based on the task
5. **Credentials from `.env`** — never commit or hardcode (see Credentials section above)
6. **PowerShell, not bash** — this project runs on Windows; avoid `&&`, `cat`, `grep`; use `;`, `Get-Content`, `Select-String`

→ See also: [references/browser_automation.md](references/browser_automation.md)

---

## 3. Exploration

Explore application structure systematically.

### Workflow: Discover Application

```python
from tools.applications_tools.tool_list_applications import list_applications
from tools.applications_tools.tool_list_templates import list_templates
from tools.templates_tools.tool_list_attributes import list_attributes

# Step 1: List applications
apps = list_applications.invoke({})
target_app = next(
    (a for a in apps["data"] if "target_application" in a["Name"]),
    None
)

# Step 2: List templates in application
templates = list_templates.invoke({
    "application_system_name": target_app["Application system name"]
})

# Step 3: Get schema for each template
for tmpl in templates["data"][:5]:
    attrs = list_attributes.invoke({
        "application_system_name": target_app["Application system name"],
        "template_system_name": tmpl["Template system name"]
    })
```

### Utility Script

For batch exploration, use:
```bash
python .agents/skills/cmw-platform/scripts/explore_templates.py \
    --app <app_name> --templates Template1,Template2
```

→ See also: [references/workflow_sequences.md](references/workflow_sequences.md)

---

## 4. Data Operations

### Query Records with Filters

```python
from tools.templates_tools.tool_list_records import list_template_records

result = list_template_records.invoke({
    "application_system_name": "your_app",
    "template_system_name": "your_template",
    "filters": {"SomeAttribute": {"$gt": 30}},
    "limit": 100
})
if result["success"]:
    for record in result["data"]:
        print(record["id"], record.get("Name", ""))
```

### Create a Record

```python
from tools.templates_tools.tool_create_edit_record import create_edit_record

result = create_edit_record.invoke({
    "operation": "create",
    "application_system_name": "your_app",
    "template_system_name": "your_template",
    "values": {
        "AttributeName": "value",
        "AnotherAttribute": 123.45
    }
})
```

### Pagination

Hard limit: 100 records per request. Paginate using `offset`:

```python
def fetch_all(app_name: str, template: str, page_size: int = 100):
    all_records = []
    offset = 0
    while True:
        result = list_template_records.invoke({
            "application_system_name": app_name,
            "template_system_name": template,
            "limit": page_size,
            "offset": offset
        })
        if not result["success"] or not result["data"]:
            break
        all_records.extend(result["data"])
        if len(result["data"]) < page_size:
            break
        offset += page_size
    return all_records
```

### Utility Scripts

- **Paginated query with filters:** `query_with_filter.py`
- **Statistical analysis:** `analyze_stats.py`
- **Batch attribute editing:** `batch_edit_attributes.py`

→ See also: [references/workflow_sequences.md](references/workflow_sequences.md)

---

## 3.5. Import/Export Applications

### Export Application

Export an entire application (templates, attributes, workflows) to CTF format:

```python
from tools.transfer_tools.tool_export_application import export_application

result = export_application.invoke({
    "application_system_name": "my_app",
    "save_to_file": True  # Saves .ctf file to /tmp/cmw-transfer/
})
if result["success"]:
    print(f"CTF saved to: {result['ctf_file_path']}")
    print(f"CTF data: {len(result['ctf_data'])} chars")
```

### Import Application

Import an application from CTF format. The import is a 2-step process (upload + execute):

```python
from tools.transfer_tools.tool_import_application import import_application

# Option 1: From exported CTF file path
result = import_application.invoke({
    "application_system_name": "new_app_name",
    "ctf_file_path": "/tmp/cmw-transfer/my_app_abc123.ctf"
})

# Option 2: From Base64 CTF data directly
result = import_application.invoke({
    "application_system_name": "new_app_name",
    "ctf_data": "SQBDA...<base64>..."
})
```

**API Endpoints:**
- Export: `GET /webapi/Transfer/{solutionAlias}`
- Upload: `POST /webapi/Transfer/Upload`
- Import: `POST /webapi/Transfer/{solutionAlias}/{fileId}/true/ApplyNew`

---

## 4. UI Components

Datasets, Toolbars, and Buttons are **separate API entities** with different endpoints:

| Entity | Tool to Get | Tool to Edit |
|--------|-------------|--------------|
| Dataset | `get_dataset` | `edit_or_create_dataset` |
| Toolbar | `get_toolbar` | `edit_or_create_toolbar` |
| Button | `get_button` | `edit_or_create_button` |

### List and Edit Toolbars

```python
from tools.templates_tools.tools_toolbar import list_toolbars, get_toolbar, edit_or_create_toolbar

toolbars = list_toolbars.invoke({
    "application_system_name": "<app>",
    "template_system_name": "<template>"
})
for tb in toolbars["data"]:
    print(f"{tb['globalAlias']['alias']}: {tb['name']}")

toolbar = get_toolbar.invoke({
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "toolbar_system_name": "<toolbar>"
})
for item in toolbar.get("items", []):
    print(f"  - {item['name']} ({item['action']['alias']})")

edit_or_create_toolbar.invoke({
    "operation": "edit",
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "toolbar_system_name": "<toolbar>",
    "name": "<New Name>",
    "items": [
        {"button_system_name": "create", "display_name": "<Label>", "item_order": 0},
    ]
})
```

**⚠️ Dataset-Specific Toolbars:** If a dataset shares a toolbar with other datasets, editing that toolbar affects ALL linked datasets. Create a NEW toolbar for dataset-specific buttons.

### List and Edit Buttons

```python
from tools.templates_tools.tools_button import list_buttons, edit_or_create_button

buttons = list_buttons.invoke({
    "application_system_name": "<app>",
    "template_system_name": "<template>"
})

edit_or_create_button.invoke({
    "operation": "edit",
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "button_system_name": "<button>",
    "name": "<Name>",
    "description": "<Description>"
})
```

**⚠️ Toolbar Item Names Override Button Names:** Toolbar items have their own `name` field that overrides the button's display name.

### Create-Kind Buttons

For buttons with `kind='Create'`, you **MUST** specify `create_form` (and optionally `create_template`):

```python
edit_or_create_button.invoke({
    "operation": "edit",
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "button_system_name": "<button>",
    "name": "Create New Record",
    "kind": "Create",
    "context": "Record",  # or "List" depending on where button appears
    "create_form": "defaultForm",  # REQUIRED for Create buttons
    "create_template": "<target_template>",  # Optional, defaults to current template
})
```

**Key requirements:**
- `create_form` is **mandatory** for `kind='Create'` buttons
- `context` must match where the button appears (`"Record"` for record forms, `"List"` for list toolbars)
- `create_template` defaults to the current template if omitted
- The API builds `relatedAction.formGlobalAlias` from these parameters

**⚠️ UI Cache Issue:** The platform UI may display stale context values (e.g., `#List` instead of `Record`) even after successful API updates. Always verify via `get_button` API call, not the UI. Hard refresh (Ctrl+F5) or clearing browser cache may be required to see correct values in the UI.

### Button Kinds (Action Types)

CMW Platform supports 29 button action types. The tool accepts LLM-friendly terms and maps them to API values.

**Common Button Kinds:**

| LLM Term | API Term | Description | Russian (RU) |
|----------|----------|-------------|--------------|
| Trigger scenario | UserEvent | Execute custom scenario (default) | Вызвать событие «Нажата кнопка» |
| Create | Create | Create new record (requires `create_form`) | Создать |
| Edit | Edit | Edit existing record | Редактировать |
| Delete | Delete | Delete record | Удалить |
| Archive | Archive | Archive record | Архивировать |
| Unarchive | Unarchive | Restore archived record | Разархивировать |
| Script | Script | Execute C# script | С# скрипт |

**All 29 Valid Button Kinds:**

| # | API Term | English | Russian (RU) |
|---|----------|---------|--------------|
| 1 | Undefined | No specific action | - |
| 2 | Create | Create new record (requires `create_form`) | Создать |
| 3 | Edit | Edit existing record | Редактировать |
| 4 | Delete | Delete record | Удалить |
| 5 | Archive | Archive record | Архивировать |
| 6 | Unarchive | Restore archived record | Разархивировать |
| 7 | ExportObject | Export single object | Экспорт записи |
| 8 | ExportList | Export list of objects | Экспорт таблицы |
| 9 | CreateRelated | Create related record | Создать связанную запись |
| 10 | CreateToken | Create token | Создать токен |
| 11 | RetryTokens | Retry tokens | Перезапустить токены |
| 12 | Migrate | Migrate data | Мигрировать |
| 13 | StartCase | Start case | - |
| 14 | StartLinkedCase | Start linked case | - |
| 15 | StartProcess | Start process | Запустить процесс |
| 16 | StartLinkedProcess | Start linked process | Запустить процесс по связанному шаблону |
| 17 | CompleteTask | Complete task | Завершить задачу |
| 18 | ReassignTask | Reassign task | Переназначить |
| 19 | Defer | Defer action | Отложить выполнение |
| 20 | Accept | Accept action | Принять |
| 21 | Uncomplete | Mark as incomplete | Открыть заново |
| 22 | Follow | Follow record | Привязать к шаблону |
| 23 | Unfollow | Unfollow record | Отвязать от шаблону |
| 24 | Exclude | Exclude from list | - |
| 25 | Include | Include in list | - |
| 26 | Script | Execute C# script | С# скрипт |
| 27 | Cancel | Cancel action | Остановить процесс |
| 28 | EditDiagram | Edit diagram | - |
| 29 | UserEvent | Execute custom scenario (use "Trigger scenario") | Вызвать событие «Нажата кнопка» |

**Usage Examples:**

```python
# Default: Trigger custom scenario
edit_or_create_button.invoke({
    "operation": "create",
    "application_system_name": "FacilityManagement",
    "template_system_name": "MaintenancePlans",
    "button_system_name": "run_maintenance_check",
    "name": "Run Maintenance Check",
    "kind": "Trigger scenario",  # Maps to UserEvent in API
})

# Explicit API term also works
edit_or_create_button.invoke({
    "operation": "create",
    "kind": "UserEvent",  # Direct API term
})

# Other common kinds
edit_or_create_button.invoke({
    "operation": "create",
    "kind": "Archive",  # Archive button
})

# Case-insensitive variants work
edit_or_create_button.invoke({
    "operation": "create",
    "kind": "trigger_scenario",  # Snake_case → UserEvent
})
```

**Validation:**
- The validator is case-insensitive and handles variants like `trigger_scenario`, `TRIGGER SCENARIO`
- Invalid kinds (e.g., "Test") are rejected with clear error messages
- All 29 API enum values are validated

**Edit Behavior:**
- `kind` parameter is optional in edit operations
- If omitted, the existing kind is preserved
- If provided, the kind is updated (even if changing to UserEvent)
- Always verify changes via `get_button` API, not just the UI (UI may cache stale values)

### List and Edit Datasets

```python
from tools.templates_tools.tools_dataset import list_datasets, get_dataset, edit_or_create_dataset

datasets = list_datasets.invoke({
    "application_system_name": "<app>",
    "template_system_name": "<template>"
})

edit_or_create_dataset.invoke({
    "operation": "edit",
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "dataset_system_name": "<dataset>",
    "name": "<New Name>",
    "toolbar_system_name": "<toolbar>",
    "columns": {
        "<column>": {"Name": "<New Label>"},
        "<columnToHide>": {"isHidden": True},
    }
})
```

**Toolbar-Dataset Linking:**
- Datasets link to toolbars via the `toolbar_system_name` parameter
- Toolbars link back to datasets via `IsDefaultForLists` flag (set on the toolbar)
- When a toolbar has `IsDefaultForLists: true`, it becomes the default for all datasets in that template
- To make a toolbar dataset-specific, set `IsDefaultForLists: false` and link it explicitly via the dataset's `toolbar_system_name`

### Edit Form Widgets

```python
from tools.templates_tools.tools_form import get_form, edit_or_create_form

form = get_form.invoke({
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "form_system_name": "<form>"
})

edit_or_create_form.invoke({
    "operation": "edit",
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "form_system_name": "<form>",
    "widgets": [
        {"system_name": "<widget>", "label": "<New Label>"},
    ]
})
```

→ See also: [references/tool_inventory.md](references/tool_inventory.md)

---

## 5. Browser Automation (NEW)

**Use browser automation when:**
- API endpoints don't cover the feature
- Need visual verification of changes
- Working with workflow designers or visual editors
- Testing actual user workflows
- Debugging UI issues

### When to Use Browser vs API

| Operation | Use API | Use Browser |
|-----------|---------|-------------|
| List records | ✅ Fast, structured | ❌ Slow, parsing needed |
| Create/edit attributes | ✅ Direct, reliable | ❌ Complex UI navigation |
| Visual workflow designer | ❌ No API | ✅ UI-only feature |
| Admin panel configuration | ⚠️ Limited API | ✅ Full access |
| Verify UI changes | ❌ Can't see UI | ✅ Screenshots |
| Extract UI table data | ⚠️ If no API | ✅ Fallback option |

### Browser Login

```python
from tools.browser_tools.tool_browser_login import browser_login

result = browser_login.invoke({
    "base_url": "https://platform.example.com/",
    "username": "user",
    "password": "pass",
    "session_id": "user-session-123"
})

if result["success"]:
    print(f"Logged in, extracted {result['cookies_extracted']} cookies")
    print(f"Screenshot: {result['screenshot']}")
```

**What it does:**
1. Opens platform in browser
2. Fills login form
3. Extracts session cookies
4. Injects cookies into HTTP session (enables API calls)
5. Saves browser state for reuse

### Navigate to Admin Pages

```python
from tools.browser_tools.tool_browser_navigate import browser_navigate

result = browser_navigate.invoke({
    "url": "#Settings/Administration",  # Hash fragment for SPA
    "wait_for_text": "Administration",  # Wait for content
    "session_id": "user-session-123"
})

if result["success"]:
    print(f"Current URL: {result['current_url']}")
    print(f"Snapshot:\n{result['snapshot']}")
    print(f"Screenshot: {result['screenshot']}")
```

**SPA Navigation Notes:**
- CMW Platform is a Single Page Application
- Content loads dynamically after URL changes
- Always use `wait_for_text` for reliable content detection
- Additional 2-second buffer added automatically

### Interact with UI Elements

```python
from tools.browser_tools.tool_browser_interact import browser_interact

# Click a button
result = browser_interact.invoke({
    "action": "click",
    "element_ref": "@e5",  # From snapshot
    "session_id": "user-session-123"
})

# Fill a form field
result = browser_interact.invoke({
    "action": "fill",
    "element_ref": "@e8",
    "value": "New Value",
    "session_id": "user-session-123"
})

# Select dropdown option
result = browser_interact.invoke({
    "action": "select",
    "element_ref": "@e12",
    "value": "option-value",
    "session_id": "user-session-123"
})
```

**Element Refs:**
- Get refs from `browser_navigate` snapshot
- Refs are like `@e1`, `@e2`, `@e3`
- Refs invalidate after page changes (re-snapshot)

### Extract Data from UI

```python
from tools.browser_tools.tool_browser_extract import browser_extract

result = browser_extract.invoke({
    "element_ref": "@e10",  # Table or grid element
    "extraction_type": "table",
    "session_id": "user-session-123"
})

if result["success"]:
    data = result["data"]  # Structured JSON/CSV
    print(f"Extracted {len(data)} rows")
```

### Visual Verification

```python
from tools.browser_tools.tool_browser_screenshot import browser_screenshot

# Full page screenshot
result = browser_screenshot.invoke({
    "session_id": "user-session-123"
})

# Element screenshot
result = browser_screenshot.invoke({
    "element_ref": "@e5",
    "session_id": "user-session-123"
})

# Annotated screenshot (shows all refs)
result = browser_screenshot.invoke({
    "annotate": True,
    "session_id": "user-session-123"
})

print(f"Screenshot saved: {result['screenshot_path']}")
```

### Advanced: Execute Raw Commands

```python
from tools.browser_tools.tool_browser_execute import browser_execute

result = browser_execute.invoke({
    "commands": [
        "open https://platform.example.com/#Settings/Administration",
        "wait --load networkidle",
        "wait --text 'Administration'",
        "snapshot -i",
        "screenshot administration.png"
    ],
    "session_id": "user-session-123"
})

for cmd_result in result["results"]:
    print(f"{cmd_result['command']}: {cmd_result['success']}")
```

### Browser Session Management

**Automatic:**
- Sessions isolated per user (via `session_id`)
- State saved automatically after operations
- Sessions restored on next use
- Auto-cleanup after timeout (default: 1 hour)

**Manual:**
```python
from tools.browser_tools.browser_session_manager import BrowserSessionManager

# Cleanup specific session
BrowserSessionManager.cleanup_session("user-session-123")

# Cleanup all sessions
BrowserSessionManager.cleanup_all()
```

### Known URL Patterns

CMW Platform uses hash-based routing (SPA). All admin pages verified by browser exploration:

**Appearance:**
| Page | URL Pattern |
|------|-------------|
| Themes | `#Settings/Theme` |
| Login/Registration Design | `#Settings/LoginDesign` |

**Architecture:**
| Page | URL Pattern |
|------|-------------|
| Applications | `#Settings/Applications` |
| Navigation Sections | `#Settings/NavigationSections` |
| Templates | `#Settings/Templates` |
| Diagrams | `#Settings/Diagrams` |
| Functions | `#Settings/Functions` |
| Data Transfer Paths | `#Settings/DataTransferPaths` |

**Account Administration:**
| Page | URL Pattern |
|------|-------------|
| Accounts | `#Settings/Accounts` |
| Groups | `#Settings/Groups` |
| System Roles | `#Settings/Roles` |
| Permissions Audit | `#Settings/PermissionsAudit` |
| Substitutions | `#Settings/Substitutions` |
| Registration and Login | `#Settings/RegistrationAndLogin` |

**Infrastructure:**
| Page | URL Pattern |
|------|-------------|
| Monitoring | `#Settings/Monitoring` |
| Event Logs | `#Settings/EventLogs` |
| Licensing | `#Settings/Licensing` |
| Backup | `#Settings/Backup` |
| Connections | `#Settings/Connections` |
| Performance | `#Settings/Performance` |
| Logging Configuration | `#Settings/LoggingConfiguration` |
| Global Configuration | `#Settings/GlobalConfiguration` |
| Adapters | `#Settings/Adapters` |
| Authentication Keys | `#Settings/AuthenticationKeys` |

**Corporate Architecture:**
| Page | URL Pattern |
|------|-------------|
| Org Structure | `#Settings/OrgStructure` |
| Processes | `#Settings/Processes` |
| Version Management | `#Settings/VersionManagement` |

**Other:**
| Page | URL Pattern |
|------|-------------|
| Administration (hub) | `#Settings/Administration` |
| Global Security | `#Settings/globalSecurity` |
| Solutions | `#solutions` |
| Dashboard | `#desktop/` |

**Naming convention:** PascalCase (`#Settings/Applications`), one camelCase exception (`#Settings/globalSecurity`).

### Browser Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Element not found" | Refs invalidated | Re-snapshot after page changes |
| "Timeout waiting for content" | SPA loading slow | Increase wait time or use `wait --text` |
| "Session expired" | Inactivity timeout | Re-login with `browser_login` |
| "Screenshot shows old content" | Snapshot taken too early | Add `wait_for_text` parameter |

→ See also: [references/browser_automation.md](references/browser_automation.md)

---

## 6. Localization (System Names)

Localization workflow for renaming system names (aliases) in Comindware Platform applications. This is a **multi-step interactive process** requiring user confirmation at each phase.

### Phase Overview

| Phase | Action | Show Table | User Confirm |
|-------|--------|------------|--------------|
| 1 | Export CTF (if not provided) | — | — |
| 2 | Collect aliases from JSON, tag by object type | **Yes** | — |
| 3 | Verify IDs via get_ontology_objects | **Yes** | — |
| 4 | Analyze Expression fields, suggest suffixes | **Yes** | **Yes** |
| 5 | Rename via update_object_property | **Yes** | **Yes** |
| 6 | Ask user to restart platform | — | **Yes** |
| 7 | Re-export CTF | — | — |
| 8 | Replace dangerous aliases in JSON | **Yes** | — |
| 9 | Import modified CTF | — | **Yes** |

**Table columns:** `type`, `systemName`, `jsonPath`, `id`, `renamedSystemName`

### Step-by-Step Workflow

#### Phase 1: Export CTF

```python
from tools.transfer_tools.tool_export_application import export_application

result = export_application.invoke({
    "application_system_name": "my_app",
    "save_to_file": True,
})
if result["success"]:
    ctf_path = result["ctf_file_path"]
```

#### Phase 2: Collect Aliases from JSON

Traverse the exported JSON folder. For each file, extract `"Alias"` values and tag with the **object type inferred from the parent folder name**:

```
RecordTemplates/ → RecordTemplate
ProcessTemplates/ → ProcessTemplate
Datasets/ → Dataset
Forms/ → Form
Toolbars/ → Toolbar
UserCommands/ → UserCommand
Attributes/ → Attribute
Workspaces/ → Workspace
Pages/ → Page
Roles/ → RoleTemplate
Accounts/ → AccountTemplate
```

Build a lookup dict: `{alias: {"type": obj_type, "json_path": path}}`.

**Show table to user:**

| type | systemName | jsonPath | id | renamedSystemName |
|------|------------|----------|----|-------------------|
| Form | MyForm | Forms/MyForm.json#$.GlobalAlias.Alias | — | — |
| RecordTemplate | MyRecord | RecordTemplates/MyRecord.json#$.GlobalAlias.Alias | — | — |
| ... | ... | ... | ... | ... |

#### Phase 3: Verify Aliases via get_ontology_objects

```python
from tools.applications_tools.tool_get_ontology_objects import get_ontology_objects

result = get_ontology_objects.invoke({
    "application_system_name": "my_app",
    "types": ["RecordTemplate", "ProcessTemplate", "Dataset", "Form", "Toolbar", "UserCommand", "Attribute", "Workspace", "Page"],
    "parameter": "alias",
    "min_count": 1,
    "max_count": 10000,
})
```

Compare results with Phase 2 aliases. **Only include aliases found in platform** — silently exclude aliases not matched. Build verified dict: `{alias: object_id}`.

**Show table to user (id column now filled):**

| type | systemName | jsonPath | id | renamedSystemName |
|------|------------|----------|----|-------------------|
| Form | MyForm | Forms/MyForm.json#$.GlobalAlias.Alias | form.338 | — |
| RecordTemplate | MyRecord | RecordTemplates/MyRecord.json#$.GlobalAlias.Alias | container.42 | — |
| ... | ... | ... | ... | ... |

#### Phase 4: Analyze Expression Fields

Scan all JSON files for `"Expression"` fields containing alias references **outside of `"Alias"` context**.

```python
import re

EXPRESSION_KEYS = {"Expression", "Code", "ValueExpression", "ValidationScript"}

def check_dangerous_aliases(json_folder: str, aliases: set[str]) -> dict[str, bool]:
    dangerous = {a: False for a in aliases}
    for json_file in Path(json_folder).rglob("*.json"):
        content = open(json_file).read()
        for alias in aliases:
            if dangerous[alias]:
                continue
            for key in EXPRESSION_KEYS:
                pattern = rf'"{key}"\s*:\s*"[^"]*{re.escape(alias)}[^"]*"'
                if re.search(pattern, content):
                    dangerous[alias] = True
                    break
    return dangerous
```

Assign suffixes and **show complete table to user:**

| type | systemName | jsonPath | id | renamedSystemName |
|------|------------|----------|----|-------------------|
| Form | MyForm | Forms/MyForm.json#$.GlobalAlias.Alias | form.338 | MyForm_calc |
| RecordTemplate | MyRecord | RecordTemplates/MyRecord.json#$.GlobalAlias.Alias | container.42 | MyRecord_sv |
| ... | ... | ... | ... | ... |

- `type` — object type from folder name
- `systemName` — original alias from JSON
- `jsonPath` — path to the JSON file with JSONPath to the alias field
- `id` — platform object ID (from get_ontology_objects)
- `renamedSystemName` — new alias with suffix (`_calc` for dangerous, `_sv` for safe)

| Category | Suffix | Meaning |
|----------|--------|---------|
| **Dangerous** | `_calc` (default) | Mentioned in Expression — rename to `{alias}{suffix}` everywhere (e.g. `MyAlias_calc`) |
| **Safe** | `_sv` (default) | Only in `GlobalAlias.Alias` context — rename to `{alias}{suffix}` (e.g. `MyAlias_sv`) |

**Show table to user and ask to confirm** the rename plan before proceeding.

#### Phase 5: Apply Renames via update_object_property

```python
from tools.applications_tools.tool_update_object_property import update_object_property

for alias, new_alias in rename_map.items():
    object_id = verified_aliases[alias]
    object_type = TYPE_MAPPING[alias_type]

    result = update_object_property.invoke({
        "object_id": object_id,
        "object_type": object_type,
        "new_value": new_alias,
    })
```

**Important:** Alias values must be without spaces — use CamelCase or underscores (e.g. `myAlias_calc`, `anotherOne_sv`).

**Show table to user** with rename results:

| type | systemName | jsonPath | id | renamedSystemName | status |
|------|------------|----------|----|-------------------|--------|
| Form | MyForm | Forms/MyForm.json#$.GlobalAlias.Alias | form.338 | MyForm_calc | ✅ renamed |
| RecordTemplate | MyRecord | RecordTemplates/MyRecord.json#$.GlobalAlias.Alias | container.42 | MyRecord_sv | ✅ renamed |
| ... | ... | ... | ... | ... | ... |

**Show table and ask user to confirm** before proceeding to restart.

#### Phase 6: Request Platform Restart

Inform the user:
> "System names have been renamed. Please restart the Comindware Platform service now. Once restarted, confirm to proceed with re-export."

**Wait for user confirmation** that restart is complete.

#### Phase 7: Re-Export CTF

```python
result = export_application.invoke({
    "application_system_name": "my_app",
    "save_to_file": True,
})
```

#### Phase 8: Replace Dangerous Aliases in JSON

In the newly exported JSON files, replace all occurrences of **dangerous** aliases (in both `Alias` and `Expression` fields) with their new suffixed names:

```python
for alias, new_alias in dangerous_renames.items():
    safe_pattern = re.escape(alias)
    for json_file in Path(json_folder).rglob("*.json"):
        content = open(json_file).read()
        content = re.sub(r'"Alias"\s*:\s*"' + safe_pattern + r'"', '"Alias": "' + new_alias + '"', content)
        for key in EXPRESSION_KEYS:
            content = re.sub(rf'"{key}"\s*:\s*"[^"]*{safe_pattern}[^"]*"',
                            lambda m: m.group(0).replace(alias, new_alias), content)
        open(json_file, "w").write(content)
```

**Important:** Only dangerous aliases are replaced. Safe aliases remain untouched in JSON (their rename is only in platform).

**Show table to user** with updated jsonPath (if changed):

| type | systemName | jsonPath | id | renamedSystemName | jsonUpdated |
|------|------------|----------|----|-------------------|-------------|
| Form | MyForm | Forms/MyForm.json#$.GlobalAlias.Alias | form.338 | MyForm_calc | ✅ |
| RecordTemplate | MyRecord | RecordTemplates/MyRecord.json#$.GlobalAlias.Alias | container.42 | MyRecord_sv | — |
| ... | ... | ... | ... | ... | ... |

#### Phase 9: Import Modified CTF (Update Existing)

```python
from tools.transfer_tools.tool_import_application import import_application

result = import_application.invoke({
    "application_system_name": "my_app",
    "ctf_file_path": "/path/to/modified_ctf.ctf",
    "update_existing": True,
})
```

**Use `update_existing: True`** to update the existing application by system name, not create a new one.

Save the final table to files:

```python
import json
from pathlib import Path

# Save as JSON
table_data = [
    {"type": "Form", "systemName": "MyForm", "jsonPath": "Forms/MyForm.json#$.GlobalAlias.Alias", "id": "form.338", "renamedSystemName": "MyForm_calc"},
    {"type": "RecordTemplate", "systemName": "MyRecord", "jsonPath": "RecordTemplates/MyRecord.json#$.GlobalAlias.Alias", "id": "container.42", "renamedSystemName": "MyRecord_sv"},
]

output_dir = Path("/path/to/output")
(output_dir / "localization_table.json").write_text(json.dumps(table_data, indent=2))

# Save as Markdown
md_lines = ["| type | systemName | jsonPath | id | renamedSystemName |",
           "|------|------------|----------|----|-------------------|"]
for row in table_data:
    md_lines.append(f"| {row['type']} | {row['systemName']} | {row['jsonPath']} | {row['id']} | {row['renamedSystemName']} |")
(output_dir / "localization_table.md").write_text("\n".join(md_lines))
```

**Show final table and ask user to confirm** before importing.

### Type-Folder Mapping Reference

```python
TYPE_FOLDER_MAPPING = {
    "RecordTemplate": "RecordTemplates",
    "ProcessTemplate": "ProcessTemplates",
    "RoleTemplate": "Roles",
    "AccountTemplate": "Accounts",
    "OrgStructureTemplate": "OrgStructure",
    "MessageTemplate": "MessageTemplates",
    "Workspace": "Workspaces",
    "Page": "Pages",
    "Attribute": "Attributes",
    "Dataset": "Datasets",
    "Toolbar": "Toolbars",
    "Form": "Forms",
    "UserCommand": "UserCommands",
    "Card": "Cards",
    "Cart": "Carts",
    "Trigger": "Triggers",
    "Role": "Roles",
    "WidgetConfig": "WidgetConfigs",
}

TYPE_PREDICATE_MAPPING = {
    "RecordTemplate": "cmw.container.alias",
    "ProcessTemplate": "cmw.container.alias",
    "RoleTemplate": "cmw.container.alias",
    "AccountTemplate": "cmw.container.alias",
    "OrgStructureTemplate": "cmw.container.alias",
    "MessageTemplate": "cmw.message.type.alias",
    "Workspace": "cmw.alias",
    "Page": "cmw.desktopPage.alias",
    "Attribute": "cmw.object.alias",
    "Dataset": "cmw.alias",
    "Toolbar": "cmw.alias",
    "Form": "cmw.alias",
    "UserCommand": "cmw.alias",
    "Card": "cmw.alias",
    "Cart": "cmw.cart.alias",
    "Trigger": "cmw.trigger.alias",
    "Role": "cmw.role.alias",
    "WidgetConfig": "cmw.form.alias",
}

# For Role objects with aliasProperty:
# - Role type supports both "cmw.role.alias" (direct) and "cmw.role.aliasProperty" (indirect)
# - When cmw.role.aliasProperty is present, it contains an attribute ID (e.g., "op.2")
# - Use GetAxiomsByPredicate endpoint to resolve: {"id": "role.2", "predicate": "op.2"}
# - Response returns the actual alias value: ["Администратор"]
# - ID prefix for Role objects: "role."
# - ID prefix for WidgetConfig objects: "fw."
```

### tool_localize - Localization Tool

The `tool_localize` (function name: `localize_aliases`) provides automated localization workflow for collecting and tracking aliases and display names.

**Capabilities:**
- Collect aliases (system names) from CTF JSON
- Collect display names (Name property) from CTF JSON
- Verify aliases via API (GetWithMultipleValues)
- Track both in localization workflow
- Generate reports with aliases and/or display names
- Apply alias renames via API (OntologyService/AddStatement)

**Important Notes:**
- Both alias and displayName collection are optional (can be enabled/disabled independently)
- Aliases are applied via API (step 6 in workflow)
- DisplayNames are applied via CTF import (step 10 in workflow)
- DisplayNames are NOT verified via API (CTF-based workflow)

**Parameters:**
- `collect_aliases`: bool (default: True) - Collect alias data (system names)
- `collect_display_names`: bool (default: True) - Collect displayName data (Name property)
- `dry_run`: bool (default: True) - Preview changes without applying
- `dangerous_suffix`: str (default: "_calc") - Suffix for dangerous aliases
- `safe_suffix`: str (default: "_sv") - Suffix for safe aliases

**Usage Examples:**

```python
from tools.localization_tools.tool_localize import localize_aliases

# Collect both aliases and display names (default)
result = localize_aliases.invoke({
    "application_system_name": "MyApp",
    "json_folder": "/path/to/ctf",
    "collect_aliases": True,
    "collect_display_names": True,
    "dry_run": True
})

# Collect only aliases
result = localize_aliases.invoke({
    "application_system_name": "MyApp",
    "json_folder": "/path/to/ctf",
    "collect_aliases": True,
    "collect_display_names": False
})

# Collect only display names
result = localize_aliases.invoke({
    "application_system_name": "MyApp",
    "json_folder": "/path/to/ctf",
    "collect_aliases": False,
    "collect_display_names": True
})
```

**Return Structure:**
```python
{
    "success": bool,
    "aliases_collected": int,           # Count of aliases collected
    "display_names_collected": int,     # Count of display names collected
    "aliases_verified": int,            # Count of aliases verified via API
    "aliases_missing": list,            # Aliases not found in platform
    "dangerous_aliases": list,          # Aliases used in expressions
    "safe_aliases": list,               # Aliases only in alias fields
    "collect_aliases": bool,            # What was collected
    "collect_display_names": bool,      # What was collected
    "errors": list
}
```

→ See also: [references/localization.md](references/localization.md)

---

## 7. Troubleshooting

### Error Handling

| Status | Meaning | Action |
|--------|---------|--------|
| 401 | Bad credentials | Check .env configuration |
| 408 | Query timeout | Reduce `limit` parameter (max 100) |
| 500 | Server error | Retry with exponential backoff |

### Retry Pattern

```python
import time

def retry_with_backoff(func, payload, max_retries=3, delay=1):
    for attempt in range(max_retries):
        result = func.invoke(payload)
        if result["success"]:
            return result
        if result.get("status_code") in (500, 503, 408):
            time.sleep(delay * (2 ** attempt))
            continue
        return result
    return {"success": False, "error": "Max retries exceeded"}
```

### Safe Attribute Translation

The edit_or_create tools have **smart partial update support**:

| Operation | Behavior | Mechanism |
|-----------|----------|----------|
| **Create** | Requires ALL type-specific fields | Model validator raises error if missing |
| **Edit - partial** | Missing fields fetched from API and merged | `tool_utils.py` patch fills gaps |
| **Edit - explicit** | Provided fields override existing values | User intent respected |

### Diagnostic Script

```bash
python .agents/skills/cmw-platform/scripts/diagnose_connection.py
```

Exit code 0 = pass, 1 = fail.

→ See also: [references/errors.md](references/errors.md)

---

## Reference Index

| Document | Purpose |
|---------|---------|
| [references/tool_inventory.md](references/tool_inventory.md) | Complete tool catalog with signatures |
| [references/api_endpoints.md](references/api_endpoints.md) | HTTP endpoint reference |
| [references/errors.md](references/errors.md) | Error handling playbook |
| [references/workflow_sequences.md](references/workflow_sequences.md) | Reusable code patterns |
| [references/localization.md](references/localization.md) | Russian→English translation guide |
| [references/browser_automation.md](references/browser_automation.md) | Browser automation guide |
| [browser-switch skill](file:///C:/Users/ased/.agents/skills/browser-switch/skills/browser-switch/SKILL.md) | Decide between agent-browser and Playwright |
| [agent-browser skill](file:///C:/Users/ased/.agents/skills/agent-browser/SKILL.md) | Full agent-browser CLI reference |
| [playwright-cli skill](file:///C:/Users/ased/.config/opencode/skills/playwright/SKILL.md) | Playwright CLI reference |

---

*End of SKILL.md - Updated 2026-04-27 with import/export application tools and unified browser automation guidance*
