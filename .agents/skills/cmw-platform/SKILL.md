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
agent-browser open "https://your-platform.example.com/"
agent-browser wait --load networkidle
agent-browser snapshot -i
agent-browser fill @e14 "bobragent"
agent-browser fill @e20 "$env:CMW_PASSWORD"
agent-browser click @e21
agent-browser screenshot "cmw-platform-workspace/logged_in.png"
agent-browser close

# Headed mode (user can watch)
agent-browser --headed open "https://your-platform.example.com/"

# Named persistent session (survives across shell invocations)
agent-browser --session-name cmw-admin open "https://your-platform.example.com/"
# ... login once ...
agent-browser --session-name cmw-admin close   # state auto-saved
# Later:
agent-browser --session-name cmw-admin open "https://your-platform.example.com/#Settings/Administration"

# Chain commands (& not && in PowerShell pipelines; use ; for simple chaining)
agent-browser open "https://example.com"; agent-browser wait --load networkidle; agent-browser screenshot out.png

# Get cdp url to connect playwright to the same Chrome
agent-browser get cdp-url
```

**Auth vault (credentials stay encrypted, never in shell history):**
```powershell
$env:CMW_PASSWORD | agent-browser auth save cmw --url "https://your-platform.example.com/" --username bobragent --password-stdin
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
playwright-cli open "https://your-platform.example.com/"
playwright-cli snapshot
playwright-cli click e14
playwright-cli fill e14 "bobragent"
playwright-cli fill e20 "$env:CMW_PASSWORD"
playwright-cli click e21
playwright-cli screenshot page.png
playwright-cli close

# Headed mode (user can watch)
playwright-cli --headed open "https://your-platform.example.com/"

# Cross-browser testing
playwright-cli --browser firefox open "https://your-platform.example.com/"
playwright-cli --browser webkit open "https://your-platform.example.com/"

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

## 5. UI Components

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

## 6. Localization

Russian→English translation workflow for Comindware Platform JSON configs.

### Workflow

1. **Harvest** strings from JSON files
2. **Build** translation dictionary
3. **Apply** translations to files
4. **Update** CSV reference

### Scripts

```bash
# Extract translatable strings (outputs JSON)
python .agents/skills/cmw-platform/scripts/harvest_strings.py \
    "path/to/Workspaces" --output harvested.json

# Build translation dict (edit JSON manually or use LLM)
python .agents/skills/cmw-platform/scripts/build_translations.py \
    harvested.json --output translations.json

# Apply translations to JSON files
python .agents/skills/cmw-platform/scripts/apply_translations.py \
    "path/to/Workspaces" translations.json

# Update CSV reference
python .agents/skills/cmw-platform/scripts/update_csv.py \
    translations.json translations.csv
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

*End of SKILL.md - Updated 2026-04-27: unified browser automation section covering agent-browser MCP, playwright MCP, and agent-browser CLI*
