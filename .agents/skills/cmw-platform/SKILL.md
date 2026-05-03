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

### Guiding Principles

Persist Context. Read Before Write. Idempotent Operations. Explicit Over Implicit.

→ See also: [references/principles.md](references/principles.md)

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

### Knowledge Base

When uncertain about platform behavior, call the MCP tool `get_knowledge_base_articles` from the Comindware knowledge base server. Never use `ask_comindware` MCP tool.

→ See also: [references/knowledge_base.md](references/knowledge_base.md)

### Response Structure

```python
{
    "success": bool,      # True if operation succeeded
    "status_code": int,   # HTTP status code
    "data": list|dict,     # Response payload
    "error": str|dict      # Error details if success=False
}
```

### System Prompt Alignment

For platform ops, `agent_ng/system_prompt.json` is PRIMARY (agentic behavior); `AGENTS.md` is SECONDARY (project work).

→ See also: [references/system_prompt_alignment.md](references/system_prompt_alignment.md)

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
→ See also: [references/working_files.md](references/working_files.md) for the fetch-and-save pattern + file naming convention.

---

## 2. Browser Automation

Use browser automation when the operation is not available via API. All four options are pre-configured in `~/.config/opencode/opencode.json`.

### Decision Guide

| Scenario | Use |
|----------|-----|
| Agent conversation, Chrome OK | **agent-browser MCP** — most token-efficient (~5.7× vs Playwright) |
| Cross-browser / visual debugging / complex forms | **playwright MCP** — headed by default, rich `[ref=eN]` snapshots |
| Standalone script, persistent sessions, Chrome OK | **agent-browser CLI** — named sessions survive shell restarts |
| Standalone script, cross-browser, PDF / tracing | **playwright-cli** — no session persistence, full Playwright feature set |

### When to Use Browser vs API

| Operation | Use API | Use Browser |
|-----------|---------|-------------|
| List records | ✅ Fast, structured | ❌ Slow, parsing needed |
| Create/edit attributes | ✅ Direct, reliable | ❌ Complex UI navigation |
| Visual workflow designer | ❌ No API | ✅ UI-only feature |
| Admin panel configuration | ⚠️ Limited API | ✅ Full access |
| Verify UI changes | ❌ Can't see UI | ✅ Screenshots |
| Extract UI table data | ⚠️ If no API | ✅ Fallback option |

### Quick Invocation Reference

**agent-browser MCP** (tool calls in agent, see [browser_automation.md](references/browser_automation.md#option-1-agent-browser-mcp)):

```
browser_new_session → browser_navigate → browser_snapshot → browser_click / browser_fill → browser_screenshot → browser_close_session
```

**playwright MCP** (tool calls in agent — headed by default):

```
playwright_browser_navigate → playwright_browser_snapshot → playwright_browser_fill_form / playwright_browser_click → playwright_browser_take_screenshot
```

⚠️ Re-snapshot after every DOM-changing action — refs (`[ref=eN]`) invalidate on navigation.

**agent-browser CLI** (PowerShell):

```powershell
agent-browser open "https://host/" ; agent-browser snapshot -i ; agent-browser fill @e14 "user" ; agent-browser screenshot out.png
```

**playwright-cli** (PowerShell):

```powershell
playwright-cli open "https://host/" ; playwright-cli snapshot ; playwright-cli fill e14 "user" ; playwright-cli screenshot page.png
```

The flows above use **short names**; your MCP host may namespace tools (names differ by client). Example full names used in this skill’s reference: `agent-browser_browser_*`, `playwright_browser_*` — see [MCP Tool Interface Reference](references/browser_automation.md#mcp-tool-interface-reference). Call whatever appears in your client’s tool list.

### Credentials

Always load from `.env` — never hardcode. See [references/browser_automation.md](references/browser_automation.md#credentials-loading-from-env) for Python and PowerShell patterns.

### CMW Platform URL Patterns

See [references/browser_automation.md](references/browser_automation.md#cmw-platform-url-patterns-spa-hash-routing) for the entity ID prefix registry (`oa.{N}`, `event.{N}`, etc.), all admin/template/form URL patterns, and the `resolve_entity` tool reference.

### Best Practices

1. **Snapshot → Act → Re-snapshot** — after every DOM-changing action
2. **Save before edit** — screenshot current state before destructive changes
3. **Session isolation** — unique `session_name` / `sessionId` per workflow
4. **Headed for debugging, headless for automation**
5. **PowerShell, not bash** — use `;` not `&&`, `Get-Content` not `cat`

→ See also: [references/browser_automation.md](references/browser_automation.md) — full tool lists for all 4 options, CMW Platform URL patterns, troubleshooting, platform testing lessons

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

For batch exploration, use `explore_templates.py`.

→ See also: [references/workflow_sequences.md](references/workflow_sequences.md) — ready-made scripts and usage patterns.

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

For script-backed data operations, use:
- `query_with_filter.py`
- `analyze_stats.py`
- `batch_edit_attributes.py`

→ See also: [references/workflow_sequences.md](references/workflow_sequences.md) — ready-made scripts, batch edit workflow, and usage examples.

---

## 5. Import/Export Applications

Use import/export when you need to move an entire application in CTF form.

- **Export** — create a CTF package for a full application
- **Import** — upload and apply a CTF package to create or update an application

→ See also: [references/import_export.md](references/import_export.md)
→ See also: [references/api_endpoints.md](references/api_endpoints.md)

---

## 6. UI Components

Datasets, Toolbars, and Buttons are **separate API entities** with different endpoints:

| Entity | Tool to Get | Tool to Edit |
|--------|-------------|--------------|
| Dataset | `get_dataset` | `edit_or_create_dataset` |
| Toolbar | `get_toolbar` | `edit_or_create_toolbar` |
| Button | `get_button` | `edit_or_create_button` |

Keep these core rules in mind:
- **Create-kind buttons** require `create_form`
- **Toolbar item names** override button display names
- **Dataset-specific toolbars** should not be shared blindly across datasets
- **Verify UI component edits via API**, not only via the UI, because the UI may cache stale state

→ See also: [references/ui_components.md](references/ui_components.md)
→ See also: [references/tool_inventory.md](references/tool_inventory.md)
→ See also: [references/workflow_sequences.md](references/workflow_sequences.md)

---

## 7. Localization (System Names)

Two distinct workflows — choose before starting:

| Goal | Workflow |
|------|----------|
| Rename system names / aliases | **Workflow A** — 9-phase process with platform restart and CTF round-trip |
| Translate Russian UI strings to English | **Workflow B** — harvest → translate → apply → CSV |

→ See also: [references/localization.md](references/localization.md) — both workflows in full detail, phase map, suffix rules, `tool_localize`, step scripts, type/predicate mappings.

---

## 8. Troubleshooting

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

→ See also: [references/edit_or_create.md](references/edit_or_create.md) — required fields per type, per-tool validation rules, and partial-update mechanics.

### Diagnostic Script

Use `diagnose_connection.py` to verify platform connectivity.
Exit code `0` = pass, `1` = fail.

→ See also: [references/errors.md](references/errors.md) — diagnostic command and recovery guidance.

---

## Reference Index

| Document | Purpose |
|---------|---------|
| [references/principles.md](references/principles.md) | Guiding principles for all platform work |
| [references/working_files.md](references/working_files.md) | Fetch-and-save pattern + workspace file naming |
| [references/edit_or_create.md](references/edit_or_create.md) | `edit_or_create_*` validation, required fields, partial updates |
| [references/knowledge_base.md](references/knowledge_base.md) | `get_knowledge_base_articles` MCP usage |
| [references/system_prompt_alignment.md](references/system_prompt_alignment.md) | `system_prompt.json` vs `AGENTS.md` precedence |
| [references/tool_inventory.md](references/tool_inventory.md) | Complete tool catalog with signatures |
| [references/api_endpoints.md](references/api_endpoints.md) | HTTP endpoint reference |
| [references/import_export.md](references/import_export.md) | Full import/export application reference |
| [references/ui_components.md](references/ui_components.md) | Full UI components reference: toolbars, buttons, datasets, forms |
| [references/errors.md](references/errors.md) | Error handling playbook |
| [references/workflow_sequences.md](references/workflow_sequences.md) | Reusable code patterns |
| [references/localization.md](references/localization.md) | Both localization workflows: alias rename (9-phase) + RU→EN UI text translation |
| [references/browser_automation.md](references/browser_automation.md) | Browser automation guide |

## Optional companion skills (GitHub)

- [web-search](https://github.com/arterm-sedov/web-search-skill)
- [human-search](https://github.com/arterm-sedov/human-search-skill)
- [searxng-agent-skills](https://github.com/arterm-sedov/searxng-agent-skills)
- [browser-switch](https://github.com/arterm-sedov/browser-switch)
- [deep-research](https://github.com/arterm-sedov/deep-research)
- [agent-browser](https://github.com/vercel-labs/agent-browser)
- [Playwright CLI](https://playwright.dev/docs/cli)
- [microsoft/playwright](https://github.com/microsoft/playwright)

Install: [`npx skills add`](https://www.npmjs.com/package/skills) (e.g. `npx skills add arterm-sedov/web-search-skill --skill web-search`).

---
