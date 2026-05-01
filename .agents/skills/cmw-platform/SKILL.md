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

When uncertain about platform behavior, use the `cmw_platform_knowledge-base` MCP `get_knowledge_base_articles` tool. Never use `ask_comindware`.

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

**agent-browser MCP** (tool calls in agent):

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

## 6. UI Components

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

→ See also: [references/workflow_sequences.md](references/workflow_sequences.md#8-dataset-specific-toolbars-3-step-workflow)

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

→ See also: [references/workflow_sequences.md](references/workflow_sequences.md#9-toolbar-item-names-override-button-names)

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

## 7. Localization (System Names)

System-name (alias) rename workflow for Comindware Platform applications. Keep this section as the decision and phase map; use the reference files for the full 9-phase procedure and for RU→EN UI text translation.

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

**Suffix rules:** `dangerous` aliases used in expressions usually get `_calc`; `safe` aliases used only in alias fields usually get `_sv`.

### Key Tools

| Purpose | Tool / Reference |
|---------|------------------|
| Export / re-export CTF | `export_application` |
| Verify aliases in platform | `get_ontology_objects` |
| Apply alias rename in platform | `update_object_property` |
| Full alias-rename procedure | [references/localization_workflow.md](references/localization_workflow.md) |
| RU→EN UI text translation workflow | [references/localization.md](references/localization.md) |
| Large-app batched scripts | `.agents/skills/cmw-platform/scripts/tool_*.py` |

→ See also: [references/localization_workflow.md](references/localization_workflow.md) — full 9-phase alias rename workflow, type/predicate mappings, `tool_localize`, step scripts, and workspace outputs.

→ See also: [references/localization.md](references/localization.md) — Russian→English UI text translation guide (different workflow from alias rename).

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
| [references/errors.md](references/errors.md) | Error handling playbook |
| [references/workflow_sequences.md](references/workflow_sequences.md) | Reusable code patterns |
| [references/localization_workflow.md](references/localization_workflow.md) | Full 9-phase system-name (alias) rename workflow |
| [references/localization.md](references/localization.md) | Russian→English UI text translation guide |
| [references/browser_automation.md](references/browser_automation.md) | Browser automation guide |
| [browser-switch skill](file:///C:/Users/ased/.agents/skills/browser-switch/skills/browser-switch/SKILL.md) | Decide between agent-browser and Playwright |
| [agent-browser skill](file:///C:/Users/ased/.agents/skills/agent-browser/SKILL.md) | Full agent-browser CLI reference |
| [playwright-cli skill](file:///C:/Users/ased/.config/opencode/skills/playwright/SKILL.md) | Playwright CLI reference |

---
