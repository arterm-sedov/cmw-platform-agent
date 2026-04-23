---
name: cmw-platform
description: Use when working with Comindware Platform — connecting to platform, listing applications, exploring templates, managing records, querying data, creating or editing attributes, or any task requiring autonomous platform interaction. Triggers on platform operations, CMW queries, tenant management, rental lot operations, debt tracking, or record CRUD. Also triggers when user mentions working directly with API credentials, using manual HTTP approaches, or trying to bypass the tool layer. ALSO triggers when user mentions browser automation, UI-only features, visual verification, or accessing platform features not available via API.
---

# CMW Platform Skill

Enables autonomous interaction with Comindware Platform using tools from the agent's `tools/` directory.

**NEW:** Browser automation capabilities for UI-only features and visual verification.

---

## 1. Core Concepts

### Platform Terminology

**Critical:** Use human-readable terms, never expose API terms to LLMs:

| API Term | LLM-Friendly | Notes |
|----------|--------------|-------|
| alias | system name | Entity identifier |
| instance | record | Data entry |
| user command | button | UI action |
| container | template | Application/template |
| property | attribute | Field/column |
| dataset | table | List view |

### Workflow

Always follow: `Intent → Plan → Validate → Execute → Result`

### Tool Usage Discipline

- Check for duplicate calls before invoking
- Transform results to human-readable format (never raw JSON)
- Validate required context before execution

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

## 2. Exploration

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

## 3. Data Operations

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

## 6. Localization

Russian→English translation workflow for Comindware Platform.

### Step 1: Get Object IDs by Type

For CTF-based localization, first retrieve all object IDs using `get_ontology_objects`:

```python
from tools.applications_tools.tool_get_ontology_objects import get_ontology_objects

result = get_ontology_objects.invoke({
    "application_system_name": "supportTest",
    "types": ["RecordTemplate", "ProcessTemplate", "RoleTemplate", "AccountTemplate",
              "Dataset", "Form", "Toolbar", "UserCommand", "Card", "Cart", "Trigger"],
    "parameter": "alias",
    "min_count": 1,
    "max_count": 10000,
})

if result["success"]:
    print(f"Found {result['total_count']} objects")
    for obj in result["data"]:
        print(f"  - {obj['type']}: {obj['id']}")
```

**Returns:**

```python
{
    "success": true,
    "data": [
        {"type": "RecordTemplate", "id": "oa.230", "systemName": "oa.230"},
        {"type": "ProcessTemplate", "id": "pa.1", "systemName": "pa.1"},
        {"type": "Dataset", "id": "lst.558", "systemName": "lst.558"},
        {"type": "Form", "id": "form.2024", "systemName": "form.2024"},
        {"type": "Toolbar", "id": "tb.4465", "systemName": "tb.4465"},
        {"type": "UserCommand", "id": "event.7029", "systemName": "event.7029"},
        {"type": "Card", "id": "card.xxx", "systemName": "card.xxx"},
        ...
    ],
    "errors": {},
    "total_count": 20085
}
```

**Supported types:**

| Type | Predicate | ID Prefix |
|------|-----------|-----------|
| RecordTemplate | cmw.container.alias | oa. |
| ProcessTemplate | cmw.container.alias | pa. |
| RoleTemplate | cmw.container.alias | ra. |
| AccountTemplate | cmw.container.alias | aa. |
| OrgStructureTemplate | cmw.container.alias | os. |
| MessageTemplate | cmw.message.type.alias | msgt. |
| Workspace | cmw.alias | workspace. |
| Page | cmw.desktopPage.alias | - |
| Attribute | cmw.object.alias | - |
| Dataset | cmw.alias | lst. |
| Toolbar | cmw.alias | tb. |
| Form | cmw.alias | form. |
| UserCommand | cmw.alias | event. |
| Card | cmw.alias | card. |
| Cart | cmw.cart.alias | cart. |
| Trigger | cmw.trigger.alias | trigger. |
| Routes | cmw.procedure.name | - |

**Notes:**
- Results are filtered by ID prefix to match correct types
- Duplicate IDs are removed
- MessageTemplate: only IDs starting with `msgt.` are included

### Step 2: Update Object Properties

Use `update_object_property` to rename display names:

```python
from tools.applications_tools.tool_update_object_property import update_object_property

result = update_object_property.invoke({
    "object_id": "form.338",
    "object_type": "Form",
    "new_value": "MigrationForm",
})

if result["success"]:
    print(f"Updated {result['object_id']}: {result['predicate']} = {result['new_value']}")
```

**Important:** Alias values must be without spaces (use CamelCase or underscores).

**Supported types and predicates:**

| Type | Predicate |
|------|-----------|
| RecordTemplate | cmw.container.alias |
| ProcessTemplate | cmw.container.alias |
| RoleTemplate | cmw.container.alias |
| AccountTemplate | cmw.container.alias |
| OrgStructureTemplate | cmw.container.alias |
| MessageTemplate | cmw.message.type.alias |
| Workspace | cmw.alias |
| Page | cmw.desktopPage.alias |
| Attribute | cmw.object.alias |
| Dataset | cmw.alias |
| Toolbar | cmw.alias |
| Form | cmw.alias |
| UserCommand | cmw.alias |
| Card | cmw.alias |
| Cart | cmw.cart.alias |
| Trigger | cmw.trigger.alias |

### Step 3-4: Export/Import CTF for Localization

```bash
# Export app to CTF
python .agents/skills/cmw-platform/scripts/harvest_strings.py \
    "path/to/Workspaces" --output harvested.json

# Build translation dict
python .agents/skills/cmw-platform/scripts/build_translations.py \
    harvested.json --output translations.json

# Apply translations
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
| [references/browser_automation.md](references/browser_automation.md) | **NEW:** Browser automation guide |

---

*End of SKILL.md - Updated 2026-04-23 with import/export application tools*
