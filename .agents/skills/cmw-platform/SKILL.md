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

def check_dangerous_aliases(json_folder: str, aliases: set[str]) -> dict[str, bool]:
    dangerous = {a: False for a in aliases}
    for json_file in Path(json_folder).rglob("*.json"):
        content = open(json_file).read()
        for alias in aliases:
            if dangerous[alias]:
                continue
            pattern = r'"Expression"\s*:\s*"[^"]*' + re.escape(alias) + r'[^"]*"'
            if re.search(pattern, content):
                dangerous[alias] = True
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
        # Replace in alias field
        content = re.sub(r'"Alias"\s*:\s*"' + safe_pattern + r'"', '"Alias": "' + new_alias + '"', content)
        # Replace in expression field
        content = re.sub(r'"Expression"\s*:\s*"[^"]*' + safe_pattern + r'[^"]*"',
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
| [references/browser_automation.md](references/browser_automation.md) | **NEW:** Browser automation guide |

---

*End of SKILL.md - Updated 2026-04-23 with import/export application tools*
