---
name: cmw-platform
description: Use when working with Comindware Platform — connecting to platform, listing applications, exploring templates, managing records, querying data, creating or editing attributes, or any task requiring autonomous platform interaction. Triggers on platform operations, CMW queries, tenant management, rental lot operations, debt tracking, or record CRUD. Also triggers when user mentions working directly with API credentials, using manual HTTP approaches, or trying to bypass the tool layer.
---

# cmw-platform Skill

Enables autonomous interaction with Comindware Platform using tools from the agent's tools directory.

## Guiding Principles

**Follow these principles for all platform operations (per AGENTS.md):**

1. **Persist Context**: Always save complete schemas and results to `cmw-platform-workspace/` before making changes. This provides recovery points and prevents context loss.

2. **Read Before Write**: Fetch current state first, save it, then modify. Never edit without reading and persisting first.

3. **Idempotent Operations**: Design operations so running them multiple times produces the same result.

4. **Explicit Over Implicit**: If you provide a value, it overrides. If you don't, the patch preserves existing.

## Quick Start

1. Import tool: `from tools.<category>.<tool_file> import <tool_function>`
2. Invoke: `<tool>.invoke({...})`
3. Check: `result["success"]`

## Tool Invocation Pattern

```python
from tools.applications_tools.tool_list_applications import list_applications
result = list_applications.invoke({})
if not result["success"]:
    print(f"Error: {result["error"]}")
    return
print(result["data"])
```

## Response Structure

```python
{
    "success": bool,      # True if operation succeeded
    "status_code": int,   # HTTP status code
    "data": list|dict,    # Response payload
    "error": str|dict     # Error details if success=False
}
```

## Core Tools

| Tool | Import | Use |
|------|--------|-----|
| list_applications | tools.applications_tools.tool_list_applications | List apps |
| list_templates | tools.applications_tools.tool_list_templates | List templates |
| list_attributes | tools.templates_tools.tool_list_attributes | Get schema |
| list_template_records | tools.templates_tools.tool_list_records | Query records |
| create_edit_record | tools.templates_tools.tool_create_edit_record | CRUD records |

## Core Workflows

### Explore Application Structure

```python
from tools.applications_tools.tool_list_applications import list_applications
from tools.applications_tools.tool_list_templates import list_templates
from tools.templates_tools.tool_list_attributes import list_attributes

# 1. List applications
apps = list_applications.invoke({})
target_app = next(
    (a for a in apps["data"] if "target_application" in a["Name"]),
    None
)

if not target_app:
    return {"error": "Application not found"}

# 2. List templates in application
templates = list_templates.invoke({
    "application_system_name": target_app["Application system name"]
})

# 3. Get schema for each template
for tmpl in templates["data"][:5]:
    attrs = list_attributes.invoke({
        "application_system_name": target_app["Application system name"],
        "template_system_name": tmpl["Template system name"]
    })
```

### Query Records with Filters

```python
from tools.templates_tools.tool_list_records import list_template_records

# Find records matching filter criteria
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

## Error Handling

| Status | Meaning | Action |
|--------|---------|--------|
| 401 | Bad credentials | Check configuration |
| 408 | Query timeout | Reduce `limit` parameter (max 100) |
| 500 | Server error | Retry with exponential backoff |

### Retry Pattern

```python
import time

def retry_call(func, payload, max_retries=3):
    for attempt in range(max_retries):
        result = func.invoke(payload)
        if result["success"]:
            return result
        if result.get("status_code") in (500, 503, 408):
            time.sleep(2 ** attempt)
            continue
        return result
    return {"success": False, "error": "Max retries exceeded"}
```

## References

- [Tool Inventory](references/tool_inventory.md) - Complete tool catalog with signatures
- [API Endpoints](references/api_endpoints.md) - HTTP endpoint reference
- [Workflow Sequences](references/workflow_sequences.md) - Reusable code patterns
- [Error Playbook](references/errors.md) - Recovery procedures

## System Prompt Alignment

For all platform operations, follow this precedence:

**1. `agent_ng/system_prompt.json`** (PRIMARY - agentic behavior):
- Intent → Plan → Validate → Execute → Result workflow
- Tool usage discipline (no duplicate calls, cache results)
- CMW Platform vocabulary and terminology
- Idempotency and confirmation rules
- Error handling (401/403, 404, 409, 5xx)
- **Always use this for platform-specific guidance**

**2. `AGENTS.md`** (SECONDARY - project work guidelines):
- TDD/SDD principles
- Non-breaking changes
- Lean/Dry patterns
- Logging and persistence to workspace
- **General development approach, not platform-specific**

**For platform operations:**
1. Follow the Intent → Plan → Validate → Execute → Result from system_prompt.json
2. Use CMW Platform terminology (alias=system_name, instance=record, etc.)
3. Confirm risky edits before execution
4. Present results in human-readable format, not raw JSON

## Diagnostic Script

```bash
python .agents/skills/cmw-platform/scripts/diagnose_connection.py
```

Exit code 0 = pass, 1 = fail. Run this to verify platform connectivity.

## Safe Attribute Translation

### How edit_or_create Tools Work

The edit_or_create attribute tools have **smart partial update support**:

| Operation | Behavior | Mechanism |
|-----------|----------|----------|
| **Create** | Requires ALL type-specific fields | Model validator raises error if missing |
| **Edit - partial** | Missing fields fetched from API and merged | `tool_utils.py` patch fills gaps |
| **Edit - explicit** | Provided fields override existing values | User intent respected |

### How the Patch Works

When editing with missing fields:

```python
# Agent calls edit with only name/description
edit_or_create_numeric_attribute.invoke({
    "operation": "edit",
    "name": "New Name",
    "system_name": "Ploschad",
    "application_system_name": "Volga",
    "template_system_name": "RentLots"
    # number_decimal_places NOT provided
})
```

**What happens internally:**

1. `remove_values()` strips `None` fields → `number_decimal_places` not in body
2. `tool_utils.py` fetches current `Ploschad` schema from API
3. Patch merges: adds `decimalPlaces: 2` from current
4. API receives complete schema with `decimalPlaces: 2` preserved ✅

### Edit with Explicit Values

```python
# Agent provides value - THIS WILL OVERRIDE
edit_or_create_numeric_attribute.invoke({
    "operation": "edit",
    "name": "New Name",
    "system_name": "Ploschad",
    "number_decimal_places": 3  # ← Explicit value overrides existing
})
```

**Result:** API receives `decimalPlaces: 3` - existing value is overridden.

### Required Fields for Create

| Attribute Type | Required Fields |
|---------------|----------------|
| Decimal | `number_decimal_places` |
| Enum | `display_format`, `enum_values` |
| DateTime | `display_format` |
| Document | `display_format` |
| Image | `rendering_color_mode` |
| Duration | `display_format` |
| Account | `related_template_system_name` |
| Record | `related_template_system_name` |

### Safe Translation Scripts

```bash
# Safe script - direct API, guaranteed partial
python .agents/skills/cmw-platform/scripts/safe_translate_attribute.py \
    --app Volga --template RentLots --attr Ploschad --name "Lot Area" --desc "Area in sqm"

# Safe script - edit name/description only
python .agents/skills/cmw-platform/scripts/safe_edit_attribute.py \
    --app Volga --template RentLots --attr Ploschad --name "Lot Area"
```

These scripts bypass the tool layer entirely for guaranteed control.

## Working Files

**ALWAYS save fetched schemas to `cmw-platform-workspace/` immediately after fetching.**

**Never rely solely on in-memory context.** Every time you fetch a schema, record, or query result, save it to a file right away. This enables:
- Recovery after context loss or interruption
- Comparison before/after changes
- Reference during future sessions

This directory is gitignored - use it for:
- Complete schemas (before and after changes)
- **Temporary scripts** created during exploration or debugging
- Evaluation outputs
- Intermediate query results
- Debug logs
- Test artifacts
- Any ad-hoc Python scripts for data analysis or fixes

### Pattern: Fetch and Save Immediately

```python
import json
from tools.templates_tools.tool_list_attributes import list_attributes

# Step 1: FETCH and SAVE current complete schema
attrs = list_attributes.invoke({
    "application_system_name": "Volga",
    "template_system_name": "RentLots"
})

# Save immediately - don't wait for "before changes"
with open("cmw-platform-workspace/rentlots_schema_20260415.json", "w") as f:
    json.dump(attrs, f, indent=2)

# Step 2: Now you can safely work with the data
```

### File Naming Convention

```
{entity}_schema_BEFORE.json   # State before changes
{entity}_schema_AFTER.json    # State after changes (optional)
{entity}_changes.json        # Summary of what changed
```

**If the LLM hangs or context is lost, retry can resume from saved files.**