---
name: cmw-platform
description: Use when working with Comindware Platform — connecting to platform, listing applications, exploring templates, managing records, querying data, creating or editing attributes, or any task requiring autonomous platform interaction. Triggers on platform operations, CMW queries, tenant management, rental lot operations, debt tracking, or record CRUD. Also triggers when user mentions working directly with API credentials, using manual HTTP approaches, or trying to bypass the tool layer.
---

# cmw-platform Skill

Enables autonomous interaction with Comindware Platform using tools from the agent's tools directory.

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

## Diagnostic Script

```bash
python .agents/skills/cmw-platform/scripts/diagnose_connection.py
```

Exit code 0 = pass, 1 = fail. Run this to verify platform connectivity.

## Working Files

**Always store intermediate steps and results in `cmw-platform-workspace/` directory.**

This directory is gitignored - use it for:
- Evaluation outputs
- Intermediate query results
- Debug logs
- Test artifacts

If the LLM hangs or context is lost, retry can resume from saved files instead of starting over.