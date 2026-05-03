# Working Files

**ALWAYS save fetched schemas to `cmw-platform-workspace/` immediately after fetching.**

**Never rely solely on in-memory context.** Every time you fetch a schema, record, or query result, save it to a file right away. This enables:

- Recovery after context loss or interruption
- Comparison before/after changes
- Reference during future sessions

This directory is gitignored — use it for:

- Complete schemas (before and after changes)
- **Temporary scripts** created during exploration or debugging
- Evaluation outputs
- Intermediate query results
- Debug logs
- Test artifacts
- Any ad-hoc Python scripts for data analysis or fixes

## Pattern: Fetch and Save Immediately

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

## File Naming Convention

```
{entity}_schema_BEFORE.json   # State before changes
{entity}_schema_AFTER.json    # State after changes (optional)
{entity}_changes.json         # Summary of what changed
```

**If the LLM hangs or context is lost, retry can resume from saved files.**

---

→ Why this matters: [principles.md](principles.md)
