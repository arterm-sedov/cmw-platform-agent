# Workflow Sequences

## 1. Explore Application Structure

```python
from tools.applications_tools.tool_list_applications import list_applications
from tools.applications_tools.tool_list_templates import list_templates
from tools.templates_tools.tool_list_attributes import list_attributes

def explore_app(app_name: str):
    # Step 1: List all applications
    apps_result = list_applications.invoke({})
    if not apps_result["success"]:
        return {"error": apps_result["error"]}
    
    # Find target application
    target = next(
        (a for a in apps_result["data"] if app_name.lower() in a["Name"].lower()),
        None
    )
    if not target:
        return {"error": f"Application '{app_name}' not found"}
    
    # Step 2: List templates in target application
    templates_result = list_templates.invoke({
        "application_system_name": target["Application system name"]
    })
    if not templates_result["success"]:
        return {"error": templates_result["error"]}
    
    # Step 3: Get attributes for first 5 templates
    for tmpl in templates_result["data"][:5]:
        attrs_result = list_attributes.invoke({
            "application_system_name": target["Application system name"],
            "template_system_name": tmpl["Template system name"]
        })
        print(f"{tmpl['Name']}: {len(attrs_result.get('data', []))} attributes")
    
    return templates_result
```

## 2. Find Records with Filters

```python
from tools.templates_tools.tool_list_records import list_template_records

def find_records_with_filter(app_name: str, template_name: str, filter_attr: str, min_value: int = 0):
    result = list_template_records.invoke({
        "application_system_name": app_name,
        "template_system_name": template_name,
        "filters": {filter_attr: {"$gt": min_value}},
        "limit": 100
    })
    
    if not result["success"]:
        return {"error": result["error"]}
    
    # Extract relevant fields
    records = []
    for record in result["data"]:
        records.append({
            "id": record.get("id"),
            "Name": record.get("Name"),
            filter_attr: record.get(filter_attr)
        })
    
    return {"data": records, "count": len(records)}
```

## 3. Create a Record

```python
from tools.templates_tools.tool_create_edit_record import create_edit_record

def create_record(app_name: str, template_name: str, values: dict):
    result = create_edit_record.invoke({
        "operation": "create",
        "application_system_name": app_name,
        "template_system_name": template_name,
        "values": values
    })
    
    if not result["success"]:
        return {"error": result["error"]}
    
    return {"record_id": result.get("record_id")}
```

## 4. Edit a Record

```python
from tools.templates_tools.tool_create_edit_record import create_edit_record

def edit_record(record_id: str, app_name: str, template_name: str, values: dict):
    result = create_edit_record.invoke({
        "operation": "edit",
        "application_system_name": app_name,
        "template_system_name": template_name,
        "record_id": record_id,
        "values": values
    })
    
    if not result["success"]:
        return {"error": result["error"]}
    
    return {"record_id": result.get("record_id")}
```

## 5. Paginate Through Records

```python
from tools.templates_tools.tool_list_records import list_template_records

def fetch_all_records(app_name: str, template_name: str, page_size: int = 100):
    all_records = []
    offset = 0
    
    while True:
        result = list_template_records.invoke({
            "application_system_name": app_name,
            "template_system_name": template_name,
            "limit": page_size,
            "offset": offset
        })
        
        if not result["success"]:
            break
            
        page_data = result.get("data", [])
        if not page_data:
            break
            
        all_records.extend(page_data)
        
        if len(page_data) < page_size:
            break  # Last page
            
        offset += page_size
    
    return {"data": all_records, "total": len(all_records)}
```

## 6. Get Template Schema Before Creating

```python
from tools.templates_tools.tool_list_attributes import list_attributes
from tools.templates_tools.tool_create_edit_record import create_edit_record

def create_with_schema(app_name: str, template_name: str, values: dict):
    # First, get the template schema to understand attribute types
    schema_result = list_attributes.invoke({
        "application_system_name": app_name,
        "template_system_name": template_name
    })
    
    if not schema_result["success"]:
        return {"error": schema_result["error"]}

    # Build attribute type map
    attr_types = {}
    for attr in schema_result.get("data", []):
        attr_types[attr["Attribute system name"]] = attr["Attribute type"]
    
    # Now create the record with proper type coercion
    result = create_edit_record.invoke({
        "operation": "create",
        "application_system_name": app_name,
        "template_system_name": template_name,
        "values": values
    })
    
    return result
```

## 7. Safe Attribute Translation (READ → EDIT)

**The edit_or_create tools now support partial updates automatically.**

### How It Works

1. **Create**: All type-specific fields required (model validator enforces)
2. **Edit - partial**: Tool's `tool_utils.py` patch fetches current schema, fills missing fields
3. **Edit - explicit**: Provided values override existing ones

### Using the Tools Directly

```python
from tools.attributes_tools.tools_decimal_attribute import edit_or_create_numeric_attribute

# EDIT with only name - existing type/format PRESERVED via patch
edit_or_create_numeric_attribute.invoke({
    "operation": "edit",
    "name": "Lot Area",
    "system_name": "Ploschad",
    "application_system_name": "Volga",
    "template_system_name": "RentLots"
    # number_decimal_places NOT provided - patch fills from current schema
})
```

### Partial Update Behavior Summary

| Scenario | Field Provided? | Result |
|----------|----------------|--------|
| Edit - nothing | No | Patch fills missing → **preserved** |
| Edit - value | Yes | Value sent → **overridden** |
| Edit - explicit None | None | Stripped → patch fills → **preserved** |
| Create - missing | N/A | Validator error → **rejected** |

→ Full validation rules, required fields per type, and System Buttons warning: [edit_or_create.md](edit_or_create.md)

## Ready-Made Scripts

These scripts are in `scripts/` directory and can be run directly:

| Script | Purpose | Usage |
|--------|---------|-------|
| `diagnose_connection.py` | Verify platform connectivity | `python scripts/diagnose_connection.py` |
| `explore_templates.py` | Explore multiple templates | `python scripts/explore_templates.py --app <app> --templates T1,T2` |
| `query_with_filter.py` | Paginated query with in-code filter | `python scripts/query_with_filter.py --app <app> --template <tmpl> --filter-attr <attr> --filter-op gt --filter-value 0` |
| `analyze_stats.py` | Statistical analysis of numeric attributes | `python scripts/analyze_stats.py --app <app> --template <tmpl> --attr <attr> --top 10` |
| `batch_edit_attributes.py` | Batch edit multiple attributes in one call | `python scripts/batch_edit_attributes.py --app <app> --template <tmpl> --mapping edits.json [--dry-run | --execute]` |

## Batch Edit Workflow

For editing multiple attributes at once, use `batch_edit_attributes.py`:

```bash
# Preview changes (dry-run)
python .agents/skills/cmw-platform/scripts/batch_edit_attributes.py \
    --app FacilityManagement --template Buildings \
    --mapping edits.json --dry-run

# Execute changes
python .agents/skills/cmw-platform/scripts/batch_edit_attributes.py \
    --app FacilityManagement --template Buildings \
    --mapping edits.json --execute
```

### Mapping File Format

Any attribute field can be included. Fields not provided are preserved.

```json
{
    "attributes": {
        "AttrSystemName": {
            "name": "New Name",
            "description": "New Description",
            "isMandatory": true
        },
        "AnotherAttr": {
            "name": "Another Name",
            "displayFormat": "PlainText"
        }
    }
}
```

Supported types: String, Text, Decimal, Enum, Record, DateTime, Document, Image, Duration, Account.

## 8. Dataset-Specific Toolbars (3-step workflow)

If a dataset shares a toolbar with other datasets, editing that toolbar will affect ALL linked datasets.
To have buttons specific to ONE dataset only, create a dedicated toolbar and link it:

```python
from tools.templates_tools.tools_toolbar import edit_or_create_toolbar
from tools.templates_tools.tools_dataset import edit_or_create_dataset

# Step 1: Create toolbar
edit_or_create_toolbar.invoke({
    "operation": "create",
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "toolbar_system_name": "<toolbar>",
    "name": "<Name>"
})

# Step 2: Add items to toolbar
edit_or_create_toolbar.invoke({
    "operation": "edit",
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "toolbar_system_name": "<toolbar>",
    "items": [
        {"button_system_name": "<button>", "display_name": "<Label>", "item_order": 0},
    ]
})

# Step 3: Link toolbar to dataset
edit_or_create_dataset.invoke({
    "operation": "edit",
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "dataset_system_name": "<dataset>",
    "toolbar_system_name": "<toolbar>"
})
```

## 9. Toolbar Item Names Override Button Names

Toolbar items have their own `name` field that **overrides** the button's display name.
To change a button's label as shown in a toolbar, edit the toolbar item — not the button.

```python
# WRONG: Editing button name won't affect toolbar display
edit_or_create_button.invoke({
    "operation": "edit",
    "button_system_name": "<button>",
    "name": "<Label>"
})

# CORRECT: Update the toolbar item's display_name
edit_or_create_toolbar.invoke({
    "operation": "edit",
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "toolbar_system_name": "<toolbar>",
    "items": [
        {"button_system_name": "<button>", "display_name": "<Label>", "item_order": 0}
    ]
})
```

## 10. Archive / Unarchive Button

```python
from tools.templates_tools.tools_button import archive_unarchive_button

# Archive a button
archive_unarchive_button.invoke({
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "button_system_name": "<button>",
    "operation": "archive"
})

# Unarchive a button
archive_unarchive_button.invoke({
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "button_system_name": "<button>",
    "operation": "unarchive"
})
```

**⚠️ Do not archive system buttons** (`create`, `edit`, `archive`, `delete`, `unarchive`).

## 11. Dataset Advanced Options

### Column Edit Operations

```python
edit_or_create_dataset.invoke({
    "operation": "edit",
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "dataset_system_name": "<dataset>",
    "columns": {
        "<column>": {"Name": "<New Label>"},               # rename existing column
        "<columnToHide>": {"isHidden": True},              # hide column from UI
        "<columnToDelete>": {"_delete": True},             # delete column (also: null)
        "<newColumn>": {                                    # add new column
            "Name": "<Label>",
            "propertyPath": [{"type": "Attribute", "owner": "<template>", "alias": "<attribute>"}]
        }
    }
})
```

### Sorting and Grouping Options

```python
edit_or_create_dataset.invoke({
    "operation": "edit",
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "dataset_system_name": "<dataset>",
    "is_default": True,            # set as default dataset
    "show_disabled": False,        # hide disabled records
    "sorting": [
        {"propertyPath": [...], "direction": "Asc", "nullValuesOnTop": False}
    ],
    "grouping": [
        {
            "propertyPath": [...],
            "name": "<GroupName>",
            "direction": "Asc",
            "level": 1,
            "fields": [
                {
                    "propertyPath": [...],
                    "aggregationMethod": "Count",  # Count|Sum|Min|Max|Avg
                    "type": "Number",              # String|Number|Boolean|Record
                    "format": "Undefined"
                }
            ]
        }
    ]
})
```

**Note:** Add `fields` with `aggregationMethod` to enable totals/summary rows in grouping views.
