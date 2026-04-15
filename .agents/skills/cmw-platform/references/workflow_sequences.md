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

## Ready-Made Scripts

These scripts are in `scripts/` directory and can be run directly:

| Script | Purpose | Usage |
|--------|---------|-------|
| `diagnose_connection.py` | Verify platform connectivity | `python scripts/diagnose_connection.py` |
| `explore_templates.py` | Explore multiple templates | `python scripts/explore_templates.py --app <app> --templates T1,T2` |
| `query_with_filter.py` | Paginated query with in-code filter | `python scripts/query_with_filter.py --app <app> --template <tmpl> --filter-attr <attr> --filter-op gt --filter-value 0` |
| `analyze_stats.py` | Statistical analysis of numeric attributes | `python scripts/analyze_stats.py --app <app> --template <tmpl> --attr <attr> --top 10` |