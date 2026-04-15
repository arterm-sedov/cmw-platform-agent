# Tool Inventory

All tools live in `tools/` directory. Import pattern:
```python
from tools.<category>.<tool_file> import <tool_function>
```

## Applications Tools

### list_applications
- **Import:** `from tools.applications_tools.tool_list_applications import list_applications`
- **Signature:** `list_applications.invoke({})`
- **Parameters:** None (empty dict)
- **Returns:** `{"success": bool, "data": [{"Application system name": str, "Name": str, ...}]}`

### list_templates
- **Import:** `from tools.applications_tools.tool_list_templates import list_templates`
- **Signature:** `list_templates.invoke({"application_system_name": str, "template_type": "record"})`
- **Parameters:**
  - `application_system_name` (required): System name of the application
  - `template_type` (optional): "record" | "process" | "account" (default: "record")
- **Returns:** `{"success": bool, "data": [{"Template system name": str, "Name": str, ...}]}`

## Templates Tools

### list_attributes
- **Import:** `from tools.templates_tools.tool_list_attributes import list_attributes`
- **Signature:** `list_attributes.invoke({"application_system_name": str, "template_system_name": str})`
- **Parameters:**
  - `application_system_name` (required)
  - `template_system_name` (required)
- **Returns:** `{"success": bool, "data": [{"Attribute system name": str, "Attribute type": str, "Name": str, ...}]}`

### list_template_records
- **Import:** `from tools.templates_tools.tool_list_records import list_template_records`
- **Signature:** `list_template_records.invoke({...})`
- **Parameters:**
  - `application_system_name` (required): Application system name
  - `template_system_name` (required): Template system name
  - `attributes` (optional): List of attribute system names to return
  - `filters` (optional): Dict of attribute->value filters (client-side)
  - `limit` (optional): 1-100, default 100, max 100 per call
  - `offset` (optional): Pagination offset, default 0
  - `sort_by` (optional): Attribute to sort by, default "creationDate"
  - `sort_desc` (optional): True for descending, default False
- **Returns:** `{"success": bool, "data": [record, ...]}`

### create_edit_record
- **Import:** `from tools.templates_tools.tool_create_edit_record import create_edit_record`
- **Signature:** `create_edit_record.invoke({...})`
- **Parameters:**
  - `operation` (required): "create" or "edit"
  - `application_system_name` (required)
  - `template_system_name` (required)
  - `values` (required): Dict of attribute system name -> value pairs
  - `record_id` (optional): Required for edit operation
- **Returns:** `{"success": bool, "record_id": str|None, "error": str|None}`

## Attributes Tools

Attribute tools are organized by type. Pattern:
```python
from tools.attributes_tools.tool_<type>_attribute import create_or_edit_<type>_attribute
```

Available types:
- `text` (String attributes)
- `decimal` (Decimal attributes)
- `enum` (Enum attributes)
- `boolean` (Boolean attributes)
- `datetime` (DateTime attributes)
- `document` (Document attributes)
- `image` (Image attributes)
- `drawing` (Drawing attributes)
- `role` (Role attributes)
- `record` (Record attributes)
- `account` (Account attributes)

## Response Structure

All tools return this structure:
```python
{
    "success": bool,      # True if operation succeeded
    "status_code": int,   # HTTP status code
    "data": list|dict,    # Response payload (None on error)
    "error": str|dict     # Error details if success=False
}
```

## Client-Side Filtering

`list_template_records` supports client-side filtering via `filters` dict:
```python
filters = {"FieldName": value}           # Exact match
filters = {"FieldName": {"$gt": 30}}     # Greater than (for numeric)
```

## Pagination

Hard limit: 100 records per request. Paginate using `offset`:
```python
# Page 1
result = list_template_records.invoke({"offset": 0, "limit": 100})
# Page 2
result = list_template_records.invoke({"offset": 100, "limit": 100})
```