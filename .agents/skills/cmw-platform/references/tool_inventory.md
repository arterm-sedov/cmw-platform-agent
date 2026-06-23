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

### archive_unarchive_button
- **Import:** `from tools.templates_tools.tools_button import archive_unarchive_button`
- **Signature:** `archive_unarchive_button.invoke({...})`
- **Parameters:**
  - `operation` (required): `"archive"` or `"unarchive"`
  - `application_system_name` (required): App system name
  - `template_system_name` (required): Template system name
  - `button_system_name` (required): Button system name
- **Returns:** `{"success": bool, "status_code": int, "data": dict, "error": str|dict}`
- **Note:** Only use on non-system buttons. System buttons (`create`, `edit`, `archive`, `delete`, `unarchive`) should not be archived.

### get_form
- **Import:** `from tools.templates_tools.tools_form import get_form`
- **Signature:** `get_form.invoke({"application_system_name": str, "template_system_name": str, "form_system_name": str})`
- **Returns:** `{"success": bool, "data": dict, "error": str|dict}`
- **Note:** Verify visible forms by counting `FieldComponent` nodes in `data.root`.

### list_forms
- **Import:** `from tools.templates_tools.tools_form import list_forms`
- **Signature:** `list_forms.invoke({"application_system_name": str, "template_system_name": str})`
- **Returns:** `{"success": bool, "data": list, "error": str|dict}`

### edit_or_create_form
- **Import:** `from tools.templates_tools.tools_form import edit_or_create_form`
- **Signature:** `edit_or_create_form.invoke({...})`
- **Parameters:** `operation`, `application_system_name`, `template_system_name`, `form_system_name`, optional `name`, `form_size`, `widgets`, `replace_root`, `create_from_attributes`, `copy_from`, `verify`
- **Behavior:** `widgets` are desired visible fields to upsert. Empty forms rebuild a root tree with `FieldComponent` nodes.

### copy_form_from_template
- **Import:** `from tools.templates_tools.tool_copy_form import copy_form_from_template`
- **Signature:** `copy_form_from_template.invoke({...})`
- **Behavior:** Copies a source form, rewrites source tokens, replaces the target via DELETE + POST, and verifies `FieldComponent` nodes.

### create_form_from_attributes
- **Import:** `from tools.templates_tools.tool_create_form_from_attributes import create_form_from_attributes`
- **Signature:** `create_form_from_attributes.invoke({"application_system_name": str, "template_system_name": str, "form_system_name": "defaultForm", "form_name": str})`
- **Behavior:** Lists attributes, excludes system/archived attributes, builds a visible form, and verifies field count.

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

## Transfer Tools

### export_application
- **Import:** `from tools.transfer_tools.tool_export_application import export_application`
- **Signature:** `export_application.invoke({...})`
- **Parameters:**
  - `application_system_name` (required): System name of the application to export
  - `save_to_file` (optional): True saves CTF to /tmp/cmw-transfer/ (default: True)
- **Returns:** `{"success": bool, "ctf_data": str, "ctf_file_path": str, "result_message": str}`

### import_application
- **Import:** `from tools.transfer_tools.tool_import_application import import_application`
- **Signature:** `import_application.invoke({...})`
- **Parameters:**
  - `application_system_name` (required): System name for the imported application
  - `ctf_data` (optional): Base64-encoded CTF string
  - `ctf_file_path` (optional): Path to .ctf file (takes precedence over ctf_data)
- **Returns:** `{"success": bool, "file_id": str, "validation_errors": list|null}`
- **Note:** Import is a 2-step process (upload + execute). Use ctf_file_path for simplicity.

## Knowledge Base Tools

### get_knowledge_base_articles
- **Server:** Comindware knowledge base MCP (tool `get_knowledge_base_articles`)
- **Invocation:** Call via your MCP harness (not via Python import)
- **Parameters:**
  - `query` (required): Natural language search query about platform behavior
  - `top_k` (optional): Number of articles to return (default: 5)
- **Returns:** List of relevant documentation articles with titles and content
- **Use when:** Uncertain about attribute types, API behavior, or platform best practices

→ Full guidance and anti-patterns: [knowledge_base.md](knowledge_base.md)
