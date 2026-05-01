# Import / Export Applications

Use these operations when you need to move an entire Comindware Platform application in CTF form.

## Export Application

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

## Import Application

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

## API Endpoints

- Export: `GET /webapi/Transfer/{solutionAlias}`
- Upload: `POST /webapi/Transfer/Upload`
- Import: `POST /webapi/Transfer/{solutionAlias}/{fileId}/true/ApplyNew`

---

→ See also: [api_endpoints.md](api_endpoints.md)
