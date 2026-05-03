#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
from tools.applications_tools.tool_list_templates import list_templates
import json

result = list_templates.invoke({"application_system_name": "FacilityManagement"})
print(f"Result: {json.dumps(result, indent=2, default=str)}")

if result.get('success') and result.get('data'):
    templates = result['data']
    print(f'\nTotal templates: {len(templates)}')
    for tpl in templates:
        print(f"  - {tpl.get('Template system name', 'Unknown')}")
