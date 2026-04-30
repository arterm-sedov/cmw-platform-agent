#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
from tools.applications_tools.tool_list_applications import list_applications
import json

result = list_applications.invoke({})
apps = result.get('data', [])
print(f'Total apps: {len(apps)}')
for app in apps:
    print(f"  - {app.get('Application system name', 'Unknown')}")
