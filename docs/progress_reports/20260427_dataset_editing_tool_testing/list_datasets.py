#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
from tools.templates_tools.tools_dataset import list_datasets
import json

result = list_datasets.invoke({
    "application_system_name": "FacilityManagement",
    "template_system_name": "MaintenancePlans"
})

print(f"Result success: {result.get('success')}")
print(f"Status code: {result.get('status_code')}")

if result.get('data'):
    datasets = result['data']
    print(f'\nTotal datasets: {len(datasets)}')
    for ds in datasets:
        print(f"  - {ds.get('alias', 'Unknown')}: {ds.get('name', 'Unknown')}")
        print(f"    Full: {json.dumps(ds, indent=6, default=str)}")
else:
    print(f"Error: {result.get('error')}")
