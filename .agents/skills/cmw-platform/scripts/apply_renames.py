#!/usr/bin/env python3
"""
Apply or reverse localization renames from rename table.

Usage:
    python apply_renames.py              # Apply renames (original -> renamed)
    python apply_renames.py --reverse    # Reverse renames (renamed -> original)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.applications_tools.tool_update_object_property import update_object_property

RENAME_FILE = Path("/tmp/cmw-transfer/Volga1-rename-table.json")

type_map = {
    "RecordTemplate": "RecordTemplate",
    "ProcessTemplate": "ProcessTemplate",
    "Workspace": "Workspace",
    "Page": "Page",
    "RoleTemplate": "RoleTemplate",
    "Role": "Role",
    "AccountTemplate": "AccountTemplate",
    "OrgStructureTemplate": "OrgStructureTemplate",
    "Attribute": "Attribute",
    "Dataset": "Dataset",
    "Form": "Form",
    "Toolbar": "Toolbar",
    "UserCommand": "UserCommand",
    "WidgetConfig": "WidgetConfig",
}

def main(reverse: bool = False):
    if not RENAME_FILE.exists():
        print(f"File not found: {RENAME_FILE}")
        return

    with open(RENAME_FILE) as f:
        table = json.load(f)

    print(f"Total: {len(table)}")

    success = 0
    failed = 0

    for i, row in enumerate(table):
        if not row.get("id"):
            continue

        original = row["systemName"]
        new_name = row["renamedSystemName"]

        target_name = original if reverse else new_name
        plat_type = type_map.get(row["type"])

        if not plat_type:
            continue

        result = update_object_property.invoke({
            "object_id": row["id"],
            "object_type": plat_type,
            "new_value": target_name,
        })

        if result.get("success"):
            success += 1
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{len(table)} | S: {success}, F: {failed}")

        time.sleep(0.02)

    print(f"=== Complete ===")
    print(f"Success: {success}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    reverse = "--reverse" in sys.argv
    main(reverse)