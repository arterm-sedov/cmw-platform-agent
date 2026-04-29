#!/usr/bin/env python3
"""
Apply or reverse localization renames from rename table.

Usage:
    python apply_renames.py --output-dir /path/to/output

    The script reads {output_dir}/{app}_verified_complete.json and applies renames.
    Duplicates are deduplicated by (type, alias) - each unique pair renamed once.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from tools.applications_tools.tool_update_object_property import update_object_property

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

def main(output_dir: str, reverse: bool = False):
    output_path = Path(output_dir)

    # Find the verified complete file
    json_files = list(output_path.glob("*_verified_complete.json"))
    if not json_files:
        print(f"No verified_complete.json found in {output_dir}")
        return

    rename_file = json_files[0]
    with open(rename_file) as f:
        table = json.load(f)

    print(f"Loaded {len(table)} entries from {rename_file.name}")

    # Deduplicate by (type, alias) - each unique alias renamed once
    unique_renames = {}
    for row in table:
        if not row.get("ids"):
            continue

        original = row.get("aliasOriginal", "")
        new_name = row.get("aliasRenamed", "")
        if not original or not new_name:
            continue

        key = (row.get("type"), original)
        if key not in unique_renames:
            unique_renames[key] = {
                "new_name": new_name,
                "ids": row.get("ids", []),
            }

    print(f"Unique (type, alias) pairs to rename: {len(unique_renames)}")

    success = 0
    failed = 0
    skipped = 0
    processed_ids = 0

    for (obj_type, alias), data in unique_renames.items():
        plat_type = type_map.get(obj_type)
        if not plat_type:
            skipped += 1
            continue

        target_name = alias if reverse else data["new_name"]

        for obj_id in data["ids"]:
            result = update_object_property.invoke({
                "object_id": obj_id,
                "object_type": plat_type,
                "new_value": target_name,
            })

            if result.get("success"):
                success += 1
            else:
                failed += 1
            processed_ids += 1

        time.sleep(0.02)

        if (success + failed) % 100 == 0:
            print(f"Progress: {success + failed} IDs | S: {success}, F: {failed}")

    print(f"\n=== Complete ===")
    print(f"Unique aliases renamed: {len(unique_renames)}")
    print(f"Total IDs processed: {processed_ids}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    print(f"Skipped (no type map): {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply or reverse localization renames")
    parser.add_argument("--output-dir", default="/tmp/cmw-transfer/Volga-extract/Volga_tr", help="Directory with _verified_complete.json")
    parser.add_argument("--reverse", action="store_true", help="Reverse renames (renamed -> original)")
    args = parser.parse_args()

    main(args.output_dir, args.reverse)