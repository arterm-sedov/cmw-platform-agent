#!/usr/bin/env python3
"""
Step 5: Finalize.

Merges all folder verified files into complete output in schema.json format:

Output fields (schema.json compliant):
  - type (str)
  - ids (array of str)
  - parent_template (str)
  - aliasOriginal (str)
  - aliasRenamed (str, empty by default)
  - displayNameOriginal (str)
  - displayNameRenamed (str, empty by default)
  - jsonPathOriginal (array of str)
  - jsonPathRenamed (array of str, empty by default)
  - expressions (array of {jsonPathOriginal, jsonPathRenamed, expressionOriginal, expressionRenamed})
"""

import argparse
import json
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(APP_DIR))


def main():
    parser = argparse.ArgumentParser(description="Step 5: Finalize verified aliases")
    parser.add_argument("--app", required=True)
    parser.add_argument("--output-dir", default="/tmp/cmw-transfer/Volga-extract/Volga_tr")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Step 5: Finalize for {args.app} ===")

    # Load all verified/skipped files
    verified = []
    skipped = []

    verified_files = sorted(output_dir.glob(f"{args.app}_*_verified.json"))
    print(f"Found {len(verified_files)} folder verified files")

    for vf in verified_files:
        try:
            with open(vf) as f:
                data = json.load(f)
            folder = data.get("folder", vf.stem.replace(f"{args.app}_", "").replace("_verified", ""))
            print(f"  {folder}: {data.get('verified_count', 0)} verified, {data.get('skipped_count', 0)} skipped")
            verified.extend(data.get("verified", []))
            skipped.extend(data.get("skipped", []))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: Failed to load {vf}: {e}")

    # Load dangerous aliases
    dangerous_file = output_dir / f"{args.app}_dangerous_aliases.json"
    dangerous = set()
    dangerous_expressions = []  # list of {alias, jsonPathOriginal, expressionOriginal}

    if dangerous_file.exists():
        try:
            with open(dangerous_file) as f:
                dangerous_data = json.load(f)
            dangerous = set(dangerous_data.get("dangerous_aliases", []))
            # Load expressions
            dangerous_expressions = dangerous_data.get("expressions", [])
            print(f"Loaded {len(dangerous)} dangerous aliases with {len(dangerous_expressions)} expressions")
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: Failed to load dangerous file: {e}")
    else:
        print(f"Warning: Dangerous aliases file not found at {dangerous_file}")

    print(f"\nUpdating aliasLocked flags...")

    # Process verified entries - map to schema format
    schema_verified = []
    for obj in verified:
        alias = obj.get("alias", obj.get("aliasOriginal", ""))
        display_name = obj.get("displayName", obj.get("displayNameOriginal", ""))
        parent_template = obj.get("parent_template", "")

        # Get ids (handle both "ids" and "id")
        obj_ids = obj.get("ids", obj.get("id", []))
        if isinstance(obj_ids, str):
            obj_ids = [obj_ids]
        elif not isinstance(obj_ids, list):
            obj_ids = []

        # Get jsonPathOriginal (handle both field names)
        json_path = obj.get("jsonPathOriginal", obj.get("path", obj.get("json_file", "")))
        if isinstance(json_path, str):
            json_path = [json_path] if json_path else []
        elif not isinstance(json_path, list):
            json_path = []

        # Get expressions for this alias
        expressions = [e for e in dangerous_expressions if e.get("alias", "") == alias]

        # Check if dangerous
        is_dangerous = alias in dangerous

        # aliasLocked logic for verified entries
        alias_locked = obj.get("aliasLocked", False)
        if alias_locked and is_dangerous:
            alias_locked = False  # Unlock dangerous aliases

        schema_obj = {
            "type": obj.get("type", ""),
            "ids": obj_ids,
            "parent_template": parent_template,
            "aliasOriginal": alias,
            "aliasRenamed": "",
            "displayNameOriginal": display_name,
            "displayNameRenamed": "",
            "jsonPathOriginal": json_path,
            "jsonPathRenamed": [],
            "expressions": expressions,
            "aliasLocked": alias_locked,
        }

        schema_verified.append(schema_obj)

    # Process skipped entries with new logic
    schema_skipped = []
    for obj in skipped:
        alias = obj.get("alias", obj.get("aliasOriginal", ""))
        display_name = obj.get("displayName", obj.get("displayNameOriginal", ""))
        parent_template = obj.get("parent_template", "")

        # Get jsonPathOriginal
        json_path = obj.get("jsonPathOriginal", obj.get("path", obj.get("json_file", "")))
        if isinstance(json_path, str):
            json_path = [json_path] if json_path else []
        elif not isinstance(json_path, list):
            json_path = []

        # Get expressions for this alias
        expressions = [e for e in dangerous_expressions if e.get("alias", "") == alias]

        is_dangerous = alias in dangerous

        if is_dangerous:
            # Skipped but dangerous - unlock for rename
            alias_locked = False
        elif display_name:
            # Has displayName - lock for rename
            alias_locked = True
        else:
            # No displayName - allow rename
            alias_locked = False

        schema_obj = {
            "type": obj.get("type", ""),
            "ids": [],
            "parent_template": parent_template,
            "aliasOriginal": alias,
            "aliasRenamed": "",
            "displayNameOriginal": display_name,
            "displayNameRenamed": "",
            "jsonPathOriginal": json_path,
            "jsonPathRenamed": [],
            "expressions": expressions,
            "aliasLocked": alias_locked,
        }

        schema_skipped.append(schema_obj)

    # Merge all entries
    all_entries = schema_verified + schema_skipped

    verified_locked = [v for v in all_entries if v.get("aliasLocked")]
    verified_normal = [v for v in all_entries if not v.get("aliasLocked")]

    print(f"\nTotal entries: {len(all_entries)}")
    print(f"  aliasLocked=true (will skip): {len(verified_locked)}")
    print(f"  aliasLocked=false (will rename): {len(verified_normal)}")

    # Output single file in schema format
    verified_file = output_dir / f"{args.app}_verified_complete.json"
    with open(verified_file, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    print(f"\n=== Final Output ===")
    print(f"Verified complete: {verified_file} ({len(all_entries)} objects)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
