#!/usr/bin/env python3
"""
Update jsonPathRenamed from aliasRenamed in _tr.json file.

Usage:
    python update_path.py --app Volga --output-dir C:\\tmp\\cmw-transfer
"""
import argparse
import json
import sys
import re
from pathlib import Path

APP_DIR = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(APP_DIR))


def get_domain() -> str:
    import os
    domain = os.environ.get("CMW_DOMAIN", "")
    if domain:
        return domain
    try:
        base_url = os.environ.get("CMW_BASE_URL", "")
        if base_url:
            from urllib.parse import urlparse
            return urlparse(base_url).netloc.split(".")[0] or "cmw"
    except Exception:
        pass
    return "cmw"


def build_alias_map(tr_data: list) -> dict[str, str]:
    """Build mapping of aliasOriginal -> aliasRenamed from tr_data."""
    alias_map = {}
    for entry in tr_data:
        ao = entry.get("aliasOriginal", "")
        ar = entry.get("aliasRenamed", "")
        if ao and ar and ao not in alias_map:
            alias_map[ao] = ar
    return alias_map


def calculate_new_json_path(original_path: str, tr_data: list | None = None) -> str:
    """Replace ALL aliases in JSON path with their aliasRenamed."""
    if not tr_data:
        return original_path

    # Determine separator from original path
    sep = "\\" if "\\" in original_path else "/"

    # Build alias mapping
    alias_map = build_alias_map(tr_data)

    # Split path into parts (folders and filename)
    parts = original_path.split(sep)

    # Replace each part if it exists in alias map
    result_parts = []
    for part in parts:
        # Check if part is a folder name (no .json)
        if part in alias_map:
            result_parts.append(alias_map[part])
        # Check if part is a filename (has .json)
        elif part.endswith(".json"):
            base_name = part[:-5]  # Remove .json
            if base_name in alias_map:
                result_parts.append(f"{alias_map[base_name]}.json")
            else:
                result_parts.append(part)
        else:
            result_parts.append(part)

    return sep.join(result_parts)


def main(app: str, output_dir: str) -> int:
    output_path = Path(output_dir)
    domain = get_domain()

    tr_file = output_path / f"{domain}_{app}_tr.json"
    if not tr_file.exists():
        print(f"Error: Translation file not found: {tr_file}")
        return 1

    with open(tr_file, encoding="utf-8") as f:
        tr_data = json.load(f)

    # Build alias map once
    alias_map = build_alias_map(tr_data)
    print(f"Built alias map with {len(alias_map)} entries")

    updated_count = 0

    # First pass: update jsonPathRenamed for entries that have aliasRenamed
    for obj in tr_data:
        alias_orig = obj.get("aliasOriginal", "")
        alias_renamed = obj.get("aliasRenamed", "")
        if not alias_orig or not alias_renamed:
            continue

        json_paths_orig = obj.get("jsonPathOriginal", [])
        if json_paths_orig:
            obj["jsonPathRenamed"] = [
                calculate_new_json_path(p, tr_data)
                for p in json_paths_orig
            ]
            updated_count += 1

        # Update jsonPathRenamed for each displayName in displayNames array
        display_names = obj.get("displayNames", [])
        for dn in display_names:
            dn_orig_paths = dn.get("jsonPathOriginal", [])
            if dn_orig_paths:
                dn["jsonPathRenamed"] = [
                    calculate_new_json_path(p, tr_data)
                    for p in dn_orig_paths
                ]

    # Second pass: update expressions for ALL entries (even without aliasRenamed)
    # because expressions may reference OTHER aliases that were renamed
    for obj in tr_data:
        expressions = obj.get("expressions", [])
        for expr in expressions:
            expr_orig_path = expr.get("jsonPathOriginal", "")
            if expr_orig_path:
                expr["jsonPathRenamed"] = calculate_new_json_path(
                    expr_orig_path, tr_data
                )

    with open(tr_file, "w", encoding="utf-8") as f:
        json.dump(tr_data, f, indent=2, ensure_ascii=False)

    print(f"Updated jsonPathRenamed in {updated_count} objects")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update jsonPathRenamed from aliasRenamed")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--output-dir", required=True, help="Path to output directory with _tr.json")
    args = parser.parse_args()
    sys.exit(main(args.app, args.output_dir))