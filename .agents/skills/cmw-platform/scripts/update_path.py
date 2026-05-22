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


def calculate_new_json_path(original_path: str, old_alias: str, new_alias: str, tr_data: list | None = None) -> str:
    """Replace alias in JSON path with new alias."""
    if old_alias == new_alias and not tr_data:
        return original_path

    # Determine separator from original path
    sep = "\\" if "\\" in original_path else "/"

    # Split path into parts
    parts = original_path.split(sep)

    # Replace alias in each part (folder names and filenames)
    result_parts = []
    for part in parts:
        if part == old_alias:
            result_parts.append(new_alias)
        elif part == f"{old_alias}.json":
            result_parts.append(f"{new_alias}.json")
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

    updated_count = 0

    for obj in tr_data:
        alias_orig = obj.get("aliasOriginal", "")
        alias_renamed = obj.get("aliasRenamed", "")
        if not alias_orig or not alias_renamed:
            continue

        json_paths_orig = obj.get("jsonPathOriginal", [])
        if json_paths_orig:
            obj["jsonPathRenamed"] = [
                calculate_new_json_path(p, alias_orig, alias_renamed, tr_data)
                for p in json_paths_orig
            ]
            updated_count += 1

        # Update jsonPathRenamed for each displayName in displayNames array
        display_names = obj.get("displayNames", [])
        for dn in display_names:
            dn_orig_paths = dn.get("jsonPathOriginal", [])
            if dn_orig_paths:
                dn["jsonPathRenamed"] = [
                    calculate_new_json_path(p, alias_orig, alias_renamed, tr_data)
                    for p in dn_orig_paths
                ]

        # Update jsonPathRenamed for each expression in expressions array
        expressions = obj.get("expressions", [])
        for expr in expressions:
            expr_orig_path = expr.get("jsonPathOriginal", "")
            if expr_orig_path:
                expr["jsonPathRenamed"] = calculate_new_json_path(
                    expr_orig_path, alias_orig, alias_renamed, tr_data
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