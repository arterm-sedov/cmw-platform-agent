#!/usr/bin/env python3
"""
Apply or reverse localization renames from _tr.json.

Usage:
    python apply_renames.py --app Volga --output-dir /path/to/output

    The script reads {output_dir}/{domain}_{app}_tr.json and applies renames.
    Duplicates are deduplicated by (type, alias) - each unique pair renamed once.

    Resumable: By default, skips entries with applied: true.
    Use --force to reprocess all entries regardless of applied status.

    Incremental save: Saves progress every 50 aliases to survive interruptions.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from tools.applications_tools.tool_update_object_property import update_object_property

type_map = {
    "RecordTemplate": "RecordTemplate",
    "ProcessTemplate": "ProcessTemplate",
    "Workspace": "Workspace",
    "Page": "Page",
    "SimplePage": "Page",
    "RoleTemplate": "RoleTemplate",
    "Role": "Role",
    "AccountTemplate": "AccountTemplate",
    "OrgStructureTemplate": "OrgStructureTemplate",
    "MessageTemplate": "MessageTemplate",
    "Cart": "Cart",
    "Trigger": "Trigger",
    "Attribute": "Attribute",
    "Dataset": "Dataset",
    "Form": "Form",
    "Toolbar": "Toolbar",
    "UserCommand": "UserCommand",
    "Card": "Card",
    "WidgetConfig": "WidgetConfig",
    "ExportTemplate": "ExportTemplate",
    "DesktopWidgetConfig": "DesktopWidgetConfig",
    "DesktopComponent": "DesktopComponent",
    "RoleConfiguration": "Attribute",
    "RoleWorkspace": "Workspace",
    "MessageTemplateProperty": "Attribute",
}


def get_domain() -> str:
    """Extract domain from config URL or environment."""
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


def filter_by_expressions(rows: list, mode: str) -> list:
    """Filter rows by expressions presence.

    Args:
        rows: List of row dicts with optional 'expressions' field
        mode: 'all' - all rows, 'safe' - no expressions, 'danger' - has expressions

    Returns:
        Filtered list of rows
    """
    if mode == "all":
        return rows
    if mode == "safe":
        return [r for r in rows if not r.get("expressions")]
    return [r for r in rows if r.get("expressions")]


def main(
    app: str,
    output_dir: str,
    reverse: bool = False,
    resume: bool = True,
    force: bool = False,
    mode: str = "safe",
):
    output_path = Path(output_dir)
    domain = get_domain()

    tr_file = output_path / f"{domain}_{app}_tr.json"
    if not tr_file.exists():
        print(f"No {domain}_{app}_tr.json found in {output_dir}")
        return

    with open(tr_file, encoding="utf-8") as f:
        table = json.load(f)

    print(f"Loaded {len(table)} entries from {tr_file.name}")

    already_applied = sum(1 for row in table if row.get("applied") == True)
    if already_applied > 0 and resume and not force:
        print(f"Found {already_applied} already applied entries (use --force to reprocess)")

    skip_applied = resume and not force

    filtered_table = []
    filtered_indices = []
    for idx, row in enumerate(table):
        if skip_applied and row.get("applied") == True:
            continue

        if row.get("failed") == True and row.get("aliasLocked") == True:
            row.pop("failed", None)

        if row.get("aliasLocked") == True:
            continue

        if not row.get("aliasRenamed"):
            continue

        if row.get("aliasRenamed") == row.get("aliasOriginal"):
            continue

        filtered_table.append(row)
        filtered_indices.append(idx)

    print(f"Processing {len(filtered_table)} entries after filtering")

    filtered_table = filter_by_expressions(filtered_table, mode)
    print(f"After mode filter ({mode}): {len(filtered_table)} entries")

    if mode == "danger":
        print("\nWARNING: Running in DANGER mode!")
        print("This will process aliases with expressions, which may have dependencies.")
        try:
            confirm = input("Are you sure you want to continue? (yes/no): ").strip().lower()
            if confirm not in ("yes", "y"):
                print("Aborted by user.")
                return
        except EOFError:
            print("Interactive input not available, proceeding...")

    unique_renames = {}
    for row in filtered_table:
        ids = row.get("ids", [])
        if not ids:
            continue

        original = row.get("aliasOriginal", "")
        new_name = row.get("aliasRenamed", "")
        if not original or not new_name:
            continue

        key = (row.get("type"), original)
        if key not in unique_renames:
            unique_renames[key] = {
                "new_name": new_name,
                "ids": ids,
                "row_indices": [],
            }

    for filtered_idx, row in enumerate(filtered_table):
        original = row.get("aliasOriginal", "")
        obj_type = row.get("type", "")
        key = (obj_type, original)
        if key in unique_renames:
            unique_renames[key]["row_indices"].append(filtered_idx)

    print(f"Unique (type, alias) pairs to rename: {len(unique_renames)}")

    success = 0
    failed = 0
    skipped = 0
    processed_ids = 0
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    save_interval = 50

    for alias_idx, (obj_type, alias) in enumerate(unique_renames.keys()):
        data = unique_renames[(obj_type, alias)]
        plat_type = type_map.get(obj_type)
        if not plat_type:
            skipped += 1
            continue

        target_name = alias if reverse else data["new_name"]
        all_success = True

        for obj_id in data["ids"]:
            result = update_object_property.invoke(
                {"object_id": obj_id, "object_type": plat_type, "new_value": target_name}
            )

            if result.get("success"):
                success += 1
            else:
                failed += 1
                all_success = False
            processed_ids += 1

        for filtered_row_idx in data["row_indices"]:
            original_idx = filtered_indices[filtered_row_idx]
            if all_success:
                table[original_idx]["applied"] = True
                table[original_idx]["appliedAt"] = timestamp
            else:
                table[original_idx]["applied"] = False
                table[original_idx]["failed"] = True

        time.sleep(0.02)

        if (alias_idx + 1) % save_interval == 0:
            with open(tr_file, "w", encoding="utf-8") as f:
                json.dump(table, f, indent=2, ensure_ascii=False)
            print(f"[Auto-saved] Progress: {alias_idx + 1}/{len(unique_renames)} aliases")

    with open(tr_file, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)

    print("\n=== Complete ===")
    print(f"Unique aliases renamed: {len(unique_renames)}")
    print(f"Total IDs processed: {processed_ids}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    print(f"Skipped (no type map): {skipped}")
    print(f"Skipped (already applied): {already_applied if resume and not force else 0}")
    print(f"Updated {tr_file.name} with applied status")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply or reverse localization renames")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory with {domain}_{app}_tr.json",
    )
    parser.add_argument("--reverse", action="store_true", help="Reverse renames (renamed -> original)")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip already applied entries (default: True)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force reprocess all entries, ignore applied status",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "safe", "danger"],
        default="safe",
        help="Filter aliases by expressions: 'all' - all, 'safe' - no expressions, 'danger' - has expressions (default: safe)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or f"/tmp/cmw-transfer/{args.app}_tr"
    main(args.app, output_dir, args.reverse, args.resume, args.force, args.mode)