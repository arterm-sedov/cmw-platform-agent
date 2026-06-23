#!/usr/bin/env python3
"""
Phase 1: Filter and deduplicate localization data
Phase 2: Apply or reverse renames from filtered files

Usage:
    Phase 1 (filter):
        python apply_renames.py --app Volga --phase filter --output-dir /path/to/output

    Phase 2 (apply):
        python apply_renames.py --app Volga --phase apply --mode safe --resume
        python apply_renames.py --app Volga --phase apply --mode danger --reverse --resume

Files:
    {domain}_{app}_tr.json                 - Original (always clean)
    {domain}_{app}_safe_filtered.json      - Safe entries + applied status
    {domain}_{app}_danger_filtered.json   - Danger entries + applied status
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
    "Variant": "Variant",
    "ExportTemplate": "ExportTemplate",
    "Solution": "Solution",
}

ALWAYS_SKIP_ALIASES = {
    # Add aliases to skip during filter phase
    # Examples: "test", "demo", "admin"
    "cmw_account_department",
    "cmw_account_fullName",
    "cmw_account_title",
    "cmw_account_mbox",
    "cmw_account_skype",
    "cmw_account_username",
    "cmw_account_active",
    "cmw_account_phone",
    "cmw_account_manager",
    "cmw_account_office",
    "cmw_account_lastLoginDate",
    "cmw_ou_type",
    "cmw_ou_name",
    "cmw_ou_description",
    "cmw_process_status",
    "cmw_process_currentLifetimeStatus",
    "cmw_process_name",
    "cmw_process_minDueDate",
    "cmw_process_businessObject",
    "cmw_process_hasTokenError",
    "cmw_process_activeVersion",
    "cmw_process_currentActivity",
    "cmw_task_planEndDate",
    "cmw_task_objectId",
    "cmw_task_displayId",
    "cmw_task_owner",
    "cmw_task_name",
    "cmw_role_name",
    "cmw_role_description",
    "defaultCard",
    "defaultList",
    "defaultForm",
    "defaultFormToolbar",
    "defaultListToolbar",
    "creator",
    "superiorUnit",
    "subordinateUnit",
    "alias",
    "creationDate",
    "isDisabled",
    "lastWriteDate",
    "unitForm",
    "OrgStructureTemplate",
    "defaultModelToolbar",
    "isReassignProhibited",
    "possibleAssignee",
    "description",
    "title",
    "defaultDiagramToolbar",
    "defaultProcessToolbar",
    "defaultTaskToolbar",
    "archiveProcess",
    "cancelProcess",
    "completeTask",
    "createProcess",
    "createToken",
    "createTokenForm",
    "editDiagram",
    "editTask",
    "migrate",
    "Migration",
    "reassignTask",
    "TaskReasignForm",
    "retryTokens",
    "dueDate",
    "assignee",
    "include",
    "create",
    "deleteAccount",
    "edit",
    "excludeAccount",
    "deleteRole",
    "endEdit",
    "startEdit",
    "superiorRole",
    "subordinateRole",
    "roleForm",
    "RoleTemplate",
    "deleteRole",
    "processes",
    "archive",
    "delete",
    "color",
    "id",
    "conversation",
    "completionDate",
    "cmw.account.language",
}


def get_domain(domain: str | None = None) -> str:
    if domain:
        return domain

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


def deduplicate_rows(rows: list) -> list:
    """Deduplicate rows by (type, aliasOriginal). Returns list with unique pairs."""
    unique_map = {}

    for row in rows:
        ids = row.get("ids", [])
        if not ids:
            continue

        original = row.get("aliasOriginal", "")
        new_name = row.get("aliasRenamed", "")
        if not original or not new_name:
            continue

        key = (row.get("type"), original)

        if key not in unique_map:
            unique_map[key] = {
                "type": row.get("type"),
                "aliasOriginal": original,
                "aliasRenamed": new_name,
                "ids": list(ids),
                "expressions": row.get("expressions"),
            }
        else:
            existing_ids = set(unique_map[key]["ids"])
            existing_ids.update(ids)
            unique_map[key]["ids"] = list(existing_ids)

    return list(unique_map.values())


def split_by_mode(rows: list) -> tuple[list, list]:
    """Split rows into safe (no _calc) and danger (has _calc or Solution type)."""
    safe = []
    danger = []

    for row in rows:
        alias_renamed = row.get("aliasRenamed", "")
        row_type = row.get("type", "")

        if row_type == "Solution" or alias_renamed.endswith("_calc"):
            danger.append(row)
        else:
            safe.append(row)

    return safe, danger


def phase_filter(
    app: str,
    output_dir: str,
    domain: str | None = None,
    rebuild: bool = False,
):
    """Phase 1: Filter and deduplicate data into separate safe/danger files."""
    output_path = Path(output_dir)
    domain = get_domain(domain)

    tr_file = output_path / f"{domain}_{app}_tr.json"
    if not tr_file.exists():
        print(f"No {domain}_{app}_tr.json found in {output_dir}")
        return

    with open(tr_file, encoding="utf-8") as f:
        table = json.load(f)

    print(f"Loaded {len(table)} entries from {tr_file.name}")

    safe_file = output_path / f"{domain}_{app}_safe_filtered.json"
    danger_file = output_path / f"{domain}_{app}_danger_filtered.json"
    skipped_file = output_path / f"{domain}_{app}_skipped_filtered.json"

    if safe_file.exists() and danger_file.exists() and skipped_file.exists() and not rebuild:
        print("Filtered files already exist. Use --rebuild to overwrite.")
        print(f"  - {safe_file.name}")
        print(f"  - {danger_file.name}")
        print(f"  - {skipped_file.name}")
        return

    filtered_rows = []
    skipped_rows = []
    for row in table:
        if row.get("aliasOriginal") in ALWAYS_SKIP_ALIASES:
            if row.get("aliasRenamed"):
                skipped_rows.append(row)
            continue
        if row.get("aliasLocked") == True:
            continue
        if not row.get("aliasRenamed"):
            continue
        if row.get("aliasRenamed") == row.get("aliasOriginal"):
            continue
        if not row.get("ids"):
            continue
        filtered_rows.append(row)

    print(f"After basic filters: {len(filtered_rows)} entries")
    print(f"Skipped by ALWAYS_SKIP_ALIASES: {len(skipped_rows)} entries")

    safe_rows, danger_rows = split_by_mode(filtered_rows)
    print(f"Safe (no _calc): {len(safe_rows)}")
    print(f"Danger (has _calc): {len(danger_rows)}")

    safe_dedup = deduplicate_rows(safe_rows)
    danger_dedup = deduplicate_rows(danger_rows)
    skipped_dedup = deduplicate_rows(skipped_rows)
    print(f"Safe after dedup: {len(safe_dedup)} unique (type, alias) pairs")
    print(f"Danger after dedup: {len(danger_dedup)} unique (type, alias) pairs")
    print(f"Skipped after dedup: {len(skipped_dedup)} unique (type, alias) pairs")

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    safe_output = {
        "_meta": {
            "generated_at": timestamp,
            "total_count": len(safe_dedup),
            "applied_count": 0,
        },
        "entries": safe_dedup,
    }

    danger_output = {
        "_meta": {
            "generated_at": timestamp,
            "total_count": len(danger_dedup),
            "applied_count": 0,
        },
        "entries": danger_dedup,
    }

    skipped_output = {
        "_meta": {
            "generated_at": timestamp,
            "total_count": len(skipped_dedup),
            "applied_count": 0,
        },
        "entries": skipped_dedup,
    }

    with open(safe_file, "w", encoding="utf-8") as f:
        json.dump(safe_output, f, indent=2, ensure_ascii=False)
    print(f"Saved: {safe_file.name}")

    with open(danger_file, "w", encoding="utf-8") as f:
        json.dump(danger_output, f, indent=2, ensure_ascii=False)
    print(f"Saved: {danger_file.name}")

    with open(skipped_file, "w", encoding="utf-8") as f:
        json.dump(skipped_output, f, indent=2, ensure_ascii=False)
    print(f"Saved: {skipped_file.name}")

    print("\n=== Phase 1 Complete ===")
    print(f"Safe: {len(safe_dedup)} unique pairs")
    print(f"Danger: {len(danger_dedup)} unique pairs")
    print(f"Skipped: {len(skipped_dedup)} unique pairs")


def phase_apply(
    app: str,
    output_dir: str,
    mode: str,
    reverse: bool = False,
    resume: bool = True,
    force: bool = False,
    domain: str | None = None,
):
    """Phase 2: Apply or reverse renames from filtered file."""
    output_path = Path(output_dir)
    domain = get_domain(domain)

    safe_file = output_path / f"{domain}_{app}_safe_filtered.json"
    danger_file = output_path / f"{domain}_{app}_danger_filtered.json"
    skipped_file = output_path / f"{domain}_{app}_skipped_filtered.json"

    if mode == "safe":
        filtered_file = safe_file
    elif mode == "danger":
        filtered_file = danger_file
    else:
        filtered_file = skipped_file

    if not filtered_file.exists():
        print(f"No {filtered_file.name} found. Run phase 1 first:")
        print(f"  python apply_renames.py --app {app} --phase filter --output-dir {output_dir}")
        return

    with open(filtered_file, encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", [])
    print(f"Loaded {len(entries)} entries from {filtered_file.name}")

    already_applied = sum(1 for e in entries if e.get("applied") == True)
    if already_applied > 0 and resume and not force:
        print(f"Found {already_applied} already applied entries (use --force to reprocess)")

    skip_applied = resume and not force

    entries_to_process = []
    for entry in entries:
        if skip_applied and entry.get("applied") == True:
            continue
        entries_to_process.append(entry)

    print(f"After resume filter: {len(entries_to_process)} entries")

    if mode == "danger" and not force:
        print("\nWARNING: Running in DANGER mode!")
        print("This will process aliases with expressions, which may have dependencies.")
        try:
            confirm = input("Are you sure you want to continue? (yes/no): ").strip().lower()
            if confirm not in ("yes", "y"):
                print("Aborted by user.")
                return
        except EOFError:
            print("Interactive input not available, proceeding...")

    success = 0
    failed = 0
    skipped = 0
    processed_ids = 0
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    save_interval = 50

    for entry_idx, entry in enumerate(entries_to_process):
        obj_type = entry.get("type", "")
        original = entry.get("aliasOriginal", "")
        new_name = entry.get("aliasRenamed", "")
        ids = entry.get("ids", [])

        plat_type = type_map.get(obj_type)
        if not plat_type:
            skipped += 1
            continue

        target_name = original if reverse else new_name

        all_success = True
        for obj_id in ids:
            result = update_object_property.invoke(
                {"object_id": obj_id, "object_type": plat_type, "new_value": target_name}
            )

            if result.get("success"):
                success += 1
            else:
                failed += 1
                all_success = False
            processed_ids += 1

        if all_success:
            entry["applied"] = True
            entry["appliedAt"] = timestamp
        else:
            entry["applied"] = False
            entry["failed"] = True

        time.sleep(0.02)

        if (entry_idx + 1) % save_interval == 0:
            with open(filtered_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            applied_now = sum(1 for e in entries if e.get("applied") == True)
            print(f"[Auto-saved] Progress: {entry_idx + 1}/{len(entries_to_process)} entries ({applied_now} applied)")

    with open(filtered_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    applied_count = sum(1 for e in entries if e.get("applied") == True)
    data["_meta"]["applied_count"] = applied_count

    with open(filtered_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n=== Phase 2 Complete ===")
    print(f"Unique entries processed: {len(entries_to_process)}")
    print(f"Total IDs processed: {processed_ids}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    print(f"Skipped (no type map): {skipped}")
    print(f"Applied: {applied_count}/{len(entries)}")
    print(f"Updated {filtered_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and apply localization renames")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument(
        "--phase",
        choices=["filter", "apply"],
        required=True,
        help="Phase: 'filter' - create filtered files, 'apply' - apply renames",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory with localization files",
    )
    parser.add_argument(
        "--mode",
        choices=["safe", "danger", "skipped"],
        default="safe",
        help="Mode for apply phase: 'safe', 'danger', or 'skipped' (default: safe)",
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
        "--rebuild",
        action="store_true",
        default=False,
        help="Rebuild filtered files (phase 1 only)",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Domain (e.g., mz-fr.test.cbap.ru)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or f"/tmp/cmw-transfer/{args.app}_tr"

    if args.phase == "filter":
        phase_filter(args.app, output_dir, args.domain, args.rebuild)
    else:
        phase_apply(args.app, output_dir, args.mode, args.reverse, args.resume, args.force, args.domain)