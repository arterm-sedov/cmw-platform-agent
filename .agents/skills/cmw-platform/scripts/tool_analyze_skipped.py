#!/usr/bin/env python3
"""
Analyze and save skipped aliases from CTF that were not found in platform.

Usage:
    python analyze_skipped.py --app Volga
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(APP_DIR))

from tools.applications_tools.tool_get_ontology_objects import get_ontology_objects


TYPE_FOLDER_MAP = {
    "RecordTemplate": "RecordTemplates",
    "ProcessTemplate": "ProcessTemplates",
    "Workspace": "Workspaces",
    "Page": "Pages",
    "RoleTemplate": "Roles",
    "Role": "Roles",
    "AccountTemplate": "Accounts",
    "OrgStructureTemplate": "OrgStructure",
    "MessageTemplate": "MessageTemplates",
    "Stream": "Streams",
    "Route": "Routes",
    "Trigger": "Triggers",
    "Attribute": "Attributes",
    "Dataset": "Datasets",
    "Form": "Forms",
    "Toolbar": "Toolbars",
    "UserCommand": "UserCommands",
    "Card": "Cards",
    "WidgetConfig": "WidgetConfigs",
}

FOLDER_TYPE_MAP = {v: k for k, v in TYPE_FOLDER_MAP.items()}
SUB_TYPES = {"Attributes", "Datasets", "Forms", "Toolbars", "UserCommands"}

SYSTEM_ALIASES = {
    "create", "edit", "delete", "archive", "deleteRole",
    "defaultCard", "defaultList", "defaultFormToolbar", "defaultListToolbar",
    "defaultModelToolbar", "defaultTaskToolbar", "defaultProcessToolbar",
    "defaultDiagramToolbar",
    "startEdit", "endEdit",
    "RoleTemplate", "OrgStructureTemplate", "RecordTemplate", "ProcessTemplate",
    "systemPage_architect",
    "editTask", "migrate", "editDiagram", "cancelProcess", "reassignTask",
    "completeTask", "archiveProcess", "createProcess", "retryTokens", "createToken",
    "noneStartEvent1", "executionFlow1", "pool1", "noneEndEvent1",
    "defaultForm",
    "roleForm", "Test_OrganizationalStructure",
}

SYSTEM_PREFIXES = (
    "cmw.", "oa.", "pa.", "msgt.", "aa.", "ra.", "os.",
    "form.", "tb.", "lst.", "event.", "card.", "trigger.", "workspace.",
    "role.", "fw.",
)


def collect_all_aliases(base: Path, app_name: str) -> list[dict]:
    """Collect all aliases from CTF JSON - including skipped ones."""
    app_dir = base / app_name
    if not app_dir.exists():
        print(f"Error: {app_dir} not found")
        return []

    objects = []

    def scan_folder(folder_path: Path, relative_prefix: str = ""):
        if not folder_path.exists():
            return

        for item in folder_path.iterdir():
            if item.is_dir():
                sub_prefix = f"{relative_prefix}/{item.name}" if relative_prefix else item.name
                scan_folder(item, sub_prefix)
            elif item.suffix == ".json":
                scan_json_file(item, relative_prefix)

    def scan_json_file(json_path: Path, folder_prefix: str):
        try:
            data = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            return

        global_alias = data.get("GlobalAlias", {})
        alias = global_alias.get("Alias")
        if not alias:
            return

        if alias in SYSTEM_ALIASES:
            return
        if alias.startswith(SYSTEM_PREFIXES):
            return

        obj_type = _infer_type(json_path, folder_prefix, global_alias)
        display_name = data.get("Name", "")

        path_in_ctf = f"{folder_prefix}/{json_path.name}" if folder_prefix else json_path.name

        objects.append({
            "type": obj_type,
            "aliasOriginal": alias,
            "displayNameOriginal": display_name,
            "jsonPathOriginal": path_in_ctf,
        })

    def _infer_type(json_path: Path, folder_prefix: str, global_alias: dict) -> str:
        parts = Path(folder_prefix).parts if folder_prefix else []
        if parts:
            immediate_parent = parts[-1] if parts else ""
            if immediate_parent in SUB_TYPES:
                return immediate_parent[:-1] if immediate_parent.endswith("s") else immediate_parent
            first_folder = parts[0]
            if first_folder in FOLDER_TYPE_MAP:
                return FOLDER_TYPE_MAP[first_folder]
            if first_folder in SUB_TYPES:
                return first_folder[:-1] if first_folder.endswith("s") else first_folder

        if global_alias.get("$type") == "cmw.container":
            return "RecordTemplate"
        if global_alias.get("$type") == "cmw.process":
            return "ProcessTemplate"

        return "Unknown"

    scan_folder(app_dir)
    return objects


def verify_and_split(objects: list[dict], app_name: str) -> tuple[list[dict], list[dict]]:
    """Verify aliases against platform, return (verified, skipped)."""
    all_types = list(set(o["type"] for o in objects if o["type"] != "Unknown"))

    verified = {}
    for obj_type in all_types:
        print(f"  Verifying {obj_type}...")
        result = get_ontology_objects.invoke({
            "application_system_name": app_name,
            "types": [obj_type],
            "parameter": "alias",
            "min_count": 1,
            "max_count": 5000,
        })

        if result["success"] and result.get("data"):
            for obj in result["data"]:
                verified[obj["systemName"]] = obj["id"]

        time.sleep(0.3)

    verified_objects = []
    skipped_objects = []

    for obj in objects:
        alias = obj["aliasOriginal"]
        if alias in verified:
            obj["id"] = verified[alias]
            verified_objects.append(obj)
        else:
            skipped_objects.append(obj)

    return verified_objects, skipped_objects


def main():
    parser = argparse.ArgumentParser(description="Analyze skipped aliases from CTF")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument(
        "--extract-dir",
        default="/tmp/cmw-transfer/Volga-extract",
        help="CTF extraction directory"
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/cmw-transfer/Volga-extract/Volga_tr",
        help="Output directory for JSON files"
    )

    args = parser.parse_args()

    extract_dir = Path(args.extract_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Analyzing skipped aliases for: {args.app} ===")

    # Step 1: Collect all aliases from CTF (including skipped)
    print(f"\n[1] Collecting all aliases from CTF...")
    all_objects = collect_all_aliases(extract_dir, args.app)
    print(f"  Total CTF objects: {len(all_objects)}")

    # Step 2: Verify against platform and split
    print(f"\n[2] Verifying against platform...")
    verified, skipped = verify_and_split(all_objects, args.app)
    print(f"  Verified: {len(verified)}")
    print(f"  Skipped: {len(skipped)}")

    # Step 3: Save verified objects
    verified_file = output_dir / f"{args.app}_verified.json"
    with open(verified_file, "w", encoding="utf-8") as f:
        json.dump(verified, f, indent=2, ensure_ascii=False)
    print(f"\n[3] Saved verified: {verified_file}")

    # Step 4: Save skipped objects
    skipped_file = output_dir / f"{args.app}_skipped.json"
    with open(skipped_file, "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2, ensure_ascii=False)
    print(f"  Saved skipped: {skipped_file}")

    # Step 5: Summary by type for skipped
    print(f"\n=== Skipped Objects Summary by Type ===")
    from collections import Counter
    type_counts = Counter(o["type"] for o in skipped)
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")

    print(f"\n=== Complete ===")
    print(f"Verified: {len(verified)} -> {verified_file}")
    print(f"Skipped: {len(skipped)} -> {skipped_file}")


if __name__ == "__main__":
    main()