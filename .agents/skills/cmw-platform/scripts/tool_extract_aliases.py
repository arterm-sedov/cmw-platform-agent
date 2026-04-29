#!/usr/bin/env python3
"""
Step 1: Extract Aliases Per Folder

Extracts aliases from CTF JSON files, folder by folder.
Can resume from interruption - skips completed folders.

Usage:
    python tool_extract_aliases.py --app Volga --extract-dir /path/to/extract --output-dir /path/to/output
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

APP_DIR = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(APP_DIR))

from tools.applications_tools.tool_get_ontology_objects import get_ontology_objects

SKIP_TYPES = {
    "ClientActivity",
    "Image",
    "ReferenceToConnection",
}

SKIP_TYPE_PREFIXES = {
    "Component_": ["DesktopComponent"],
    "WidgetConfig_": ["DesktopComponent", "DesktopWidgetConfig"],
    "form_": ["ProcessForm"],
    "form_userTask": ["ProcessForm"],
    "Page_": ["SimplePage"],
    "VerticalLayout": ["DesktopComponent"],
    "HorizontalLayout": ["DesktopComponent"],
    "myTaskList": ["DesktopComponent"],
    "myTasksList": ["DesktopComponent"],
    "_ReassignTask": ["MessageTemplate"],
    "OrgStructureTemplate": ["OrgStructureTemplate"],
    "_Account": ["RecordTemplate"],
    "RoleTemplate": ["RoleTemplate"],
    "systemPage_": ["SimplePage"],
    "workspace": ["RoleWorkspace"],
    "defaultList": ["Dataset"],
    "list": ["Dataset"],
    "unitForm": ["Form"],
    "roleForm": ["Form"],
    "accountForm": ["Form"],
    "defaultForm": ["Form"],
    "defaultFormToolbar": ["Toolbar"],
    "defaultListToolbar": ["Toolbar"],
    "defaultModelToolbar": ["Toolbar"],
    "defaultTaskToolbar": ["Toolbar"],
    "defaultProcessToolbar": ["Toolbar"],
    "defaultDiagramToolbar": ["Toolbar"],
    "ToolbarComponent": ["Toolbar"],
    "newToolbar": ["Toolbar"],
    "Token": ["Trigger"],
    "deleteRole": ["UserCommand"],
    "create": ["UserCommand"],
    "edit": ["UserCommand"],
    "deleteAccount": ["UserCommand"],
    "includeAccount": ["UserCommand"],
    "excludeAccount": ["UserCommand"],
    "archive": ["UserCommand"],
    "delete": ["UserCommand"],
    "startEdit": ["UserCommand"],
    "endEdit": ["UserCommand"],
    "editTask": ["UserCommand"],
    "migrate": ["UserCommand"],
    "editDiagram": ["UserCommand"],
    "cancelProcess": ["UserCommand"],
    "reassignTask": ["UserCommand"],
    "completeTask": ["UserCommand"],
    "archiveProcess": ["UserCommand"],
    "createProcess": ["UserCommand"],
    "retryTokens": ["UserCommand"],
    "createToken": ["UserCommand"],
}

SKIP_ATTRIBUTES = {
    "AccountTemplate": {"active", "departament", "fullName", "language", "lastLoginDate", "manager", "mbox", "office", "phone", "skype", "timeZone", "title", "username"},
    "OrgStructureTemplate": {"alias", "unitName", "subordinateUnit", "superiorUnit", "unitDescription", "unitType"},
    "RoleTemplate": {"alias", "roleDescription", "roleIsActive", "roleName", "subordinateRole", "superiorRole"},
    "ModelTemplate": {"description", "name", "order", "parentProcessModelNone", "parentSubprocessEmbedded"},
    "default": {"_color", "_conversation", "_creationDate", "_creator", "_isDisabled", "_lastWriteDate", "_processes", "id"},
}

UUID_32_PATTERN = "^[0-9a-f]{32}$"
FORM_INT_PATTERN = r"^form\d+$"
TOOLBAR_UUID_PATTERN = r"^toolbar[0-9a-f]{32}$"


def should_skip_alias(alias: str, obj_type: str, displayName: str = "", parent_type: str = None) -> bool | str:
    """Check if alias should be skipped based on filtering rules."""
    import re

    if obj_type in SKIP_TYPES:
        return True

    if obj_type == "FormComponent":
        if re.match(UUID_32_PATTERN, alias, re.I):
            return True

    if obj_type == "Form":
        if re.match(FORM_INT_PATTERN, alias, re.I):
            return True

    if obj_type == "Toolbar":
        if re.match(TOOLBAR_UUID_PATTERN, alias, re.I):
            return True

    if obj_type == "Attribute":
        global_skip = SKIP_ATTRIBUTES["default"]
        parent_skip = SKIP_ATTRIBUTES.get(parent_type, set()) if parent_type else set()
        if alias in global_skip or alias in parent_skip:
            return True

    for prefix, types in SKIP_TYPE_PREFIXES.items():
        if obj_type in types and alias.startswith(prefix):
            if displayName:
                return "locked"
            return True

    return False


def scan_json_recursive(obj, path="root", parent_type=None, results=None):
    """Recursively find ALL aliases at ALL levels."""
    if results is None:
        results = []

    if isinstance(obj, dict):
        display_name = obj.get("Name", "")

        if "GlobalAlias" in obj and isinstance(obj["GlobalAlias"], dict):
            ga = obj["GlobalAlias"]
            if "Alias" in ga and ga["Alias"]:
                alias = ga["Alias"]
                obj_type = ga.get("Type", "Unknown")
                skip_result = should_skip_alias(alias, obj_type, display_name)

                if skip_result == "locked":
                    results.append({
                        "alias": alias,
                        "type": obj_type,
                        "displayName": display_name,
                        "aliasLocked": True,
                        "path": path,
                        "source": "GlobalAlias",
                        "parent_type": parent_type,
                    })
                elif not skip_result:
                    results.append({
                        "alias": alias,
                        "type": obj_type,
                        "displayName": display_name,
                        "aliasLocked": False,
                        "path": path,
                        "source": "GlobalAlias",
                        "parent_type": parent_type,
                    })

        if "Container" in obj and isinstance(obj["Container"], dict):
            container = obj["Container"]
            if "Alias" in container and container["Alias"]:
                alias = container["Alias"]
                obj_type = container.get("Type", "Unknown")
                if not should_skip_alias(alias, obj_type):
                    results.append({
                        "alias": alias,
                        "type": obj_type,
                        "displayName": "",
                        "aliasLocked": False,
                        "path": f"{path}/Container",
                        "source": "Container",
                        "parent_type": parent_type,
                    })

        if "Template" in obj and isinstance(obj["Template"], dict):
            template = obj["Template"]
            if "Alias" in template and template["Alias"]:
                alias = template["Alias"]
                obj_type = template.get("Type", "Unknown")
                if not should_skip_alias(alias, obj_type):
                    results.append({
                        "alias": alias,
                        "type": obj_type,
                        "displayName": "",
                        "aliasLocked": False,
                        "path": f"{path}/Template",
                        "source": "Template",
                        "parent_type": parent_type,
                    })

        if "InstanceGlobalAlias" in obj and isinstance(obj["InstanceGlobalAlias"], dict):
            inst = obj["InstanceGlobalAlias"]
            if "Alias" in inst and inst["Alias"]:
                alias = inst["Alias"]
                obj_type = inst.get("Type", "Unknown")
                if not should_skip_alias(alias, obj_type, "", parent_type):
                    results.append({
                        "alias": alias,
                        "type": obj_type,
                        "displayName": "",
                        "aliasLocked": False,
                        "path": f"{path}/InstanceGlobalAlias",
                        "source": "InstanceGlobalAlias",
                        "parent_type": parent_type,
                    })

        if "Root" in obj and isinstance(obj["Root"], dict):
            container_type = None
            if "Container" in obj and isinstance(obj["Container"], dict):
                container_type = obj["Container"].get("Type")
            scan_json_recursive(obj["Root"], f"{path}/Root", container_type, results)

        for array_key in ["Children", "Childrens"]:
            if array_key in obj and isinstance(obj[array_key], list):
                for i, child in enumerate(obj[array_key]):
                    scan_json_recursive(child, f"{path}/{array_key}[{i}]", parent_type, results)

        if "Rows" in obj and isinstance(obj["Rows"], dict):
            rows_data = obj["Rows"]
            if "$values" in rows_data and isinstance(rows_data["$values"], list):
                for i, row in enumerate(rows_data["$values"]):
                    scan_json_recursive(row, f"{path}/Rows.$values[{i}]", parent_type, results)

        exclude_keys = {"GlobalAlias", "Container", "Template", "InstanceGlobalAlias", "Root", "Children", "Childrens", "Rows"}
        for key, value in obj.items():
            if key not in exclude_keys:
                scan_json_recursive(value, f"{path}/{key}", parent_type, results)

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            scan_json_recursive(item, f"{path}[{i}]", parent_type, results)

    return results


def get_folders_to_process(app_dir: Path) -> list[str]:
    """Auto-detect folders from CTF structure."""
    folders = []
    if not app_dir.exists():
        return folders

    for item in app_dir.iterdir():
        if item.is_dir():
            folders.append(item.name)

    return sorted(folders)


def process_folder(folder_name: str, app_dir: Path) -> tuple[list, int]:
    """Extract aliases from a single folder."""
    folder_path = app_dir / folder_name
    if not folder_path.exists():
        return [], 0

    all_aliases = []
    file_count = 0

    def extract_parent_template(json_path: str) -> str:
        """Extract parent template name from JSON path.
        Example: 'RecordTemplates/Schetchiki/Attributes/Year.json' -> 'Schetchiki'
        """
        parts = json_path.split("/")
        if len(parts) >= 2:
            return parts[1]
        return ""

    def scan_folder(folder: Path):
        nonlocal file_count
        for item in folder.iterdir():
            if item.is_dir():
                scan_folder(item)
            elif item.suffix == ".json":
                file_count += 1
                try:
                    data = json.loads(item.read_text(encoding="utf-8"))
                    relative_path = str(item.relative_to(app_dir))
                    parent_template = extract_parent_template(relative_path)
                    aliases = scan_json_recursive(data, relative_path)
                    for a in aliases:
                        a["json_file"] = relative_path
                        a["parent_template"] = parent_template
                    all_aliases.extend(aliases)
                except (json.JSONDecodeError, OSError):
                    pass

    scan_folder(folder_path)

    # Deduplicate by (parent_template, type, alias)
    deduped = {}
    for a in all_aliases:
        key = (a.get("parent_template", ""), a.get("type"), a.get("alias"))
        if key not in deduped:
            deduped[key] = {
                **a,
                "jsonPathOriginal": [a.get("path", "")],
            }
        else:
            deduped[key]["jsonPathOriginal"].append(a.get("path", ""))
            if a.get("aliasLocked"):
                deduped[key]["aliasLocked"] = True

    return list(deduped.values()), file_count


def main(app: str = None, extract_dir: str = None, output_dir: str = None):
    """Main function - can be called with parameters or CLI args."""
    if app is None:
        parser = argparse.ArgumentParser(description="Step 1: Extract aliases per folder")
        parser.add_argument("--app", required=True)
        parser.add_argument("--extract-dir", default="/tmp/cmw-transfer/Volga-extract")
        parser.add_argument("--output-dir", default="/tmp/cmw-transfer/Volga-extract/Volga_tr")
        args = parser.parse_args()
        app = args.app
        extract_dir = Path(args.extract_dir)
        output_dir = Path(args.output_dir)
    else:
        extract_dir = Path(extract_dir)
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Step 1: Extract Aliases for {app} ===")
    print(f"Extract dir: {extract_dir}")
    print(f"Output dir: {output_dir}")

    app_dir = extract_dir / app
    if not app_dir.exists():
        print(f"Error: {app_dir} not found")
        return 1

    state_file = output_dir / f"{app}_extraction_state.json"
    state = {"app": app, "completed_folders": [], "pending_folders": [], "last_updated": ""}

    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    all_folders = get_folders_to_process(app_dir)
    print(f"Found folders: {all_folders}")

    completed = set(state.get("completed_folders", []))
    pending = [f for f in all_folders if f not in completed]

    if not pending:
        print("All folders already completed. Use --force to re-extract.")
        return 0

    print(f"Pending folders ({len(pending)}): {pending}")

    total_aliases = 0
    total_files = 0

    for i, folder in enumerate(pending, 1):
        output_file = output_dir / f"{app}_{folder}_aliases.json"

        if output_file.exists():
            try:
                with open(output_file) as f:
                    existing = json.load(f)
                if existing.get("folder") == folder and existing.get("aliases"):
                    print(f"[{i}/{len(pending)}] {folder}: Already extracted ({len(existing['aliases'])} aliases)")
                    completed.add(folder)
                    state["completed_folders"] = list(completed)
                    state["pending_folders"] = [f for f in pending if f not in completed]
                    state["last_updated"] = datetime.now().isoformat()
                    continue
            except (json.JSONDecodeError, OSError):
                pass

        print(f"[{i}/{len(pending)}] Processing {folder}...", end=" ", flush=True)
        start = time.time()

        aliases, file_count = process_folder(folder, app_dir)

        output_data = {
            "app": app,
            "folder": folder,
            "extracted_at": datetime.now().isoformat(),
            "count": len(aliases),
            "file_count": file_count,
            "aliases": aliases,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start
        print(f"{len(aliases)} aliases from {file_count} files ({elapsed:.1f}s)")

        completed.add(folder)
        total_aliases += len(aliases)
        total_files += file_count

        state["completed_folders"] = list(completed)
        state["pending_folders"] = [f for f in pending if f not in completed]
        state["last_updated"] = datetime.now().isoformat()

        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    print(f"\n=== Extraction Complete ===")
    print(f"Total aliases: {total_aliases}")
    print(f"Total files scanned: {total_files}")
    print(f"Completed folders: {list(completed)}")
    print(f"State saved to: {state_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
