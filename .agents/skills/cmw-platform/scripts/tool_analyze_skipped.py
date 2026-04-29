#!/usr/bin/env python3
"""
Complete alias extraction for localization.
Finds ALL aliases at ALL nesting levels from CTF JSON export.

Usage:
    python tool_analyze_skipped.py --app Volga
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

APP_DIR = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(APP_DIR))

from tools.applications_tools.tool_get_ontology_objects import get_ontology_objects


SYSTEM_ALIASES = {
    "create", "edit", "delete", "archive", "deleteRole",
    "defaultCard", "defaultList", "defaultFormToolbar", "defaultListToolbar",
    "defaultModelToolbar", "defaultTaskToolbar", "defaultProcessToolbar",
    "defaultDiagramToolbar", "startEdit", "endEdit",
    "RoleTemplate", "OrgStructureTemplate", "RecordTemplate", "ProcessTemplate",
    "systemPage_architect", "editTask", "migrate", "editDiagram", "cancelProcess",
    "reassignTask", "completeTask", "archiveProcess", "createProcess", "retryTokens",
    "createToken", "noneStartEvent1", "executionFlow1", "pool1", "noneEndEvent1",
    "defaultForm", "roleForm", "Test_OrganizationalStructure",
}

SYSTEM_PREFIXES = (
    "cmw.", "oa.", "pa.", "msgt.", "aa.", "ra.", "os.",
    "form.", "tb.", "lst.", "event.", "card.", "trigger.", "workspace.",
    "role.", "fw.",
)

SKIP_TYPES = {
    "ClientActivity",
    "Image",
    "ReferenceToConnection",
}

SKIP_ATTRIBUTES = {
    "AccountTemplate": {"active", "departament", "fullName", "language", "lastLoginDate", "manager", "mbox", "office", "phone", "skype", "timeZone", "title", "username"},
    "OrgStructureTemplate": {"alias", "unitName", "subordinateUnit", "superiorUnit", "unitDescription", "unitType"},
    "RoleTemplate": {"alias", "roleDescription", "roleIsActive", "roleName", "subordinateRole", "superiorRole"},
    "ModelTemplate": {"description", "name", "order", "parentProcessModelNone", "parentSubprocessEmbedded"},
    "default": {"_color", "_conversation", "_creationDate", "_creator", "_isDisabled", "_lastWriteDate", "_processes", "id"},
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

UUID_32_PATTERN = "^[0-9a-f]{32}$"
FORM_INT_PATTERN = r"^form\d+$"
TOOLBAR_UUID_PATTERN = r"^toolbar[0-9a-f]{32}$"

EXPRESSION_KEYS = {"Expression", "Code", "ValueExpression", "ValidationScript", "Calculation", "DefaultExpression"}

TYPE_TO_PLATFORM = {
    "RoleWorkspace": "Workspace",
    "MessageTemplateProperty": "Attribute",
    "RoleConfiguration": "Role",
    "SimplePage": "Page",
}


def should_skip_alias(alias: str, obj_type: str, displayName: str = "", parent_type: str = None) -> bool | str:
    """
    Check if alias should be skipped based on filtering rules.
    Returns:
        False - don't skip
        True - skip (no display name, matches pattern)
        "locked" - don't skip, but lock alias (has display name, matches pattern)
    """
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
            return True

    return False


def scan_json_recursive(obj, path="root", parent_type=None, results=None):
    """Recursively find ALL aliases at ALL levels."""
    if results is None:
        results = []

    if isinstance(obj, dict):
        # Extract displayName for main object check
        display_name = obj.get("Name", "")

        # 1. GlobalAlias - main object
        if "GlobalAlias" in obj and isinstance(obj["GlobalAlias"], dict):
            ga = obj["GlobalAlias"]
            if "Alias" in ga and ga["Alias"]:
                alias = ga["Alias"]
                obj_type = ga.get("Type", "Unknown")
                skip_result = should_skip_alias(alias, obj_type, display_name)

                if skip_result == "locked":
                    # Include - pattern matched but has displayName, allow rename if dangerous
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
                    # Normal - don't skip
                    results.append({
                        "alias": alias,
                        "type": obj_type,
                        "displayName": display_name,
                        "aliasLocked": False,
                        "path": path,
                        "source": "GlobalAlias",
                        "parent_type": parent_type,
                    })
                # skip_result == True -> skip (don't add)

        # 2. Container - for Owner reference type
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

        # 3. Template - reference to template
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

        # 4. InstanceGlobalAlias - attribute reference
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

        # 5. Root - recurse into
        if "Root" in obj and isinstance(obj["Root"], dict):
            # Get container type for parent context
            container_type = None
            if "Container" in obj and isinstance(obj["Container"], dict):
                container_type = obj["Container"].get("Type")
            scan_json_recursive(obj["Root"], f"{path}/Root", container_type, results)

        # 6. Children/Childrens - array of nested objects
        for array_key in ["Children", "Childrens"]:
            if array_key in obj and isinstance(obj[array_key], list):
                for i, child in enumerate(obj[array_key]):
                    scan_json_recursive(child, f"{path}/{array_key}[{i}]", parent_type, results)

        # 7. Rows.$values - form layout rows
        if "Rows" in obj and isinstance(obj["Rows"], dict):
            rows_data = obj["Rows"]
            if "$values" in rows_data and isinstance(rows_data["$values"], list):
                for i, row in enumerate(rows_data["$values"]):
                    scan_json_recursive(row, f"{path}/Rows.$values[{i}]", parent_type, results)

        # 8. Recurse into all other keys
        exclude_keys = {"GlobalAlias", "Container", "Template", "InstanceGlobalAlias", "Root", "Children", "Childrens", "Rows"}
        for key, value in obj.items():
            if key not in exclude_keys:
                scan_json_recursive(value, f"{path}/{key}", parent_type, results)

    elif isinstance(obj, list):
        # Recurse into array elements
        for i, item in enumerate(obj):
            scan_json_recursive(item, f"{path}[{i}]", parent_type, results)

    return results


def extract_all_aliases_complete(base: Path, app_name: str) -> list[dict]:
    """Extract ALL aliases recursively from all JSON files."""
    app_dir = base / app_name
    if not app_dir.exists():
        print(f"Error: {app_dir} not found")
        return []

    all_aliases = []
    file_count = 0

    def scan_folder(folder_path: Path):
        nonlocal file_count
        if not folder_path.exists():
            return

        for item in folder_path.iterdir():
            if item.is_dir():
                scan_folder(item)
            elif item.suffix == ".json":
                file_count += 1
                try:
                    data = json.loads(item.read_text(encoding="utf-8"))
                    relative_path = str(item.relative_to(app_dir))
                    aliases = scan_json_recursive(data, relative_path)
                    for a in aliases:
                        a["json_file"] = relative_path
                    all_aliases.extend(aliases)
                except (json.JSONDecodeError, OSError):
                    pass

    print(f"  Scanning {app_name} directory...")
    scan_folder(app_dir)
    print(f"  Scanned {file_count} JSON files")
    print(f"  Found {len(all_aliases)} alias extractions")

    return all_aliases


def verify_aliases(all_aliases: list[dict], app_name: str) -> tuple[list, list]:
    """Verify all aliases against platform."""

    print(f"\n[2] Pre-collecting platform data...")

    # Get all types that exist (include mapped types)
    unique_types = set(a["type"] for a in all_aliases)
    types_to_query = set(unique_types)
    for t in unique_types:
        if t in TYPE_TO_PLATFORM:
            types_to_query.add(TYPE_TO_PLATFORM[t])
    platform_cache = {}

    for obj_type in sorted(types_to_query):
        result = get_ontology_objects.invoke({
            "application_system_name": app_name,
            "types": [obj_type],
            "parameter": "alias",
            "min_count": 1,
            "max_count": 5000,
        })
        if result.get("success") and result.get("data"):
            alias_to_ids = {}
            for obj in result["data"]:
                alias = obj["systemName"]
                if alias not in alias_to_ids:
                    alias_to_ids[alias] = []
                alias_to_ids[alias].append(obj["id"])
            platform_cache[obj_type] = alias_to_ids
            print(f"    {obj_type}: {len(alias_to_ids)}")
        else:
            platform_cache[obj_type] = {}
            print(f"    {obj_type}: 0")
        time.sleep(0.05)

    # Verify each unique (type, alias)
    print(f"\n[3] Verifying {len(all_aliases)} alias extractions...")

    # Deduplicate by (type, alias)
    seen = {}
    for a in all_aliases:
        key = (a["type"], a["alias"])
        if key not in seen:
            seen[key] = a

    unique_aliases = list(seen.values())
    print(f"  Unique (type, alias): {len(unique_aliases)}")

    verified = []
    skipped = []

    for i, obj in enumerate(unique_aliases):
        if (i + 1) % 1000 == 0:
            print(f"    Progress: {i + 1}/{len(unique_aliases)}")

        obj_type = obj["type"]
        alias = obj["alias"]

        platform_type = TYPE_TO_PLATFORM.get(obj_type, obj_type)

        found_ids = []
        if platform_type in platform_cache and alias in platform_cache[platform_type]:
            found_ids = platform_cache[platform_type][alias]

        # Special handling for OrgStructureTemplate and RoleTemplate
        # Try multiple prefix patterns
        if not found_ids and obj_type in ("OrgStructureTemplate", "RoleTemplate"):
            if obj_type == "RoleTemplate":
                candidates = [f"{app_name}_RolesCatalog", "systemSolution_RolesCatalog"]
            else:
                candidates = [f"{app_name}_OrganizationalStructure", "systemSolution_OrganizationalStructure"]
            for prefixed_alias in candidates:
                if platform_type in platform_cache and prefixed_alias in platform_cache[platform_type]:
                    found_ids = platform_cache[platform_type][prefixed_alias]
                    break

        if found_ids:
            obj["id"] = list(dict.fromkeys(found_ids))
            verified.append(obj)
        else:
            skipped.append(obj)

    return verified, skipped


def main():
    parser = argparse.ArgumentParser(description="Complete alias extraction for localization")
    parser.add_argument("--app", required=True)
    parser.add_argument("--extract-dir", default="/tmp/cmw-transfer/Volga-extract")
    parser.add_argument("--output-dir", default="/tmp/cmw-transfer/Volga-extract/Volga_tr")

    args = parser.parse_args()

    extract_dir = Path(args.extract_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Complete Alias Extraction for: {args.app} ===")

    # Extract ALL aliases recursively
    print(f"\n[1] Extracting ALL aliases (recursive)...")
    all_aliases = extract_all_aliases_complete(extract_dir, args.app)

    # Verify
    print(f"\n[2] Verifying against platform...")
    verified, skipped = verify_aliases(all_aliases, args.app)

    # Find dangerous aliases (used in expressions)
    verified_alias_set = {(v["type"], v["alias"]) for v in verified}
    dangerous = find_dangerous_aliases(extract_dir, args.app, verified_alias_set)

    # Update aliasLocked for verified objects
    # If aliasLocked=True AND alias is dangerous -> set aliasLocked=False (allow rename)
    dangerous_renamed = 0
    for obj in verified:
        if obj.get("aliasLocked") and obj["alias"] in dangerous:
            obj["aliasLocked"] = False
            dangerous_renamed += 1
    if dangerous_renamed:
        print(f"  {dangerous_renamed} locked aliases marked as dangerous, will be renamed")

    # Stats
    verified_types = Counter(d["type"] for d in verified)
    skipped_types = Counter(d["type"] for d in skipped)
    multi_id = len([d for d in verified if isinstance(d.get("id"), list) and len(d["id"]) > 1])
    total_ids = sum(len(d["id"]) if isinstance(d.get("id"), list) else 0 for d in verified)

    # Save verified
    verified_file = output_dir / f"{args.app}_verified_complete.json"
    with open(verified_file, "w", encoding="utf-8") as f:
        json.dump(verified, f, indent=2, ensure_ascii=False)

    # Save skipped (not in platform)
    skipped_file = output_dir / f"{args.app}_skipped_complete.json"
    with open(skipped_file, "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2, ensure_ascii=False)

    # Save by type folders
    verified_dir = output_dir / "verified_by_type"
    verified_dir.mkdir(exist_ok=True)
    by_type_v = defaultdict(list)
    for obj in verified:
        by_type_v[obj["type"]].append(obj)
    for type_name, objects in by_type_v.items():
        with open(verified_dir / f"{type_name}.json", "w", encoding="utf-8") as f:
            json.dump(objects, f, indent=2, ensure_ascii=False)

    skipped_dir = output_dir / "skipped_by_type"
    skipped_dir.mkdir(exist_ok=True)
    by_type_s = defaultdict(list)
    for obj in skipped:
        by_type_s[obj["type"]].append(obj)
    for type_name, objects in by_type_s.items():
        with open(skipped_dir / f"{type_name}.json", "w", encoding="utf-8") as f:
            json.dump(objects, f, indent=2, ensure_ascii=False)

    print(f"\n=== Summary ===")
    print(f"Total alias extractions: {len(all_aliases)}")
    print(f"Unique (type, alias): {len(list({(d['type'], d['alias']): d for d in all_aliases}.values()))}")
    print(f"Verified: {len(verified)}")
    print(f"Skipped (not in platform): {len(skipped)}")
    print(f"Objects with multiple IDs: {multi_id}")
    print(f"Total platform IDs: {total_ids}")

    print(f"\n=== Verified by Type ===")
    for t, c in sorted(verified_types.items()):
        print(f"  {t}: {c}")

    print(f"\n=== Skipped by Type ===")
    for t, c in sorted(skipped_types.items()):
        print(f"  {t}: {c}")

    print(f"\n=== Complete ===")
    print(f"Verified: {len(verified)} -> {verified_file}")
    print(f"Skipped: {len(skipped)} -> {skipped_file}")


def find_dangerous_aliases(extract_dir: Path, app_name: str, verified_aliases: set[tuple[str, str]]) -> set[str]:
    """Scan ALL JSON files at ALL nesting levels for aliases used in expression contexts.

    Optimization: First find files with expressions, then only scan those.
    """
    import re
    from concurrent.futures import ThreadPoolExecutor, as_completed

    base = extract_dir / app_name
    dangerous = set()

    # Build regex patterns for each verified alias
    alias_patterns = {}
    for obj_type, alias in verified_aliases:
        escaped = re.escape(alias)
        p1 = "${" + escaped + "}"    # ${alias}
        p2 = "->{" + escaped + "}"   # ->{alias}
        p3 = "{" + escaped + "}->"   # {alias}->
        p4 = '"{"' + escaped + '}"'  # "{alias}"
        alias_patterns[alias] = [p1, p2, p3, p4]

    print(f"\n[3] Scanning ALL JSON files for aliases in expressions...")

    # Phase 1: Find files with expression content (fast scan)
    files_with_expressions = []
    for json_file in base.rglob("*.json"):
        try:
            content = json_file.read_text(encoding="utf-8")
            if any(kw in content for kw in EXPRESSION_KEYS):
                files_with_expressions.append(json_file)
        except (OSError, UnicodeDecodeError):
            continue

    print(f"  Found {len(files_with_expressions)} files with expression content")

    # Phase 2: Scan only files with expressions
    match_count = 0

    def scan_file(json_file):
        local_dangerous = set()
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return local_dangerous

        def scan_expressions(obj):
            nonlocal local_dangerous, match_count
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in EXPRESSION_KEYS and isinstance(value, str):
                        for alias, patterns in alias_patterns.items():
                            for pattern in patterns:
                                if re.search(pattern, value):
                                    local_dangerous.add(alias)
                                    match_count += 1
                                    break
                    scan_expressions(value)
            elif isinstance(obj, list):
                for item in obj:
                    scan_expressions(item)

        scan_expressions(data)
        return local_dangerous

    # Process files in batches with threading
    batch_size = 100
    for i in range(0, len(files_with_expressions), batch_size):
        batch = files_with_expressions[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(scan_file, f): f for f in batch}
            for future in as_completed(futures):
                dangerous.update(future.result())
        if (i + batch_size) % 500 == 0:
            print(f"  Processed {min(i + batch_size, len(files_with_expressions))}/{len(files_with_expressions)} files, found {len(dangerous)} dangerous")

    print(f"  Found {match_count} expression matches, {len(dangerous)} dangerous aliases")

    return dangerous


if __name__ == "__main__":
    main()