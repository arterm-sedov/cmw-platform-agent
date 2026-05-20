#!/usr/bin/env python3
"""
Step 3: Verify Aliases Per Folder

Verifies aliases from a single folder against platform cache.
Can retry on failure (specific folder).

Usage:
    python tool_verify_aliases.py --app Volga --folder RecordTemplates --output-dir /path/to/output
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(APP_DIR))

SKIP_TYPES = {
    "ClientActivity",
    "Image",
    "ReferenceToConnection",
}

SKIP_ATTRIBUTES = {
    "AccountTemplate": {"active", "departament", "fullName", "language", "lastLoginDate",
                         "manager", "mbox", "office", "phone", "skype", "timeZone", "title", "username"},
    "OrgStructureTemplate": {"alias", "unitName", "subordinateUnit", "superiorUnit",
                            "unitDescription", "unitType"},
    "RoleTemplate": {"alias", "roleDescription", "roleIsActive", "roleName",
                     "subordinateRole", "superiorRole"},
    "ModelTemplate": {"description", "name", "order", "parentProcessModelNone",
                      "parentSubprocessEmbedded"},
    "default": {"_color", "_conversation", "_creationDate", "_creator", "_isDisabled",
                "_lastWriteDate", "_processes", "id"},
}

UUID_32_PATTERN = "^[0-9a-f]{32}$"
FORM_INT_PATTERN = r"^form\d+$"
TOOLBAR_UUID_PATTERN = r"^toolbar[0-9a-f]{32}$"

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
}


def should_skip_alias(alias: str, obj_type: str, displayName: str = "", parent_type: str = None) -> bool | str:
    """Check if alias should be skipped based on filtering rules."""
    has_display_name = bool(displayName)

    # SKIP_ATTRIBUTES - locked если есть displayName, иначе skip
    if obj_type == "Attribute":
        global_skip = SKIP_ATTRIBUTES["default"]
        parent_skip = SKIP_ATTRIBUTES.get(parent_type, set()) if parent_type else set()
        if alias in global_skip or alias in parent_skip:
            return "locked" if has_display_name else True

    # SKIP_TYPES - always skip
    if obj_type in SKIP_TYPES:
        return True

    # FormComponent UUID - apply "locked" rule if has displayName
    if obj_type == "FormComponent":
        if re.match(UUID_32_PATTERN, alias, re.I):
            return "locked" if has_display_name else True

    # Toolbar UUID - apply "locked" rule if has displayName
    if obj_type == "Toolbar":
        if re.match(TOOLBAR_UUID_PATTERN, alias, re.I):
            return "locked" if has_display_name else True

    # Form with numeric pattern - apply "locked" rule if has displayName
    if obj_type == "Form":
        if re.match(FORM_INT_PATTERN, alias, re.I):
            return "locked" if has_display_name else True

    # SKIP_TYPE_PREFIXES - apply "locked" rule if has displayName
    for prefix, types in SKIP_TYPE_PREFIXES.items():
        if obj_type in types and alias.startswith(prefix):
            return "locked" if has_display_name else True

    return False


def verify_folder(folder_name: str, app_name: str, aliases_data: dict, cache: dict, output_dir: Path) -> tuple[list, list]:
    """Verify aliases for a single folder.

    Every entry looks up ALL IDs from cache by (type, alias).
    Same alias can have multiple IDs across different containers.
    """
    verified = []
    skipped = []

    TYPE_TO_PLATFORM = {
        "RoleWorkspace": "Workspace",
        "MessageTemplateProperty": "Attribute",
        "RoleConfiguration": "Role",
        "SimplePage": "Page",
    }

    for obj in aliases_data.get("aliases", []):
        obj_type = obj["type"]
        alias = obj["alias"]
        parent_template = obj.get("parent_template", "")
        platform_type = TYPE_TO_PLATFORM.get(obj_type, obj_type)

        found_ids = []

        if platform_type in cache and alias in cache[platform_type]:
            found_ids = cache[platform_type][alias].get("ids", [])

        if not found_ids and obj_type in ("OrgStructureTemplate", "RoleTemplate"):
            if obj_type == "RoleTemplate":
                candidates = [f"{app_name}_RolesCatalog", "systemSolution_RolesCatalog"]
            else:
                candidates = [f"{app_name}_OrganizationalStructure", "systemSolution_OrganizationalStructure"]
            for prefixed_alias in candidates:
                if platform_type in cache and prefixed_alias in cache[platform_type]:
                    found_ids = cache[platform_type][prefixed_alias].get("ids", [])
                    break

        if found_ids:
            obj["ids"] = found_ids
            verified.append(obj)
        else:
            # Получить displayName для проверки skip rules
            display_names = obj.get("displayNames", [])
            display_name = display_names[0].get("displayNameOriginal", "") if display_names else ""
            parent_type = obj.get("parent_type")

            # Проверить skip rules
            skip_result = should_skip_alias(alias, obj_type, display_name, parent_type)

            if skip_result == "locked":
                # Alias попадает под правила блокировки - добавить в verified с aliasLocked=True
                obj["aliasLocked"] = True
                verified.append(obj)
            elif skip_result == True:
                # Alias должен быть пропущен полностью
                skipped.append(obj)
            else:
                # Не найден в кэше, но не попадает под skip rules - добавить в verified
                verified.append(obj)

    return verified, skipped


def main():
    parser = argparse.ArgumentParser(description="Step 3: Verify aliases per folder")
    parser.add_argument("--app", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--platform-cache", default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"/tmp/cmw-transfer/{args.app}_tr")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.platform_cache:
        cache_file = Path(args.platform_cache)
    else:
        cache_file = output_dir / f"{args.app}_platform_cache.json"

    if not cache_file.exists():
        print(f"Error: Platform cache not found at {cache_file}")
        print("Run Step 2 first: python tool_collect_platform.py --app " + args.app)
        return 1

    with open(cache_file, encoding="utf-8") as f:
        cache_data = json.load(f)

    cache = cache_data.get("cache", {})

    aliases_file = output_dir / f"{args.app}_{args.folder}_aliases.json"
    if not aliases_file.exists():
        print(f"Error: Aliases file not found at {aliases_file}")
        print(f"Run Step 1 first: python tool_extract_aliases.py --app {args.app}")
        return 1

    with open(aliases_file, encoding="utf-8") as f:
        aliases_data = json.load(f)

    output_file = output_dir / f"{args.app}_{args.folder}_verified.json"

    print(f"=== Step 3: Verify {args.app}/{args.folder} ===")

    if output_file.exists():
        try:
            with open(output_file, encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("folder") == args.folder and existing.get("verified"):
                print(f"Already verified: {len(existing['verified'])} verified, {len(existing.get('skipped', []))} skipped")
                return 0
        except (json.JSONDecodeError, OSError):
            pass

    start = time.time()
    verified, skipped = verify_folder(args.folder, args.app, aliases_data, cache, output_dir)
    elapsed = time.time() - start

    output_data = {
        "app": args.app,
        "folder": args.folder,
        "verified": verified,
        "skipped": skipped,
        "verified_count": len(verified),
        "skipped_count": len(skipped),
        "verified_at": datetime.now().isoformat(),
        "verification_time_seconds": elapsed,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Verified: {len(verified)}, Skipped: {len(skipped)} ({elapsed:.1f}s)")
    print(f"Output: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
