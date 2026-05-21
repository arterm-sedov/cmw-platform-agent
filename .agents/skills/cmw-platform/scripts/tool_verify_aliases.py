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
    "AccountTemplate": {"active", "department", "departament", "fullName", "language", "lastLoginDate",
                         "manager", "mbox", "office", "phone", "skype", "timeZone", "title", "username"},
    "OrgStructureTemplate": {"alias", "unitName", "subordinateUnit", "superiorUnit",
                            "unitDescription", "unitType"},
    "RoleTemplate": {"alias", "roleDescription", "roleIsActive", "roleName",
                     "subordinateRole", "superiorRole"},
    "ProcessTemplate": {"status", "currentLifetimeStatus", "name", "minDueDate", "object",
                       "hasTokenError", "activeVersion", "currentActivity"},
    "DesktopWidgetConfig": {"fullName", "title", "name"},
    "ModelTemplate": {"description", "name", "order", "parentProcessModelNone",
                      "parentSubprocessEmbedded"},
    "default": {"_color", "_conversation", "_creationDate", "_creator", "_isDisabled",
                "_lastWriteDate", "_processes", "id", "_creator", "_creationDate", "_isDisabled", 
                "_processes", "_color"},
}

UUID_32_PATTERN = "^[0-9a-f]{32}$"
FORM_INT_PATTERN = r"^form\d+$"
TOOLBAR_UUID_PATTERN = r"^toolbar[0-9a-f]{32}$"

ALIAS_MAPPING = {
    # AccountTemplate attributes (cmw_account_* -> *)
    "cmw_account_department": "department",
    "cmw_account_departament": "departament",
    "cmw_account_fullName": "fullName",
    "cmw_account_title": "title",
    "cmw_account_mbox": "mbox",
    "cmw_account_skype": "skype",
    "cmw_account_username": "username",
    "cmw_account_active": "active",
    "cmw_account_phone": "phone",
    "cmw_account_manager": "manager",
    "cmw_account_office": "office",
    "cmw_account_lastLoginDate": "lastLoginDate",
    # OrgStructureTemplate attributes
    "cmw_ou_type": "unitType",
    "cmw_ou_name": "unitName",
    "cmw_ou_description": "unitDescription",
    "superiorUnit": "superiorUnit",
    "subordinateUnit": "subordinateUnit",
    # ProcessTemplate attributes (cmw_process_*, cmw_task_* -> *)
    "cmw_process_status": "status",
    "cmw_process_currentLifetimeStatus": "currentLifetimeStatus",
    "cmw_process_name": "name",
    "cmw_process_minDueDate": "minDueDate",
    "cmw_process_businessObject": "object",
    "cmw_process_hasTokenError": "hasTokenError",
    "cmw_process_activeVersion": "activeVersion",
    "cmw_process_currentActivity": "currentActivity",
    "cmw_task_planEndDate": "planEndDate",
    "cmw_task_objectId": "objectId",
    "cmw_task_displayId": "displayId",
    "cmw_task_owner": "owner",
    # RoleTemplate attributes
    "cmw_role_name": "roleName",
    "cmw_role_description": "roleDescription",
    "superiorRole": "superiorRole",
    "subordinateRole": "subordinateRole",
    # Common attribute mappings
    "creator": "_creator",
    "creationDate": "_creationDate",
    "isDisabled": "_isDisabled",
    "processes": "_processes",
    "color": "_color",
}

FOLDER_TO_PARENT_TYPE = {
    "AccountTemplates": "AccountTemplate",
    "OrgStructureTemplates": "OrgStructureTemplate",
    "ProcessTemplates": "ProcessTemplate",
    "RoleTemplates": "RoleTemplate",
    "RecordTemplates": "RecordTemplate",
    "MessageTemplates": "MessageTemplate",
    "Triggers": "Trigger",
    "Workspaces": "Workspace",
    "Pages": "Page",
    "WidgetConfigs": "DesktopWidgetConfig",
    "Routes": "Route",
    "Streams": "Stream",
    "Carts": "Cart",
    "Roles": "Role",
    "Application": "Application",
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
}


def apply_alias_mapping(alias: str, obj_type: str, parent_type: str = None) -> str:
    """Apply alias mapping before cache lookup and skip rules.
    
    Maps aliases with prefixes to their canonical form:
    - cmw_account_* -> * (AccountTemplate attributes)
    - cmw_ou_* -> * (OrgStructureTemplate attributes)
    - cmw_process_* -> * (ProcessTemplate attributes)
    - cmw_task_* -> * (ProcessTemplate task attributes)
    - cmw_role_* -> * (RoleTemplate attributes)
    - creator -> _creator, etc. (common attributes)
    """
    return ALIAS_MAPPING.get(alias, alias)


def should_skip_alias(alias: str, obj_type: str, displayName: str = "", parent_type: str = None, from_mapping: bool = False) -> bool | str:
    """Check if alias should be skipped based on filtering rules.
    
    Args:
        alias: The alias to check
        obj_type: The object type (Attribute, etc.)
        displayName: The display name if any
        parent_type: The parent type (AccountTemplate, etc.)
        from_mapping: If True, this alias came from ALIAS_MAPPING - always locked
    """
    has_display_name = bool(displayName)

    # SKIP_ATTRIBUTES - for mapped aliases, always locked; otherwise depends on displayName
    if obj_type == "Attribute":
        global_skip = SKIP_ATTRIBUTES["default"]
        parent_skip = SKIP_ATTRIBUTES.get(parent_type, set()) if parent_type else set()
        if alias in global_skip or alias in parent_skip:
            # If this came from alias mapping, always locked
            if from_mapping:
                return "locked"
            return "locked" if has_display_name else True

    # SKIP_TYPES - always skip
    if obj_type in SKIP_TYPES:
        return True

    # FormComponent - always locked (they are UI components, not real aliases)
    if obj_type == "FormComponent":
        return "locked"

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
        parent_type = obj.get("parent_type")
        
        # Fallback: determine parent_type from folder_name if not set
        if not parent_type and folder_name in FOLDER_TO_PARENT_TYPE:
            parent_type = FOLDER_TO_PARENT_TYPE[folder_name]
        
        # Apply alias mapping before cache lookup and skip rules
        mapped_alias = apply_alias_mapping(alias, obj_type, parent_type)
        platform_type = TYPE_TO_PLATFORM.get(obj_type, obj_type)

        found_ids = []

        if platform_type in cache and mapped_alias in cache[platform_type]:
            found_ids = cache[platform_type][mapped_alias].get("ids", [])

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

            # Check if alias was mapped
            alias_was_mapped = (mapped_alias != alias)
            
            # Проверить skip rules using MAPPED alias
            skip_result = should_skip_alias(mapped_alias, obj_type, display_name, parent_type, alias_was_mapped)

            if skip_result == "locked":
                # Alias попадает под правила блокировки - добавить в verified с aliasLocked=True
                obj["aliasLocked"] = True
                # Store the mapped alias for reference
                obj["aliasMapped"] = mapped_alias
                verified.append(obj)
            elif skip_result == True:
                # Alias должен быть пропущен полностью
                skipped.append(obj)
            else:
                # Не найден в кэше, но не попадает под skip rules - добавить в verified
                obj["aliasMapped"] = mapped_alias
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
