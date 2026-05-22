#!/usr/bin/env python3
"""
Step 5: Finalize.

Merges all folder verified files into complete output in schema.json format:

Output fields (schema.json compliant):
  - type (str)
  - ids (array of str)
  - aliasOriginal (str)
  - aliasRenamed (str, empty by default)
  - displayNames (array of {displayNameOriginal, displayNameRenamed, jsonPathOriginal, jsonPathRenamed})
  - jsonPathOriginal (array of str)
  - jsonPathRenamed (array of str, empty by default)
  - expressions (array of {jsonPathOriginal, jsonPathRenamed, expressionOriginal, expressionRenamed})
"""

import argparse
import json
import re
import sys
from pathlib import Path

APP_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(APP_DIR))

PATTERNS = [
    r"\$" + r"{alias}\b",  # $alias - variable reference
    r"->" + r"{alias}\b",  # ->alias - method call
    r"{alias}" + r"->",    # alias-> - object as target
    r'"' + r"{alias}" + r'"',  # "alias" - string literal
]


def get_display_names(obj: dict) -> list:
    """Extract displayNames array from verified/skipped entry."""
    display_names = obj.get("displayNames", [])
    if display_names:
        return display_names

    display_name = obj.get("displayNameOriginal") or obj.get("displayName", "")
    if display_name:
        json_path = obj.get("jsonPathOriginal", obj.get("json_file", ""))
        if isinstance(json_path, list):
            dn_paths = [p for p in json_path if "/Name" in p or "/DisplayName" in p]
        else:
            dn_paths = [json_path] if json_path else []

        return [{
            "displayNameOriginal": display_name,
            "displayNameRenamed": "",
            "jsonPathOriginal": dn_paths,
            "jsonPathRenamed": [],
        }]

    return []


def has_display_name(obj: dict) -> bool:
    """Check if entry has any non-empty displayName."""
    for dn in get_display_names(obj):
        if dn.get("displayNameOriginal", ""):
            return True
    return False


def alias_used_in_expression(alias: str, expression: str) -> bool:
    """Check if alias is actually used in the expression via precise patterns."""
    if not expression:
        return False
    for pattern_template in PATTERNS:
        pattern = pattern_template.format(alias=re.escape(alias))
        if re.search(pattern, expression):
            return True
    return False


def extract_template_from_path(path: str) -> str:
    """Extract template name from JSON path.
    Example: 'Volga/RecordTemplates/ProvedenieTO/Attributes/x.json' -> 'ProvedenieTO'
    """
    if not path:
        return ""
    normalized = path.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p and not p.endswith(".json")]
    template_folders = {"RecordTemplates", "ProcessTemplates", "Workspaces", "Pages",
               "Toolbars", "Datasets", "Forms", "Attributes", "UserCommands",
               "Cards", "Carts", "Roles", "Triggers", "WidgetConfigs", "AccountTemplates"}
    not_template = {"GlobalAlias", "InstanceGlobalAlias", "RootResource", "Container", "Root", "Columns", "Items"}
    
    for i, part in enumerate(parts):
        if part in template_folders:
            if i + 1 < len(parts):
                result = parts[i + 1]
                if result in not_template:
                    continue
                return result
    return ""


def match_expressions_to_entry(entry_paths: list, dangerous_expressions: list, alias: str = None) -> list:
    """Match expressions to an entry based on alias usage.
    Only includes expressions where alias is actually used ($alias, ->alias, alias->, "alias").
    """
    matched = []
    for expr in dangerous_expressions:
        if alias and not alias_used_in_expression(alias, expr.get("expressionOriginal", "")):
            continue
        matched.append(expr)
    return matched


def get_server_url(output_dir: Path, app: str) -> str:
    """Read metadata.json to get server URL."""
    candidates = [
        output_dir.parent / f"{app}_json",
        output_dir.parent / app,
    ]
    for extract_dir in candidates:
        metadata_file = extract_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, encoding="utf-8") as f:
                data = json.load(f)
            server = data.get("Server", "")
            if server:
                return server.replace("https://", "").replace("http://", "").rstrip("/")
    return "unknown"


def deduplicate_by_ids(entries: list) -> list:
    """Merge entries with same (aliasOriginal, type, ids).
    
    Merge strategy:
    - ids: union of all non-empty ids arrays
    - displayNames: first non-empty displayNames
    - aliasLocked: True if ANY entry is True (locked wins)
    - jsonPathOriginal: union of all paths
    - expressions: union of all expressions (dedup by jsonPathOriginal)
    """
    dedup_map = {}

    for obj in entries:
        ids_tuple = tuple(sorted(obj.get("ids", []))) if obj.get("ids") else ()
        key = (obj.get("aliasOriginal", ""), obj.get("type", ""), ids_tuple)

        if key not in dedup_map:
            dedup_map[key] = obj.copy()
        else:
            existing = dedup_map[key]

            if not existing.get("ids") and obj.get("ids"):
                existing["ids"] = obj["ids"]
            elif obj.get("ids") and existing.get("ids"):
                existing["ids"] = list(set(existing["ids"] + obj["ids"]))

            if obj.get("aliasLocked") and not existing.get("aliasLocked"):
                existing["aliasLocked"] = True

            if not existing.get("displayNames") and obj.get("displayNames"):
                existing["displayNames"] = obj["displayNames"]

            existing_paths = set(existing.get("jsonPathOriginal", []))
            for path in obj.get("jsonPathOriginal", []):
                if path not in existing_paths:
                    existing.setdefault("jsonPathOriginal", []).append(path)

            existing_expr = {e.get("jsonPathOriginal", "") for e in existing.get("expressions", [])}
            for expr in obj.get("expressions", []):
                key_expr = expr.get("jsonPathOriginal", "")
                if key_expr and key_expr not in existing_expr:
                    existing.setdefault("expressions", []).append(expr)

    return list(dedup_map.values())


def main():
    parser = argparse.ArgumentParser(description="Step 5: Finalize verified aliases")
    parser.add_argument("--app", required=True)
    parser.add_argument("--output-dir", default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"/tmp/cmw-transfer/{args.app}_tr")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Step 5: Finalize for {args.app} ===")

    # Load all verified/skipped files
    verified = []
    skipped = []

    verified_files = sorted(output_dir.glob(f"{args.app}_*_verified.json"))
    print(f"Found {len(verified_files)} folder verified files")

    for vf in verified_files:
        try:
            with open(vf, encoding="utf-8") as f:
                data = json.load(f)
            folder = data.get("folder", vf.stem.replace(f"{args.app}_", "").replace("_verified", ""))
            print(f"  {folder}: {data.get('verified_count', 0)} verified, {data.get('skipped_count', 0)} skipped")
            verified.extend(data.get("verified", []))
            skipped.extend(data.get("skipped", []))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: Failed to load {vf}: {e}")

    app_aliases_file = output_dir / f"{args.app}_Application_aliases.json"
    if app_aliases_file.exists():
        try:
            with open(app_aliases_file, encoding="utf-8") as f:
                app_data = json.load(f)
            app_aliases = app_data.get("aliases", [])
            if app_aliases:
                print(f"  Application: {len(app_aliases)} aliases (auto-locked)")
                for alias_entry in app_aliases:
                    alias_entry["aliasLocked"] = True
                verified.extend(app_aliases)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: Failed to load Application aliases: {e}")

    # Load dangerous aliases
    dangerous_file = output_dir / f"{args.app}_dangerous_aliases.json"
    dangerous = set()
    dangerous_expressions = []  # list of {alias, jsonPathOriginal, expressionOriginal}

    if dangerous_file.exists():
        try:
            with open(dangerous_file, encoding="utf-8") as f:
                dangerous_data = json.load(f)
            dangerous_list = dangerous_data.get("dangerous_aliases", [])
            # Handle both old format (list of strings) and new format (list of dicts)
            dangerous = set()
            for d in dangerous_list:
                if isinstance(d, dict):
                    dangerous.add(d.get("alias", ""))
                else:
                    dangerous.add(d)
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
        expressions = match_expressions_to_entry(json_path, dangerous_expressions, alias)

        # Check if dangerous
        is_dangerous = alias in dangerous

        # aliasLocked logic for verified entries
        alias_locked = obj.get("aliasLocked", False)
        obj_type = obj.get("type", "")
        if obj_type == "Application":
            alias_locked = True
        elif alias_locked and is_dangerous:
            if obj_type in ("UserCommand", "Attribute"):
                pass
            else:
                alias_locked = False

        schema_obj = {
            "type": obj.get("type", ""),
            "ids": obj_ids,
            "aliasOriginal": alias,
            "aliasRenamed": "",
            "displayNames": get_display_names(obj),
            "jsonPathOriginal": json_path,
            "jsonPathRenamed": [],
            "expressions": expressions,
            "aliasLocked": alias_locked,
        }

        schema_verified.append(schema_obj)

    # Process skipped entries
    schema_skipped = []
    for obj in skipped:
        alias = obj.get("alias", obj.get("aliasOriginal", ""))

        # Get jsonPathOriginal
        json_path = obj.get("jsonPathOriginal", obj.get("path", obj.get("json_file", "")))
        if isinstance(json_path, str):
            json_path = [json_path] if json_path else []
        elif not isinstance(json_path, list):
            json_path = []

        # Get expressions for this alias
        expressions = match_expressions_to_entry(json_path, dangerous_expressions, alias)

        is_dangerous = alias in dangerous

        alias_locked = obj.get("aliasLocked", False)
        if obj.get("type") == "Application":
            alias_locked = True

        schema_obj = {
            "type": obj.get("type", ""),
            "ids": [],
            "aliasOriginal": alias,
            "aliasRenamed": "",
            "displayNames": get_display_names(obj),
            "jsonPathOriginal": json_path,
            "jsonPathRenamed": [],
            "expressions": expressions,
            "aliasLocked": alias_locked,
        }

        schema_skipped.append(schema_obj)

    # Merge all entries
    all_entries = schema_verified + schema_skipped

    print(f"\nDeduplicating {len(all_entries)} entries by ids...")
    all_entries = deduplicate_by_ids(all_entries)
    print(f"After dedup: {len(all_entries)} entries")

    verified_locked = [v for v in all_entries if v.get("aliasLocked")]
    verified_normal = [v for v in all_entries if not v.get("aliasLocked")]

    print(f"\nTotal entries: {len(all_entries)}")
    print(f"  aliasLocked=true (will skip): {len(verified_locked)}")
    print(f"  aliasLocked=false (will rename): {len(verified_normal)}")

    # Output single file in schema format
    server_url = get_server_url(output_dir, args.app)
    verified_file = output_dir / f"{server_url}_{args.app}_tr.json"
    with open(verified_file, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    print(f"\n=== Final Output ===")
    print(f"Verified complete: {verified_file} ({len(all_entries)} objects)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
