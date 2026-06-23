#!/usr/bin/env python3
"""
Build expressions index from dangerous_aliases.json, danger_filtered.json, and _tr.json.

Creates {domain}_{app}_expressions.json with unique expressions and their metadata.

Usage:
    python build_expressions_index.py --app Volga --output-dir C:\\tmp\\cmw-transfer
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

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


ALLOWED_TYPES = {
    "RecordTemplate",
    "Attribute",
    "AccountTemplate",
    "RoleTemplate",
    "OrgStructureTemplate",
    "Variant",
}


def build_full_alias_map_from_tr(tr_data: list) -> dict[str, str]:
    """Build full mapping of aliasOriginal -> aliasRenamed from _tr.json (all types)."""
    alias_map = {}
    for entry in tr_data:
        ao = entry.get("aliasOriginal", "")
        ar = entry.get("aliasRenamed", "")
        if ao and ar:
            alias_map[ao] = ar
    return alias_map


def build_alias_map_from_tr(tr_data: list) -> dict[str, str]:
    """Build mapping of aliasOriginal -> aliasRenamed from _tr.json (filtered by ALLOWED_TYPES)."""
    alias_map = {}
    for entry in tr_data:
        entry_type = entry.get("type", "")
        if entry_type not in ALLOWED_TYPES:
            continue
        ao = entry.get("aliasOriginal", "")
        ar = entry.get("aliasRenamed", "")
        if ao and ar:
            alias_map[ao] = ar
    return alias_map


def build_alias_map_from_danger(danger_data: dict) -> dict[str, str]:
    """Build mapping of aliasOriginal -> aliasRenamed from danger_filtered.json."""
    alias_map = {}
    for entry in danger_data.get("entries", []):
        entry_type = entry.get("type", "")
        if entry_type not in ALLOWED_TYPES:
            continue
        ao = entry.get("aliasOriginal", "")
        ar = entry.get("aliasRenamed", "")
        if ao and ar:
            alias_map[ao] = ar
    return alias_map


def replace_alias_in_path(path: str, alias_map: dict[str, str], tr_data: list | None = None) -> str:
    """Replace aliasOriginal with aliasRenamed in jsonPath using exact match.

    Handles segments like 'Startovayaforma.json#FormRules' by:
    1. Splitting on # to separate file path from JSON-path reference
    2. Stripping .json extension from filename
    3. Replacing alias in filename only
    4. Reassembling: 'filename.json#FormRules'

    Special handling for Trigger files:
    - Extract ProcessTemplate aliases from tr_data
    - Replace _ProcessTemplate suffix in filenames with _ProcessTemplateRenamed
    """
    if "#" in path:
        file_part, json_path = path.split("#", 1)
        json_path = "#" + json_path
    else:
        file_part = path
        json_path = ""

    parts = file_part.replace("/", "\\").split("\\")
    result_parts = []

    for part in parts:
        if part.lower().endswith(".json"):
            base_name = part[:-5]
            alias_renamed = alias_map.get(base_name, base_name)
            part = alias_renamed + ".json"

            if tr_data and "\\Triggers\\" in file_part.replace("/", "\\"):
                process_template_aliases = {}
                for entry in tr_data:
                    if entry.get("type") == "ProcessTemplate":
                        ao = entry.get("aliasOriginal", "")
                        ar = entry.get("aliasRenamed", "")
                        if ao and ar:
                            process_template_aliases[ao] = ar

                for ao, ar in process_template_aliases.items():
                    suffix = f"_{ao}"
                    if part[:-5].endswith(suffix):
                        part = part[:-5 - len(suffix)] + f"_{ar}" + ".json"
                        break
        else:
            alias_renamed = alias_map.get(part, part)
            part = alias_renamed
        result_parts.append(part)

    return "\\".join(result_parts) + json_path


def replace_alias_in_expression(expression: str, alias_map: dict[str, str]) -> str:
    """Replace aliasOriginal with aliasRenamed in expression text.
    
    Only replaces if:
    - Before alias: nothing OR one of " ", ",", "->", "(", '"', "$"
    - After alias: nothing OR _calc (already replaced)
    """
    result = expression
    # Sort by length (longest first) to avoid partial replacements
    sorted_aliases = sorted(alias_map.items(), key=lambda x: len(x[0]), reverse=True)
    
    valid_before = set([' ', ',', '(', '"', '$', '', '\n', '\t', '-', '>'])
    delimiters = [' ', '->', ',', ')', '"', '(', '[', ']', '\n', '\t']
    
    for ao, ar in sorted_aliases:
        idx = result.find(ao)
        while idx != -1:
            # Check BEFORE - must be valid delimiter or nothing
            before = result[max(0, idx-1):idx] if idx > 0 else ''
            before_valid = before in valid_before
            
            # Check AFTER - get first character after alias
            first_after = result[idx + len(ao):idx + len(ao) + 1] if idx + len(ao) < len(result) else ''
            is_letter = first_after.isalpha() or first_after.isdigit() or first_after == '_'
            after_valid = not is_letter or first_after == ''
            
            # Replace only if both before and after are valid
            if before_valid and after_valid:
                result = result[:idx] + ar + result[idx + len(ao):]
                idx = result.find(ao, idx + len(ar))
            else:
                # Skip this occurrence, find next
                idx = result.find(ao, idx + 1)
    
    return result


def parse_expression_string(expr_str: str) -> dict | None:
    """Parse expression string like '@{expressionOriginal=...}'."""
    result = {}

    orig_match = re.search(r"expressionOriginal=([^;]+)", expr_str)
    if orig_match:
        result["expressionOriginal"] = orig_match.group(1).strip()

    aliases_match = re.search(r"aliases=\[([^\]]*)\]", expr_str)
    if aliases_match and aliases_match.group(1).strip():
        result["aliases"] = [a.strip() for a in aliases_match.group(1).split(",")]
    else:
        result["aliases"] = []

    paths_match = re.search(r"jsonPaths=\[([^\]]*)\]", expr_str)
    if paths_match and paths_match.group(1).strip():
        result["jsonPaths"] = [p.strip() for p in paths_match.group(1).split(",")]
    else:
        result["jsonPaths"] = []

    return result if "expressionOriginal" in result else None


def main(app: str, output_dir: str, domain: str | None = None) -> int:
    output_path = Path(output_dir)
    if not domain:
        domain = get_domain()

    tr_file = output_path / f"{domain}_{app}_tr.json"
    danger_file = output_path / f"{domain}_{app}_danger_filtered.json"

    if not tr_file.exists():
        print(f"Error: Translation file not found: {tr_file}")
        return 1

    with open(tr_file, encoding="utf-8") as f:
        tr_data = json.load(f)

    print(f"Loaded {len(tr_data)} entries from {tr_file.name}")

    if not danger_file.exists():
        print(f"Error: Danger filtered file not found: {danger_file}")
        return 1

    with open(danger_file, encoding="utf-8") as f:
        danger_filtered = json.load(f)

    meta = danger_filtered.get("_meta", {})
    print(f"Loaded {meta.get('total_count', 0)} entries from {danger_file.name}")

    entries = danger_filtered.get("entries", [])
    if not entries:
        entries = danger_filtered if isinstance(danger_filtered, list) else []

    print(f"Processing {len(entries)} entries for expressions...")

    tr_alias_map = build_alias_map_from_tr(tr_data)
    print(f"Built alias map with {len(tr_alias_map)} entries")

    tr_full_alias_map = build_full_alias_map_from_tr(tr_data)
    print(f"Built full alias map with {len(tr_full_alias_map)} entries")

    danger_alias_map = build_alias_map_from_danger(danger_filtered)
    print(f"Built danger alias map with {len(danger_alias_map)} entries")

    result = []
    count = 0

    for entry in entries:
        alias_orig = entry.get("aliasOriginal", "")
        alias_renamed = entry.get("aliasRenamed", "")

        if not alias_orig:
            continue

        expr_list = entry.get("expressions", [])
        if not expr_list:
            continue

        for expr in expr_list:
            if isinstance(expr, str):
                parsed = parse_expression_string(expr)
                if not parsed:
                    continue
                expr_orig = parsed.get("expressionOriginal", "")
                expr_aliases = parsed.get("aliases", [])
                json_paths_orig = parsed.get("jsonPaths", [])
            elif isinstance(expr, dict):
                expr_orig = expr.get("expressionOriginal", "")
                expr_aliases = expr.get("aliases", [])
                json_paths_orig = expr.get("jsonPaths", [])
            else:
                continue

            if not expr_orig:
                continue

            expr_renamed = replace_alias_in_expression(expr_orig, tr_alias_map)

            json_paths = []
            for path_orig in json_paths_orig:
                path_renamed = replace_alias_in_path(path_orig, tr_full_alias_map, tr_data)
                json_paths.append({
                    "jsonPathOriginal": path_orig,
                    "jsonPathRenamed": path_renamed,
                })

            aliases = []
            for a_orig in expr_aliases:
                a_renamed = danger_alias_map.get(a_orig, a_orig)
                aliases.append({
                    "aliasOriginal": a_orig,
                    "aliasRenamed": a_renamed,
                })

            if alias_orig and alias_orig not in [a["aliasOriginal"] for a in aliases]:
                aliases.append({
                    "aliasOriginal": alias_orig,
                    "aliasRenamed": alias_renamed,
                })

            result.append({
                "expressionOriginal": expr_orig,
                "expressionRenamed": expr_renamed,
                "jsonPaths": json_paths,
                "aliases": aliases,
            })
            count += 1
            if count % 50 == 0:
                print(f"  Processed {count} expressions...")

    print(f"Built {len(result)} expressions with metadata")

    output_file = output_path / f"{domain}_{app}_expressions.json"
    output_data = {"expressions": result}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Written {len(result)} expressions to {output_file.name}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build expressions index from dangerous sources"
    )
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument(
        "--output-dir", required=True, help="Path to output directory with files"
    )
    parser.add_argument("--domain", help="Domain (e.g., mz-fr.test.cbap.ru)")
    args = parser.parse_args()
    sys.exit(main(args.app, args.output_dir, args.domain))
