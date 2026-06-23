#!/usr/bin/env python3
"""
Apply displayNameRenamed to CTF JSON files.

Usage:
    python apply_display_names.py --app Volga --json-folder C:\\tmp\\cmw-transfer\\Volga_ctf --output-dir C:\\tmp\\cmw-transfer --path-mode original
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
    from urllib.parse import urlparse

    domain = os.environ.get("CMW_DOMAIN", "")
    if domain:
        return domain

    base_url = os.environ.get("CMW_BASE_URL", "")
    if not base_url:
        env_path = APP_DIR / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("CMW_BASE_URL="):
                    base_url = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if base_url:
        netloc = urlparse(base_url).netloc
        return netloc if netloc else "cmw"

    return "cmw"


def navigate_to_element(json_data: dict, path_parts: list) -> tuple[dict, str, bool]:
    """Navigate through JSON by path parts.

    path_parts = ["Root", "Rows", "$values[1]", "GlobalAlias"]
    Returns (parent_dict, key_to_update, success)
    """
    current = json_data
    parent = None
    key = None

    for part in path_parts:
        match = re.match(r'\$?values?\[(\d+)\]', part)
        if match:
            idx = int(match.group(1))
            if isinstance(current, list) and idx < len(current):
                parent = current
                current = current[idx]
                key = part
            else:
                return None, "", False
        elif isinstance(current, dict) and part in current:
            parent = current
            current = current[part]
            key = part
        else:
            return None, "", False

    return parent, key, True


def main(app: str, json_folder: str, output_dir: str, path_mode: str = "renamed") -> int:
    json_path = Path(json_folder)
    output_path = Path(output_dir)
    domain = get_domain()

    tr_file = output_path / f"{domain}_{app}_tr.json"
    if not tr_file.exists():
        print(f"Error: Translation file not found: {tr_file}")
        return 1

    with open(tr_file, encoding="utf-8") as f:
        tr_data = json.load(f)

    display_name_fields = ("Name", "DisplayName", "Text", "Description", "Title", "Header", "Tooltip", "Label", "Caption")
    updated_files = set()
    updated_count = 0

    for obj in tr_data:
        # Get displayNames array instead of single displayName
        display_names = obj.get("displayNames", [])

        for dn in display_names:
            display_name_new = dn.get("displayNameRenamed", "")
            if not display_name_new:
                continue

            # Use jsonPath from displayName object (not from root)
            if path_mode == "renamed":
                json_paths = dn.get("jsonPathRenamed", [])
            else:
                json_paths = dn.get("jsonPathOriginal", [])

            if not json_paths:
                continue

            for json_path_full in json_paths:
                # Split into file path and internal path
                if ".json/" not in json_path_full:
                    continue

                file_part, path_part = json_path_full.split(".json/", 1)
                file_part = file_part + ".json"

                ctf_path = json_path / app / file_part
                if not ctf_path.exists():
                    continue

                content = ctf_path.read_text(encoding="utf-8")
                original_content = content

                try:
                    json_data = json.loads(content)
                except json.JSONDecodeError:
                    continue

                # Parse path parts
                path_parts = path_part.split("/")

                # Navigate to element
                parent, key, success = navigate_to_element(json_data, path_parts)
                if not success or parent is None:
                    continue

                # Update the displayName field
                updated = False
                for field in display_name_fields:
                    if field in parent and parent[field] and isinstance(parent[field], str):
                        parent[field] = display_name_new
                        updated = True
                        break

                # Also check if the field is at the target level itself
                if path_parts and path_parts[-1] in display_name_fields:
                    parent[path_parts[-1]] = display_name_new
                    updated = True

                if updated:
                    content = json.dumps(json_data, ensure_ascii=False, indent=2)

                    if content != original_content:
                        ctf_path.write_text(content, encoding="utf-8")
                        updated_files.add(str(ctf_path))
                        updated_count += 1

    print(f"Updated display names in {updated_count} CTF files")
    if updated_files:
        files_list = sorted(updated_files)[:5]
        print(f"Files: {', '.join(files_list)}...")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply displayNameRenamed to CTF JSON files")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--json-folder", required=True, help="Path to folder with CTF JSON files")
    parser.add_argument("--output-dir", required=True, help="Path to output directory with _tr.json")
    parser.add_argument("--path-mode", default="renamed", choices=["original", "renamed"], help="Which path to use")
    args = parser.parse_args()
    sys.exit(main(args.app, args.json_folder, args.output_dir, args.path_mode))