#!/usr/bin/env python3
"""
Apply expressionRenamed to CTF JSON files.

Usage:
    python apply_expressions.py --app Volga --json-folder C:\\tmp\\cmw-transfer\\Volga_ctf --output-dir C:\\tmp\\cmw-transfer
"""
import argparse
import json
import sys
from pathlib import Path

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


def main(app: str, json_folder: str, output_dir: str) -> int:
    json_path = Path(json_folder)
    output_path = Path(output_dir)
    domain = get_domain()

    tr_file = output_path / f"{domain}_{app}_tr.json"
    if not tr_file.exists():
        print(f"Error: Translation file not found: {tr_file}")
        return 1

    with open(tr_file, encoding="utf-8") as f:
        tr_data = json.load(f)

    expr_fixes: dict[str, str] = {}
    for obj in tr_data:
        alias_orig = obj.get("aliasOriginal", "")
        alias_renamed = obj.get("aliasRenamed", "")
        if not alias_orig or not alias_renamed:
            continue
        for expr_item in obj.get("expressions", []):
            orig_expr = expr_item.get("expressionOriginal", "")
            new_expr = expr_item.get("expressionRenamed", "")
            if orig_expr and new_expr and orig_expr != new_expr:
                expr_fixes[orig_expr] = new_expr

    if not expr_fixes:
        print("No expression fixes to apply")
        return 0

    updated_files = set()
    updated_count = 0

    ctf_root = json_path / app
    for json_file in ctf_root.rglob("*.json"):
        if not json_file.is_file():
            continue
        content = json_file.read_text(encoding="utf-8")
        original_content = content

        for orig_expr, new_expr in expr_fixes.items():
            if orig_expr in content:
                content = content.replace(orig_expr, new_expr)

        if content != original_content:
            json_file.write_text(content, encoding="utf-8")
            updated_files.add(str(json_file))
            updated_count += 1

    print(f"Updated expressions in {updated_count} CTF files")
    if updated_files:
        files_list = sorted(updated_files)[:5]
        print(f"Files: {', '.join(files_list)}...")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply expressionRenamed to CTF JSON files")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--json-folder", required=True, help="Path to folder with CTF JSON files")
    parser.add_argument("--output-dir", required=True, help="Path to output directory with _tr.json")
    args = parser.parse_args()
    sys.exit(main(args.app, args.json_folder, args.output_dir))