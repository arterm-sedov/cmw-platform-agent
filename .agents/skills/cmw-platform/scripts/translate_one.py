#!/usr/bin/env python3
"""
Translate single alias interactively (or resume from last position).

Usage:
    python translate_one.py --app Volga --output-dir C:\\tmp\\cmw-transfer --alias "Квартира"
    python translate_one.py --app Volga --output-dir C:\\tmp\\cmw-transfer --resume
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


def calculate_new_json_path(original_path: str, old_alias: str, new_alias: str) -> str:
    """Replace alias in JSON path with new alias."""
    result = original_path
    result = result.replace(f"/{old_alias}/", f"/{new_alias}/")
    result = result.replace(f"/{old_alias}.json", f"/{new_alias}.json")
    result = result.replace(f"{old_alias}.json", f"{new_alias}.json")
    return result


def load_resume_state(output_dir: str, app_name: str) -> dict | None:
    output_path = Path(output_dir)
    state_file = output_path / f"{app_name}_resume_state.json"
    if state_file.exists():
        with open(state_file, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_resume_state(output_dir: str, app_name: str, alias: str, index: int):
    output_path = Path(output_dir)
    state_file = output_path / f"{app_name}_resume_state.json"
    state = {
        "last_alias": alias,
        "last_index": index,
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def main(app: str, output_dir: str, alias: str | None = None, resume: bool = False,
         dangerous_suffix: str = "_calc", safe_suffix: str = "_sv") -> int:
    output_path = Path(output_dir)
    domain = get_domain()

    tr_file = output_path / f"{domain}_{app}_tr.json"
    if not tr_file.exists():
        print(f"Error: Translation file not found: {tr_file}")
        return 1

    with open(tr_file, encoding="utf-8") as f:
        tr_data = json.load(f)

    resume_state = load_resume_state(output_dir, app)
    start_index = 0

    if resume and resume_state:
        start_index = resume_state.get("last_index", 0)
        print(f"Resuming from index {start_index}")

    target_index = start_index
    if alias:
        for i, obj in enumerate(tr_data):
            if obj.get("aliasOriginal") == alias:
                target_index = i
                break
        if target_index >= len(tr_data):
            print(f"Error: Alias {alias} not found in translation file")
            return 1
    else:
        if start_index >= len(tr_data):
            print("Error: No more aliases to translate")
            return 1

    obj = tr_data[target_index]
    alias_orig = obj.get("aliasOriginal", "")
    alias_new = obj.get("aliasRenamed", "")

    print(f"Alias {target_index + 1}/{len(tr_data)}: {alias_orig}")
    print(f"  Type: {obj.get('type', '')}")
    print(f"  ID: {obj.get('ids', [])}")
    print(f"  Display Name: {obj.get('displayNameOriginal', '')}")
    print(f"  Current aliasRenamed: {alias_new}")

    display_name_orig = obj.get("displayNameOriginal", "")
    display_name_new = obj.get("displayNameRenamed", "")

    if not alias_new:
        suffix = dangerous_suffix if obj.get("expressions") else safe_suffix
        new_alias = alias_orig + suffix
        print(f"  Suggested: {new_alias}")
        print("  Enter new aliasRenamed (or press Enter to accept suggested): ")

        new_input = input("  > ")
        if new_input.strip():
            obj["aliasRenamed"] = new_input.strip()
        else:
            obj["aliasRenamed"] = new_alias

        if display_name_orig:
            print(f"  Current displayNameRenamed: {display_name_new}")
            print("  Enter new displayNameRenamed (or press Enter to keep current): ")
            display_input = input("  > ")
            obj["displayNameRenamed"] = display_input.strip() if display_input.strip() else display_name_orig

        new_alias = obj["aliasRenamed"]

        for expr in obj.get("expressions", []):
            expr["jsonPathRenamed"] = calculate_new_json_path(
                expr.get("jsonPathOriginal", ""), alias_orig, new_alias
            )
            orig_expr = expr.get("expressionOriginal", "")
            if orig_expr:
                expr["expressionRenamed"] = orig_expr.replace(alias_orig, new_alias)

        save_resume_state(output_dir, app, alias_orig, target_index)

        with open(tr_file, "w", encoding="utf-8") as f:
            json.dump(tr_data, f, indent=2, ensure_ascii=False)

        print(f"Updated alias {alias_orig} -> {new_alias}")
    else:
        print(f"  aliasRenamed already set to: {alias_new}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate single alias interactively")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--output-dir", required=True, help="Path to output directory with _tr.json")
    parser.add_argument("--alias", help="Alias to translate")
    parser.add_argument("--resume", action="store_true", help="Resume from last translated alias")
    parser.add_argument("--dangerous-suffix", default="_calc", help="Suffix for dangerous aliases")
    parser.add_argument("--safe-suffix", default="_sv", help="Suffix for safe aliases")
    args = parser.parse_args()
    sys.exit(main(args.app, args.output_dir, args.alias, args.resume, args.dangerous_suffix, args.safe_suffix))