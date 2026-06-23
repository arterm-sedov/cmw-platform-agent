#!/usr/bin/env python3
"""
Apply expression renames from expressions.json to CTF files.

Usage:
    python apply_expressions.py --app Volga --output-dir C:\\tmp\\cmw-transfer
    python apply_expressions.py --app Volga --ctf-root C:\\tmp\\Volga_tr_json --dry-run

Workflow:
    1. Export CTF from platform (Volga_tr_json)
    2. Run apply_expressions to update expressions in CTF files
"""
import argparse
import json
from pathlib import Path
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


def parse_json_path(json_path: str) -> list[str]:
    """Parse JSON path like 'FormRules.$values[0].Actions.$values[0].ConditionExpression'."""
    import re

    parts = []
    current = ""
    i = 0
    while i < len(json_path):
        char = json_path[i]
        if char == ".":
            if current:
                parts.append(current)
                current = ""
            match = re.match(r"\.\$values\[(\d+)\]", json_path[i:])
            if match:
                parts.append(f".$values[{match.group(1)}]")
                i += match.end()
            else:
                i += 1
        else:
            current += char
            i += 1
    if current:
        parts.append(current)
    return parts


def get_value_by_path(obj: dict, path_parts: list[str]) -> tuple[dict, str, any]:
    """Navigate to value by path parts. Returns (parent, key, value)."""
    current = obj
    for _i, part in enumerate(path_parts[:-1]):
        if part.startswith(".$values["):
            index = int(part[9:-1])
            if not isinstance(current, dict) or "$values" not in current:
                msg = f"Expected array at {part}, got {type(current)}"
                raise ValueError(msg)
            current = current["$values"][index]
        else:
            if not isinstance(current, dict) or part not in current:
                msg = f"Expected key '{part}' in {list(current.keys()) if isinstance(current, dict) else type(current)}"
                raise ValueError(msg)
            current = current[part]
    last_part = path_parts[-1]
    if last_part.startswith(".$values["):
        index = int(last_part[9:-1])
        return current, f"$values[{index}]", current["$values"][index]
    return current, last_part, current.get(last_part)


def set_value_by_path(obj: dict, path_parts: list[str], new_value: any) -> None:
    """Set value at path."""
    parent, key, _ = get_value_by_path(obj, path_parts)
    if key.startswith("$values["):
        index = int(key[8:-1])
        parent["$values"][index] = new_value
    else:
        parent[key] = new_value


def parse_json_path_renamed(json_path_renamed: str) -> tuple[str, str, str]:
    """Parse jsonPathRenamed like 'CMW_FM\\RecordTemplates\\Meters_calc\\Forms\\form533.json#FormRules.$values[0].Actions.$values[0].ConditionExpression'.

    Returns: (solution, file_path, json_path)
    """
    if "#" not in json_path_renamed:
        msg = f"Invalid jsonPathRenamed format (missing #): {json_path_renamed}"
        raise ValueError(msg)

    path_part, json_path = json_path_renamed.split("#", 1)
    path_part = path_part.replace("/", "\\")

    parts = path_part.split("\\")
    solution = parts[0]
    file_rel_path = "\\".join(parts[1:])
    if not file_rel_path.endswith(".json"):
        file_rel_path += ".json"

    return solution, file_rel_path, json_path


def apply_expressions(
    app: str,
    output_dir: str,
    ctf_root: str | None = None,
    domain: str | None = None,
    dry_run: bool = False,
) -> int:
    """Apply expression renames to CTF files."""
    output_path = Path(output_dir)
    if not domain:
        domain = get_domain()

    if not ctf_root:
        ctf_root = str(output_path.parent / f"{app}_tr_json")

    expressions_file = output_path / f"{domain}_{app}_expressions.json"
    if not expressions_file.exists():
        print(f"Error: Expressions file not found: {expressions_file}")
        return 1

    with open(expressions_file, encoding="utf-8") as f:
        expressions_data = json.load(f)

    expressions = expressions_data.get("expressions", [])
    print(f"Loaded {len(expressions)} expressions from {expressions_file.name}")

    errors = []
    applied = 0
    skipped = 0

    ctf_path = Path(ctf_root)
    if not ctf_path.exists():
        print(f"Error: CTF root not found: {ctf_root}")
        return 1

    for idx, expr in enumerate(expressions):
        expr_orig = expr.get("expressionOriginal", "")
        expr_renamed = expr.get("expressionRenamed", "")
        json_paths = expr.get("jsonPaths", [])

        if not json_paths:
            continue

        for jp in json_paths:
            json_path_renamed = jp.get("jsonPathRenamed", "")
            if not json_path_renamed:
                continue

            try:
                solution, file_rel_path, json_path = parse_json_path_renamed(json_path_renamed)
            except ValueError as e:
                errors.append(f"Expression {idx}: Failed to parse jsonPathRenamed: {e}")
                continue

            file_path = ctf_path / solution / file_rel_path
            if not file_path.exists():
                errors.append(f"File not found: {file_path}")
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)

                path_parts = parse_json_path(json_path)
                _parent, _key, current_value = get_value_by_path(data, path_parts)

                if current_value != expr_orig:
                    errors.append(
                        f"Mismatch at {file_path}#{json_path}:\n"
                        f"  Expected: {expr_orig}\n"
                        f"  Actual:   {current_value}"
                    )
                    continue

                if dry_run:
                    print(f"[DRY-RUN] Would replace in {file_path}:")
                    print(f"  {json_path}: {expr_orig[:50]}... -> {expr_renamed[:50]}...")
                    skipped += 1
                else:
                    set_value_by_path(data, path_parts, expr_renamed)
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    applied += 1

            except Exception as e:
                errors.append(f"Error processing {file_path}: {e}")

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(expressions)} expressions...")

    print("\n=== Results ===")
    if dry_run:
        print(f"Would apply: {skipped} expressions")
    else:
        print(f"Applied: {applied} expressions")

    if errors:
        print(f"\n=== Errors ({len(errors)}) ===")
        for err in errors[:20]:
            try:
                print(f"  {err}")
            except UnicodeEncodeError:
                print(f"  [error with unicode characters]")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        return 0

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply expression renames to CTF files")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--output-dir", help="Output directory with expressions.json")
    parser.add_argument(
        "--ctf-root",
        help="CTF root directory (default: {output-dir}/../{app}_tr_json)",
    )
    parser.add_argument("--domain", help="Domain (e.g., mz-fr.test.cbap.ru)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or f"/tmp/cmw-transfer/{args.app}_tr"
    sys.exit(apply_expressions(args.app, output_dir, args.ctf_root, args.domain, args.dry_run))
