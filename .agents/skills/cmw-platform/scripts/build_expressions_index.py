#!/usr/bin/env python3
"""
Build expressions index from _tr.json.

Creates {domain}_{app}_expressions.json with unique expressions and their metadata.

Usage:
    python build_expressions_index.py --app Volga --output-dir C:\\tmp\\cmw-transfer
"""
import argparse
import json
import sys
from collections import defaultdict
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


def build_expressions_index(tr_data: list[dict]) -> list[dict]:
    """Build expressions index from _tr.json data.

    Args:
        tr_data: List of alias objects from _tr.json

    Returns:
        List of expression objects following expressions_schema.json
    """
    expressions_map: dict[str, dict] = defaultdict(lambda: {
        "aliases": [],
        "jsonPathOriginal": [],
        "jsonPathRenamed": [],
    })

    seen_aliases: dict[str, set[str]] = defaultdict(set)
    seen_paths_orig: dict[str, set[str]] = defaultdict(set)
    seen_paths_renamed: dict[str, set[str]] = defaultdict(set)

    for obj in tr_data:
        alias_orig = obj.get("aliasOriginal", "")
        alias_renamed = obj.get("aliasRenamed", "")

        if not alias_orig:
            continue

        for expr in obj.get("expressions", []):
            expr_orig = expr.get("expressionOriginal", "")
            if not expr_orig:
                continue

            key = expr_orig

            if alias_orig and alias_renamed:
                if alias_orig not in seen_aliases[key]:
                    seen_aliases[key].add(alias_orig)
                    expr_data = expressions_map[key]
                    existing = [a for a in expr_data["aliases"] if a["aliasOriginal"] == alias_orig]
                    if not existing:
                        expr_data["aliases"].append({
                            "aliasOriginal": alias_orig,
                            "aliasRenamed": alias_renamed,
                        })

            path_orig = expr.get("jsonPathOriginal", "")
            if path_orig and path_orig not in seen_paths_orig[key]:
                seen_paths_orig[key].add(path_orig)
                expressions_map[key]["jsonPathOriginal"].append(path_orig)

            path_renamed = expr.get("jsonPathRenamed", "")
            if path_renamed and path_renamed not in seen_paths_renamed[key]:
                seen_paths_renamed[key].add(path_renamed)
                expressions_map[key]["jsonPathRenamed"].append(path_renamed)

    result = []
    for expr_orig, data in sorted(expressions_map.items()):
        result.append({
            "expression_original": expr_orig,
            "expression_renamed": "",
            "aliases": data["aliases"],
            "jsonPathOriginal": data["jsonPathOriginal"],
            "jsonPathRenamed": data["jsonPathRenamed"],
        })

    return result


def main(app: str, output_dir: str) -> int:
    output_path = Path(output_dir)
    domain = get_domain()

    tr_file = output_path / f"{domain}_{app}_tr.json"
    if not tr_file.exists():
        print(f"Error: Translation file not found: {tr_file}")
        return 1

    with open(tr_file, encoding="utf-8") as f:
        tr_data = json.load(f)

    print(f"Loaded {len(tr_data)} alias entries from {tr_file.name}")

    expressions = build_expressions_index(tr_data)
    print(f"Found {len(expressions)} unique expressions")

    output_file = output_path / f"{domain}_{app}_expressions.json"
    output_data = {"objects": expressions}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Written {len(expressions)} expressions to {output_file.name}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build expressions index from _tr.json")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--output-dir", required=True, help="Path to output directory with _tr.json")
    args = parser.parse_args()
    sys.exit(main(args.app, args.output_dir))