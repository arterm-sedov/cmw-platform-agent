#!/usr/bin/env python3
"""
Fix _calc suffix for dangerous aliases.

Adds _calc suffix to aliasRenamed for:
1. Aliases with non-empty expressions
2. Safe aliases with same aliasOriginal as dangerous ones

Usage:
    python fix_calc_suffix.py --app Volga --output-dir C:\\tmp\\cmw-transfer
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


def main(app: str, output_dir: str, domain: str | None = None) -> int:
    output_path = Path(output_dir)
    if not domain:
        domain = get_domain()

    tr_file = output_path / f"{domain}_{app}_tr.json"
    if not tr_file.exists():
        print(f"Error: Translation file not found: {tr_file}")
        return 1

    with open(tr_file, encoding="utf-8") as f:
        tr_data = json.load(f)

    dangerous_aliases = {obj.get("aliasOriginal") for obj in tr_data if obj.get("expressions")}

    print(f"Loaded {len(tr_data)} entries")
    print(f"Found {len(dangerous_aliases)} dangerous aliasOriginal values")

    updated_count = 0

    for obj in tr_data:
        alias_original = obj.get("aliasOriginal", "")
        alias_renamed = obj.get("aliasRenamed", "")
        expressions = obj.get("expressions", [])

        is_dangerous = bool(expressions) or alias_original in dangerous_aliases

        if not is_dangerous:
            continue

        if alias_original in dangerous_aliases:
            if not alias_renamed:
                obj["aliasRenamed"] = alias_original + "_calc"
                updated_count += 1
            elif not alias_renamed.endswith("_calc"):
                obj["aliasRenamed"] = alias_renamed + "_calc"
                updated_count += 1

    print(f"Updated {updated_count} entries with _calc suffix")

    with open(tr_file, "w", encoding="utf-8") as f:
        json.dump(tr_data, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {tr_file}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix _calc suffix for dangerous aliases")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--output-dir", default=None, help="Path to output directory with _tr.json")
    parser.add_argument("--domain", default=None, help="Domain (e.g., mz-fr.test.cbap.ru)")
    args = parser.parse_args()

    output_dir = args.output_dir or "/tmp/cmw-transfer"
    sys.exit(main(args.app, output_dir, args.domain))