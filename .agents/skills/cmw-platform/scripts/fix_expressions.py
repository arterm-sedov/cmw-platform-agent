#!/usr/bin/env python3
"""
Fix _calc aliases in expressions - populates expressionRenamed for dangerous aliases.

Usage:
    python fix_expressions.py --app Volga --output-dir C:\\tmp\\cmw-transfer --dangerous-suffix _calc
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


def main(app: str, output_dir: str, dangerous_suffix: str = "_calc") -> int:
    output_path = Path(output_dir)
    domain = get_domain()

    tr_file = output_path / f"{domain}_{app}_tr.json"
    if not tr_file.exists():
        print(f"Error: Translation file not found: {tr_file}")
        return 1

    with open(tr_file, encoding="utf-8") as f:
        tr_data = json.load(f)

    fixed_count = 0

    for obj in tr_data:
        if not obj.get("expressions"):
            continue

        alias_original = obj.get("aliasOriginal", "")
        alias_renamed = obj.get("aliasRenamed", "")
        if not alias_renamed or not alias_renamed.endswith(dangerous_suffix):
            continue

        for expr in obj.get("expressions", []):
            orig_expr = expr.get("expressionOriginal", "")
            if not orig_expr:
                continue

            new_expr = orig_expr
            safe_alias = re.escape(alias_original)

            patterns = [
                (r'"' + safe_alias + r'"', f'"{alias_renamed}"'),
                (r"\$\{" + safe_alias + r"\}", f"${{{alias_renamed}}}"),
                (r"->\{" + safe_alias + r"\}", f"->{{{alias_renamed}}}"),
                (r"\{" + safe_alias + r"\}->", f"{{{alias_renamed}}}->"),
            ]

            for pattern, replacement in patterns:
                new_expr = re.sub(pattern, replacement, new_expr)

            if new_expr != orig_expr:
                expr["expressionRenamed"] = new_expr
                fixed_count += 1

    with open(tr_file, "w", encoding="utf-8") as f:
        json.dump(tr_data, f, indent=2, ensure_ascii=False)

    print(f"Fixed _calc aliases in {fixed_count} expressions")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix _calc aliases in expressions")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--output-dir", required=True, help="Path to output directory with _tr.json")
    parser.add_argument("--dangerous-suffix", default="_calc", help="Suffix for dangerous system names")
    args = parser.parse_args()
    sys.exit(main(args.app, args.output_dir, args.dangerous_suffix))