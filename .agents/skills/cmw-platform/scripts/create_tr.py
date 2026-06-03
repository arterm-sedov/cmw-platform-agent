#!/usr/bin/env python3
"""
Create _tr copy from original aliases file with _calc suffix fix in expressions.

Usage:
    python create_tr.py --app Volga --output-dir C:\\tmp\\cmw-transfer
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


def fix_expressions_in_memory(objects: list, dangerous_suffix: str = "_calc") -> int:
    """Fix _calc aliases in expression fields (in memory)."""
    fixed_count = 0

    for obj in objects:
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

    return fixed_count


def main(app: str, output_dir: str, dangerous_suffix: str = "_calc") -> int:
    output_path = Path(output_dir)
    domain = get_domain()

    aliases_file = output_path / f"{domain}_{app}_aliases.json"
    tr_file = output_path / f"{domain}_{app}_tr.json"

    if not aliases_file.exists():
        print(f"Error: Aliases file not found: {aliases_file}")
        return 1

    with open(aliases_file, encoding="utf-8") as f:
        tr_data = json.load(f)

    fixed = fix_expressions_in_memory(tr_data, dangerous_suffix)

    with open(tr_file, "w", encoding="utf-8") as f:
        json.dump(tr_data, f, indent=2, ensure_ascii=False)

    print(f"Created {tr_file}")
    print(f"Fixed _calc aliases in {fixed} expressions")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create _tr copy from original aliases file")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--output-dir", required=True, help="Path to output directory")
    parser.add_argument("--dangerous-suffix", default="_calc", help="Suffix for dangerous system names")
    args = parser.parse_args()
    sys.exit(main(args.app, args.output_dir, args.dangerous_suffix))