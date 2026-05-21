#!/usr/bin/env python3
"""
List non-verified aliases grouped by type.

Reads aliases from *aliases.json files, compares with *verified.json files,
and outputs aliases that were NOT FOUND in platform (in verified.json but without ids).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

APP_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(APP_DIR))


def main():
    parser = argparse.ArgumentParser(description="List non-verified aliases by type")
    parser.add_argument("--app", required=True)
    parser.add_argument("--output-dir", default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"/tmp/cmw-transfer/{args.app}_tr")

    print(f"=== List Non-Verified Aliases for {args.app} ===")

    all_aliases = {}  # {alias: {"type": "...", "jsonPath": "..."}}
    verified_with_ids = set()  # aliases with ids (found in platform)
    verified_without_ids = set()  # aliases without ids (not found in platform)

    for f in output_dir.glob(f"{args.app}_*_aliases.json"):
        if f.name == f"{args.app}_dangerous_aliases.json":
            continue
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            for a in data.get("aliases", []):
                alias = a.get("alias", a.get("aliasOriginal", ""))
                if alias:
                    all_aliases[alias] = {
                        "type": a.get("type", "Unknown"),
                        "jsonPath": a.get("jsonPathOriginal", ""),
                    }
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue

    for f in output_dir.glob(f"{args.app}_*_verified.json"):
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            for v in data.get("verified", []):
                alias = v.get("alias", v.get("aliasOriginal", ""))
                if alias:
                    ids = v.get("ids", [])
                    if ids:
                        verified_with_ids.add(alias)
                    else:
                        verified_without_ids.add(alias)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue

    non_verified = {a: info for a, info in all_aliases.items() 
                 if a in verified_without_ids and info["type"] != "FormComponent"}

    by_type = defaultdict(list)
    for alias, info in non_verified.items():
        by_type[info["type"]].append({
            "alias": alias,
            "type": info["type"],
            "jsonPath": info["jsonPath"],
        })

    output_data = {
        "app": args.app,
        "total_aliases": len(all_aliases),
        "verified_with_ids": len(verified_with_ids),
        "verified_without_ids": len(verified_without_ids),
        "non_verified_count": len(non_verified),
        "by_type": dict(by_type),
    }

    output_file = output_dir / f"{args.app}_non_verified.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Total aliases: {len(all_aliases)}")
    print(f"Verified (with ids): {len(verified_with_ids)}")
    print(f"Verified (without ids - NOT FOUND in platform): {len(verified_without_ids)}")
    print(f"Non-verified (not in verified.json): {len(non_verified)}")
    print(f"\nBy type (NOT FOUND in platform):")
    for t, items in sorted(by_type.items()):
        print(f"  {t}: {len(items)}")

    print(f"\nOutput: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())