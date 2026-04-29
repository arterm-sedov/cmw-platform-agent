#!/usr/bin/env python3
"""
Step 4: Find Dangerous Aliases.

Scans ALL JSON files for aliases used in expression contexts.
Uses two-phase approach:
  Phase 1: Fast filter - find files containing expression keywords
  Phase 2: Deep scan - parse filtered files for alias patterns

Expression patterns:
  - ${alias} - variable reference
  - ->{alias} - method call on alias
  - {alias}-> - alias as method target
  - "{alias}" - string quoted alias

Output: {app}_dangerous_aliases.json with:
  - dangerous_aliases: list of alias names
  - expressions: list of {alias, jsonPathOriginal, expressionOriginal}
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(APP_DIR))


EXPRESSION_KEYS = {"Expression", "Code", "ValueExpression", "ValidationScript", "Calculation", "DefaultExpression"}


def scan_file(json_file, aliases_list, regexes, app_dir):
    """Scan a single file for dangerous aliases.
    Returns: dict of {alias: [{"jsonPathOriginal": path, "jsonPathRenamed": "", "expressionOriginal": text, "expressionRenamed": ""}]}
    """
    result = {}

    try:
        content = json_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return result

    # Check if file has expression keywords
    if not any(kw in content for kw in EXPRESSION_KEYS):
        return result

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return result

    # Get relative CTF path
    rel_path = str(json_file.relative_to(app_dir))

    def scan_expressions(obj, path=""):
        """Recursively scan for expression fields containing aliases."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if key in EXPRESSION_KEYS and isinstance(value, str):
                    # Check each alias
                    for alias in aliases_list:
                        # Use simple string check (aliases are safe - no special regex chars)
                        if alias in value:
                            if alias not in result:
                                result[alias] = []
                            result[alias].append({
                                "jsonPathOriginal": f"{rel_path}#{current_path}",
                                "jsonPathRenamed": "",
                                "expressionOriginal": value,
                                "expressionRenamed": "",
                            })
                scan_expressions(value, current_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                scan_expressions(item, f"{path}[{i}]")

    scan_expressions(data)
    return result


def main():
    parser = argparse.ArgumentParser(description="Step 4: Find dangerous aliases")
    parser.add_argument("--app", required=True)
    parser.add_argument("--extract-dir", default="/tmp/cmw-transfer/Volga-extract")
    parser.add_argument("--output-dir", default="/tmp/cmw-transfer/Volga-extract/Volga_tr")
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extract_dir = Path(args.extract_dir)
    app_dir = extract_dir / args.app
    if not app_dir.exists():
        print(f"Error: {app_dir} not found")
        return 1

    verified_file = output_dir / f"{args.app}_verified_complete.json"
    dangerous_file = output_dir / f"{args.app}_dangerous_aliases.json"

    # Load verified aliases
    if not verified_file.exists():
        print(f"Warning: {verified_file} not found")
        print("Looking for folder verified files...")

        verified_aliases = set()
        for f in output_dir.glob(f"{args.app}_*_verified.json"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                for v in data.get("verified", []):
                    alias = v.get("alias", v.get("aliasOriginal", ""))
                    if alias:
                        verified_aliases.add(alias)
            except (json.JSONDecodeError, OSError):
                continue

        if not verified_aliases:
            print("Error: No verified aliases found. Run Steps 1-3 first.")
            return 1
    else:
        with open(verified_file) as f:
            verified_data = json.load(f)

        # Handle both formats
        verified_aliases = set()
        for v in verified_data:
            alias = v.get("alias", v.get("aliasOriginal", ""))
            if alias:
                verified_aliases.add(alias)

    print(f"=== Step 4: Find Dangerous Aliases for {args.app} ===")
    print(f"Verified aliases to check: {len(verified_aliases)}")

    # Phase 1: Find files with expression content
    print("Phase 1: Finding files with expression content...")
    phase1_start = time.time()

    files_with_expressions = []
    for json_file in app_dir.rglob("*.json"):
        try:
            content = json_file.read_text(encoding="utf-8")
            if any(kw in content for kw in EXPRESSION_KEYS):
                files_with_expressions.append(json_file)
        except (OSError, UnicodeDecodeError):
            continue

    phase1_elapsed = time.time() - phase1_start
    print(f"  Found {len(files_with_expressions)} files with expression content ({phase1_elapsed:.1f}s)")

    # Phase 2: Scan files for dangerous aliases
    print(f"Phase 2: Scanning {len(files_with_expressions)} files...")

    aliases_list = sorted(verified_aliases)
    phase2_start = time.time()

    # Results: {alias: [{"jsonPathOriginal": path, "expressionOriginal": text}]}
    all_expressions = {}
    dangerous = set()
    processed = 0

    def process_batch(batch):
        batch_results = {}
        for f in batch:
            result = scan_file(f, aliases_list, None, app_dir)
            for alias, expr_list in result.items():
                if alias not in batch_results:
                    batch_results[alias] = []
                batch_results[alias].extend(expr_list)
        return batch_results

    batch_size = 50
    for i in range(0, len(files_with_expressions), batch_size):
        batch = files_with_expressions[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_batch, [f]): f for f in batch}
            for future in as_completed(futures):
                result = future.result()
                for alias, expr_list in result.items():
                    if alias not in all_expressions:
                        all_expressions[alias] = []
                    all_expressions[alias].extend(expr_list)
                    dangerous.add(alias)

        processed += len(batch)
        if (i + batch_size) % 200 == 0 or processed >= len(files_with_expressions):
            print(f"  Processed {processed}/{len(files_with_expressions)} files, dangerous: {len(dangerous)}")

    phase2_elapsed = time.time() - phase2_start
    print(f"  Scan complete ({phase2_elapsed:.1f}s)")

    # Build output
    output_data = {
        "app": args.app,
        "dangerous_aliases": sorted(list(dangerous)),
        "expressions": [],
        "match_count": sum(len(v) for v in all_expressions.values()),
        "files_with_expressions": len(files_with_expressions),
        "files_scanned": len(files_with_expressions),
        "scanned_at": datetime.now().isoformat(),
        "phase1_time_seconds": phase1_elapsed,
        "phase2_time_seconds": phase2_elapsed,
    }

    # Add expressions to output
    for alias in sorted(all_expressions.keys()):
        for expr in all_expressions[alias]:
            output_data["expressions"].append({
                "alias": alias,
                "jsonPathOriginal": expr["jsonPathOriginal"],
                "jsonPathRenamed": "",
                "expressionOriginal": expr["expressionOriginal"],
                "expressionRenamed": "",
            })

    output_file = output_dir / f"{args.app}_dangerous_aliases.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n=== Dangerous Aliases Found ===")
    print(f"Dangerous aliases: {len(dangerous)}")
    print(f"Expression matches: {output_data['match_count']}")
    print(f"Output: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
