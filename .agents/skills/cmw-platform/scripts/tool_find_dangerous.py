#!/usr/bin/env python3
"""
Step 4: Find Dangerous Aliases

Scans ALL JSON files for aliases used in expression contexts.
Uses two-phase approach:
  Phase 1: Fast filter - find files containing expression keywords
  Phase 2: Deep scan - parse filtered files for alias patterns

Expression patterns:
  - ${alias} - variable reference
  - ->{alias} - method call on alias
  - {alias}-> - alias as method target
  - "{alias}" - string quoted alias

Usage:
    python tool_find_dangerous.py --app Volga --extract-dir /path/to/extract --output-dir /path/to/output
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


def main():
    parser = argparse.ArgumentParser(description="Step 4: Find dangerous aliases")
    parser.add_argument("--app", required=True)
    parser.add_argument("--extract-dir", default="/tmp/cmw-transfer/Volga-extract")
    parser.add_argument("--output-dir", default="/tmp/cmw-transfer/Volga-extract/Volga_tr")
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    extract_dir = Path(args.extract_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    app_dir = extract_dir / args.app
    if not app_dir.exists():
        print(f"Error: {app_dir} not found")
        return 1

    verified_file = output_dir / f"{args.app}_verified_complete.json"
    if not verified_file.exists():
        print(f"Warning: {verified_file} not found")
        print("Looking for folder verified files...")

        verified_aliases = set()
        for f in output_dir.glob(f"{args.app}_*_verified.json"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    for v in data.get("verified", []):
                        verified_aliases.add((v["type"], v["alias"]))
            except (json.JSONDecodeError, OSError):
                continue

        if not verified_aliases:
            print("Error: No verified aliases found. Run Steps 1-3 first.")
            return 1
    else:
        with open(verified_file) as f:
            verified_data = json.load(f)
        verified_aliases = {(v["type"], v["alias"]) for v in verified_data}

    print(f"=== Step 4: Find Dangerous Aliases for {args.app} ===")
    print(f"Verified aliases to check: {len(verified_aliases)}")

    state_file = output_dir / f"{args.app}_dangerous_scan_state.json"
    state = {"phase": 1, "files_processed": 0, "files_total": 0, "last_file": "", "last_updated": ""}

    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
            print(f"Resuming from state: {state.get('last_file', 'start')}")
        except (json.JSONDecodeError, OSError):
            pass

    print("Building combined patterns...")
    pattern_start = time.time()

    aliases_list = [alias for _, alias in verified_aliases]

    p1 = r"\$\{(" + "|".join(re.escape(a) for a in aliases_list) + r")\}"
    p2 = r"\->\{(" + "|".join(re.escape(a) for a in aliases_list) + r")\}"
    p3 = r"\{(" + "|".join(re.escape(a) for a in aliases_list) + r")\}->"
    p4 = r"\"" + "|".join(re.escape(a) for a in aliases_list) + r"\""

    regex1 = re.compile(p1)
    regex2 = re.compile(p2)
    regex3 = re.compile(p3)
    regex4 = re.compile(p4)

    print(f"  Patterns compiled in {time.time() - pattern_start:.1f}s")

    dangerous = set()
    patterns_found = {}
    match_count = 0

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

    class ScanResult:
        __slots__ = ("dangerous", "patterns", "matches")

        def __init__(self):
            self.dangerous = set()
            self.patterns = {}
            self.matches = 0

    def scan_file(json_file):
        result = ScanResult()
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return result

        def scan_expressions(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in EXPRESSION_KEYS and isinstance(value, str):
                        for regex, pname in [(regex1, "${%s}"), (regex2, "->{%s}"), (regex3, "{%s}->"), (regex4, "\"%s\"")]:
                            for match in regex.findall(value):
                                result.dangerous.add(match)
                                p = pname % match
                                if match not in result.patterns:
                                    result.patterns[match] = []
                                result.patterns[match].append(p)
                                result.matches += 1
                    scan_expressions(value)
            elif isinstance(obj, list):
                for item in obj:
                    scan_expressions(item)

        scan_expressions(data)
        return result

    print(f"Phase 2: Scanning {len(files_with_expressions)} files...")

    phase2_start = time.time()
    processed = 0

    batch_size = 100
    for i in range(0, len(files_with_expressions), batch_size):
        batch = files_with_expressions[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(scan_file, f): f for f in batch}
            for future in as_completed(futures):
                result = future.result()
                for alias in result.dangerous:
                    dangerous.add(alias)
                    if alias not in patterns_found:
                        patterns_found[alias] = []
                    patterns_found[alias].extend(result.patterns.get(alias, []))
                match_count += result.matches

        processed += len(batch)
        if (i + batch_size) % 200 == 0 or processed >= len(files_with_expressions):
            print(f"  Processed {processed}/{len(files_with_expressions)} files, dangerous: {len(dangerous)}")

    phase2_elapsed = time.time() - phase2_start
    print(f"  Scan complete ({phase2_elapsed:.1f}s)")

    output_data = {
        "app": args.app,
        "dangerous_aliases": sorted(list(dangerous)),
        "patterns_found": patterns_found,
        "match_count": match_count,
        "files_with_expressions": len(files_with_expressions),
        "files_scanned": len(files_with_expressions),
        "scanned_at": datetime.now().isoformat(),
        "phase1_time_seconds": phase1_elapsed,
        "phase2_time_seconds": phase2_elapsed,
    }

    output_file = output_dir / f"{args.app}_dangerous_aliases.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n=== Dangerous Aliases Found ===")
    print(f"Dangerous aliases: {len(dangerous)}")
    print(f"Pattern matches: {match_count}")
    print(f"Output: {output_file}")

    if state_file.exists():
        os.remove(state_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
