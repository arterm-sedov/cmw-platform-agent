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


def scan_file(json_file, aliases_list, regexes, extract_dir):
    """Scan a single file for dangerous aliases.
    Returns: dict of {alias: [{"jsonPathOriginal": path, "jsonPathRenamed": "", "expressionOriginal": text, "expressionRenamed": ""}]}
    """
    result = {}
    json_file_str = str(json_file)

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

    # Get relative CTF path (relative to extract_dir to include app folder prefix)
    rel_path = str(json_file.relative_to(extract_dir))

    def scan_expressions(obj, path=""):
        """Recursively scan for expression fields containing aliases."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if key in EXPRESSION_KEYS and isinstance(value, str):
                    # Check each alias with regex patterns for precise matching
                    for alias in aliases_list:
                        escaped_alias = re.escape(alias)
                        patterns = [
                            rf'\${escaped_alias}\b',    # $alias - variable
                            rf'->{escaped_alias}\b',    # ->alias - method call
                            rf'\b{escaped_alias}->',    # alias-> - object as target
                            rf'"{escaped_alias}"',      # "alias" - string literal
                        ]
                        if any(re.search(p, value) for p in patterns):
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


def save_state(output_dir: Path, app: str, phase: str, processed_count: int, dangerous: set, all_expressions: dict):
    """Save processing state for resume capability."""
    state_file = output_dir / f"{app}_dangerous_state.json"
    state = {
        "phase": phase,
        "processed_count": processed_count,
        "dangerous_aliases": list(dangerous),
        "expressions_count": sum(len(v) for v in all_expressions.values()),
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_state(output_dir: Path, app: str) -> dict | None:
    """Load previous processing state."""
    state_file = output_dir / f"{app}_dangerous_state.json"
    if state_file.exists():
        with open(state_file, encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Step 4: Find dangerous aliases")
    parser.add_argument("--app", required=True)
    parser.add_argument("--extract-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument("--batch-size", type=int, default=50, help="Files per batch")
    parser.add_argument("--force", action="store_true", help="Force restart from beginning")

    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"/tmp/cmw-transfer/{args.app}_tr")
    output_dir.mkdir(parents=True, exist_ok=True)

    extract_dir = Path(args.extract_dir or f"/tmp/cmw-transfer/{args.app}")
    app_dir = extract_dir / args.app
    if not app_dir.exists():
        print(f"Error: {app_dir} not found")
        return 1

    dangerous_file = output_dir / f"{args.app}_dangerous_aliases.json"

    # Load ALL aliases from aliases.json files (not just verified)
    all_aliases = {}  # {alias: {"type": "...", "verified": bool}}
    verified_aliases = set()

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
                        "verified": False
                    }
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue

    # Mark verified aliases (those with ids in verified.json)
    for f in output_dir.glob(f"{args.app}_*_verified.json"):
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            for v in data.get("verified", []):
                alias = v.get("alias", v.get("aliasOriginal", ""))
                if alias and alias in all_aliases:
                    all_aliases[alias]["verified"] = True
                    verified_aliases.add(alias)
                elif alias:
                    verified_aliases.add(alias)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue

    if not all_aliases:
        print("Error: No aliases found. Run Steps 1-2 first.")
        return 1

    print(f"=== Step 4: Find Dangerous Aliases for {args.app} ===")
    print(f"Total aliases to check: {len(all_aliases)}")
    print(f"Verified: {len(verified_aliases)}, Non-verified: {len(all_aliases) - len(verified_aliases)}")

    # Check for cached files list (for resume with cached Phase 1)
    files_list_cache = output_dir / f"{args.app}_files_with_expressions.json"
    if args.resume and files_list_cache.exists():
        print("Loading cached files list...")
        try:
            with open(files_list_cache, encoding="utf-8") as f:
                files_data = json.load(f)
            files_with_expressions = [Path(d["jsonPath"]) for d in files_data]
            print(f"  Loaded {len(files_with_expressions)} files from cache")
        except Exception as e:
            print(f"  Warning: Could not load cache: {e}")
            files_with_expressions = None
    else:
        files_with_expressions = None

    # Phase 1: Find files with expression content (only if not loaded from cache)
    phase1_elapsed = 0.0  # Default if using cached files list
    if files_with_expressions is None:
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

        # Save files list for future resume
        print("  Saving files list cache...")
        files_data = [{"index": i, "jsonPath": str(f)} for i, f in enumerate(files_with_expressions)]
        with open(files_list_cache, "w", encoding="utf-8") as f:
            json.dump(files_data, f, indent=2)
        print("  Files list cache saved")

    # Check for resume state
    start_index = 0
    if args.resume and not args.force:
        prev_state = load_state(output_dir, args.app)
        if prev_state:
            start_index = prev_state.get("processed_count", 0)
            print(f"Resuming from batch {start_index}")

    # Phase 2: Scan files for dangerous aliases
    print(f"Phase 2: Scanning {len(files_with_expressions)} files (starting from {start_index})...")

    aliases_list = sorted(all_aliases.keys())
    phase2_start = time.time()

    # Results: {alias: [{"jsonPathOriginal": path, "expressionOriginal": text}]}
    all_expressions = {}
    dangerous = set()
    processed = 0

    # Load previous results if resuming
    if args.resume and not args.force:
        prev_dangerous_file = output_dir / f"{args.app}_dangerous_aliases.json"
        if prev_dangerous_file.exists():
            try:
                with open(prev_dangerous_file, encoding="utf-8") as f:
                    prev_data = json.load(f)
                
                # Fast path: just copy previous dangerous and expressions directly
                # No need to re-match - trust the previous results
                dangerous = set(prev_data.get("dangerous_aliases", []))
                
                # Reconstruct all_expressions from previous expressions
                for expr in prev_data.get("expressions", []):
                    # Try to find the alias from the expression text
                    # This is slower but necessary to know which alias each expression belongs to
                    expr_text = expr.get("expressionOriginal", "")
                    alias = None
                    # Quick check - if expression contains $, try to extract alias
                    if '$' in expr_text:
                        for a in verified_aliases:
                            if f'${a}' in expr_text or f'"{a}"' in expr_text or f'->{a}' in expr_text:
                                alias = a
                                break
                    if alias:
                        if alias not in all_expressions:
                            all_expressions[alias] = []
                        all_expressions[alias].append(expr)
                
                print(f"  Loaded previous results: {len(dangerous)} dangerous aliases, {len(all_expressions)} expressions")
            except Exception as e:
                print(f"  ERROR: Could not load previous results: {e}")
                # Reset to empty - will process from beginning
                dangerous = set()
                all_expressions = {}

    def process_batch(batch):
        batch_results = {}
        for f in batch:
            result = scan_file(f, aliases_list, None, extract_dir)
            for alias, expr_list in result.items():
                if alias not in batch_results:
                    batch_results[alias] = []
                batch_results[alias].extend(expr_list)
        return batch_results

    batch_size = args.batch_size
    files_to_process = files_with_expressions[start_index:]
    total_to_process = len(files_to_process)

    print(f"  Processing {total_to_process} files in batches of {batch_size}")

    def _save_incremental(dangerous_set, expressions_dict):
        """Save dangerous_aliases.json incrementally after each file."""
        dangerous_file = output_dir / f"{args.app}_dangerous_aliases.json"
        incremental_data = {
            "app": args.app,
            "dangerous_aliases": sorted(list(dangerous_set)),
            "expressions": [],
            "match_count": sum(len(v) for v in expressions_dict.values()),
            "files_with_expressions": len(files_with_expressions),
            "files_scanned": len(files_with_expressions),
            "scanned_at": datetime.now().isoformat(),
            "phase1_time_seconds": phase1_elapsed,
            "phase2_time_seconds": time.time() - phase2_start,
            "incremental": True,
        }
        for alias in sorted(expressions_dict.keys()):
            for expr in expressions_dict[alias]:
                incremental_data["expressions"].append({
                    "jsonPathOriginal": expr["jsonPathOriginal"],
                    "jsonPathRenamed": "",
                    "expressionOriginal": expr["expressionOriginal"],
                    "expressionRenamed": "",
                })
        with open(dangerous_file, "w", encoding="utf-8") as f:
            json.dump(incremental_data, f, indent=2, ensure_ascii=False)

    processed = start_index  # инициализация до цикла
    
    for i in range(0, total_to_process, batch_size):
        batch = files_to_process[i:i + batch_size]
        
        # Sequential processing (no parallelism)
        for f in batch:
            result = process_batch([f])
            
            for alias, expr_list in result.items():
                if alias not in all_expressions:
                    all_expressions[alias] = []
                all_expressions[alias].extend(expr_list)
                dangerous.add(alias)
            
            # Save state after each file for resume capability
            processed += 1
            save_state(output_dir, args.app, "phase2", processed, dangerous, all_expressions)
            
            # Always save dangerous_aliases.json incrementally after each file
            _save_incremental(dangerous, all_expressions)

        print(f"  Processed {processed}/{len(files_with_expressions)} files, dangerous: {len(dangerous)}")

    phase2_elapsed = time.time() - phase2_start
    print(f"  Scan complete ({phase2_elapsed:.1f}s)")

    # Build output with verified status
    dangerous_list = []
    for alias in sorted(dangerous):
        dangerous_list.append({
            "alias": alias,
            "type": all_aliases.get(alias, {}).get("type", "Unknown"),
            "verified": all_aliases.get(alias, {}).get("verified", False),
        })

    output_data = {
        "app": args.app,
        "dangerous_aliases": dangerous_list,
        "expressions": [],
        "match_count": sum(len(v) for v in all_expressions.values()),
        "files_with_expressions": len(files_with_expressions),
        "files_scanned": len(files_with_expressions),
        "scanned_at": datetime.now().isoformat(),
        "phase1_time_seconds": phase1_elapsed,
        "phase2_time_seconds": phase2_elapsed,
        "total_aliases": len(all_aliases),
        "verified_count": len(verified_aliases),
    }

    # Add expressions to output with alias and verified status
    for alias in sorted(all_expressions.keys()):
        is_verified = all_aliases.get(alias, {}).get("verified", False)
        alias_type = all_aliases.get(alias, {}).get("type", "Unknown")
        for expr in all_expressions[alias]:
            output_data["expressions"].append({
                "alias": alias,
                "alias_type": alias_type,
                "verified": is_verified,
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
    print(f"  Verified: {sum(1 for d in dangerous_list if d['verified'])}")
    print(f"  Non-verified: {sum(1 for d in dangerous_list if not d['verified'])}")
    print(f"Expression matches: {output_data['match_count']}")
    print(f"Output: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
