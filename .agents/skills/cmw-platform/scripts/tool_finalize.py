#!/usr/bin/env python3
"""
Step 5: Finalize

Merges all folder verified files into complete output files.
Sets aliasLocked flags:
  - true: matches skip pattern, has displayName, NOT dangerous
  - false: normal OR matches pattern but dangerous

Usage:
    python tool_finalize.py --app Volga --output-dir /path/to/output
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(APP_DIR))


def main():
    parser = argparse.ArgumentParser(description="Step 5: Finalize verified aliases")
    parser.add_argument("--app", required=True)
    parser.add_argument("--output-dir", default="/tmp/cmw-transfer/Volga-extract/Volga_tr")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Step 5: Finalize for {args.app} ===")

    verified = []
    skipped = []

    verified_files = sorted(output_dir.glob(f"{args.app}_*_verified.json"))
    print(f"Found {len(verified_files)} folder verified files")

    for vf in verified_files:
        try:
            with open(vf) as f:
                data = json.load(f)
            folder = data.get("folder", vf.stem.replace(f"{args.app}_", "").replace("_verified", ""))
            print(f"  {folder}: {data.get('verified_count', 0)} verified, {data.get('skipped_count', 0)} skipped")
            verified.extend(data.get("verified", []))
            skipped.extend(data.get("skipped", []))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: Failed to load {vf}: {e}")

    dangerous_file = output_dir / f"{args.app}_dangerous_aliases.json"
    dangerous = set()
    if dangerous_file.exists():
        try:
            with open(dangerous_file) as f:
                dangerous_data = json.load(f)
            dangerous = set(dangerous_data.get("dangerous_aliases", []))
            print(f"Loaded {len(dangerous)} dangerous aliases")
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: Failed to load dangerous file: {e}")
    else:
        print(f"Warning: Dangerous aliases file not found at {dangerous_file}")

    print(f"\nUpdating aliasLocked flags...")

    updated_locked = 0
    updated_unlocked = 0

    for obj in verified:
        if obj.get("aliasLocked") and obj["alias"] in dangerous:
            obj["aliasLocked"] = False
            updated_unlocked += 1
        elif obj.get("aliasLocked") and obj["alias"] not in dangerous:
            pass
        else:
            pass

    verified_locked = [v for v in verified if v.get("aliasLocked")]
    verified_normal = [v for v in verified if not v.get("aliasLocked")]

    skipped_locked = [s for s in skipped if s.get("aliasLocked")]
    skipped_normal = [s for s in skipped if not s.get("aliasLocked")]

    print(f"Verified: {len(verified)} total")
    print(f"  aliasLocked=true (safe to skip): {len(verified_locked)}")
    print(f"  aliasLocked=false (will rename): {len(verified_normal)}")

    print(f"\nSkipped (not in platform): {len(skipped)} total")
    print(f"  aliasLocked=true (matched pattern): {len(skipped_locked)}")
    print(f"  aliasLocked=false (no match): {len(skipped_normal)}")

    verified_file = output_dir / f"{args.app}_verified_complete.json"
    with open(verified_file, "w", encoding="utf-8") as f:
        json.dump(verified, f, indent=2, ensure_ascii=False)

    skipped_locked_file = output_dir / f"{args.app}_skipped_locked.json"
    with open(skipped_locked_file, "w", encoding="utf-8") as f:
        json.dump(skipped_locked, f, indent=2, ensure_ascii=False)

    skipped_complete_file = output_dir / f"{args.app}_skipped_complete.json"
    with open(skipped_complete_file, "w", encoding="utf-8") as f:
        json.dump(skipped_normal, f, indent=2, ensure_ascii=False)

    print(f"\n=== Final Output Files ===")
    print(f"Verified: {verified_file} ({len(verified)} objects)")
    print(f"Skipped locked: {skipped_locked_file} ({len(skipped_locked)} objects)")
    print(f"Skipped complete: {skipped_complete_file} ({len(skipped_normal)} objects)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
