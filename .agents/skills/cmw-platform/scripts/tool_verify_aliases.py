#!/usr/bin/env python3
"""
Step 3: Verify Aliases Per Folder

Verifies aliases from a single folder against platform cache.
Can retry on failure (specific folder).

Usage:
    python tool_verify_aliases.py --app Volga --folder RecordTemplates --output-dir /path/to/output
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(APP_DIR))


def verify_folder(folder_name: str, app_name: str, aliases_data: dict, cache: dict, output_dir: Path) -> tuple[list, list]:
    """Verify aliases for a single folder.

    Every entry looks up ALL IDs from cache by (type, alias).
    Same alias can have multiple IDs across different containers.
    """
    verified = []
    skipped = []

    TYPE_TO_PLATFORM = {
        "RoleWorkspace": "Workspace",
        "MessageTemplateProperty": "Attribute",
        "RoleConfiguration": "Role",
        "SimplePage": "Page",
    }

    for obj in aliases_data.get("aliases", []):
        obj_type = obj["type"]
        alias = obj["alias"]
        parent_template = obj.get("parent_template", "")
        platform_type = TYPE_TO_PLATFORM.get(obj_type, obj_type)

        found_ids = []

        if platform_type in cache and alias in cache[platform_type]:
            found_ids = cache[platform_type][alias].get("ids", [])

        if not found_ids and obj_type in ("OrgStructureTemplate", "RoleTemplate"):
            if obj_type == "RoleTemplate":
                candidates = [f"{app_name}_RolesCatalog", "systemSolution_RolesCatalog"]
            else:
                candidates = [f"{app_name}_OrganizationalStructure", "systemSolution_OrganizationalStructure"]
            for prefixed_alias in candidates:
                if platform_type in cache and prefixed_alias in cache[platform_type]:
                    found_ids = cache[platform_type][prefixed_alias].get("ids", [])
                    break

        if found_ids:
            obj["ids"] = found_ids
            verified.append(obj)
        else:
            skipped.append(obj)

    return verified, skipped


def main():
    parser = argparse.ArgumentParser(description="Step 3: Verify aliases per folder")
    parser.add_argument("--app", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--platform-cache", default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"/tmp/cmw-transfer/{args.app}_tr")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.platform_cache:
        cache_file = Path(args.platform_cache)
    else:
        cache_file = output_dir / f"{args.app}_platform_cache.json"

    if not cache_file.exists():
        print(f"Error: Platform cache not found at {cache_file}")
        print("Run Step 2 first: python tool_collect_platform.py --app " + args.app)
        return 1

    with open(cache_file, encoding="utf-8") as f:
        cache_data = json.load(f)

    cache = cache_data.get("cache", {})

    aliases_file = output_dir / f"{args.app}_{args.folder}_aliases.json"
    if not aliases_file.exists():
        print(f"Error: Aliases file not found at {aliases_file}")
        print(f"Run Step 1 first: python tool_extract_aliases.py --app {args.app}")
        return 1

    with open(aliases_file, encoding="utf-8") as f:
        aliases_data = json.load(f)

    output_file = output_dir / f"{args.app}_{args.folder}_verified.json"

    print(f"=== Step 3: Verify {args.app}/{args.folder} ===")

    if output_file.exists():
        try:
            with open(output_file, encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("folder") == args.folder and existing.get("verified"):
                print(f"Already verified: {len(existing['verified'])} verified, {len(existing.get('skipped', []))} skipped")
                return 0
        except (json.JSONDecodeError, OSError):
            pass

    start = time.time()
    verified, skipped = verify_folder(args.folder, args.app, aliases_data, cache, output_dir)
    elapsed = time.time() - start

    output_data = {
        "app": args.app,
        "folder": args.folder,
        "verified": verified,
        "skipped": skipped,
        "verified_count": len(verified),
        "skipped_count": len(skipped),
        "verified_at": datetime.now().isoformat(),
        "verification_time_seconds": elapsed,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Verified: {len(verified)}, Skipped: {len(skipped)} ({elapsed:.1f}s)")
    print(f"Output: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
