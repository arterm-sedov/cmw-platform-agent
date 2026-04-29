#!/usr/bin/env python3
"""
Step 2: Collect Platform Data (Parallel)

Queries ALL types from the platform in parallel (ThreadPoolExecutor, 8 workers).
Single API call per type but concurrent execution for speed.

Usage:
    python tool_collect_platform.py --app Volga --output-dir /path/to/output
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

APP_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(APP_DIR))

from tools.applications_tools.tool_get_ontology_objects import get_ontology_objects


def query_type(app_name: str, obj_type: str, max_count: int = 5000) -> tuple[str, dict]:
    """Query platform for a single type. Returns (type, alias_to_id_map)."""
    result = get_ontology_objects.invoke({
        "application_system_name": app_name,
        "types": [obj_type],
        "parameter": "alias",
        "min_count": 1,
        "max_count": max_count,
    })

    alias_map = {}
    if result.get("success") and result.get("data"):
        for obj in result["data"]:
            alias = obj.get("systemName", "")
            obj_id = obj.get("id", "")
            if alias and obj_id:
                alias_map[alias] = {"id": obj_id, "systemName": alias}

    return obj_type, alias_map


def main():
    parser = argparse.ArgumentParser(description="Step 2: Collect platform data in parallel")
    parser.add_argument("--app", required=True)
    parser.add_argument("--output-dir", default="/tmp/cmw-transfer/Volga-extract/Volga_tr")
    parser.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{args.app}_platform_cache.json"

    print(f"=== Step 2: Collect Platform Data for {args.app} ===")
    print(f"Using {args.workers} parallel workers")

    TYPE_TO_PLATFORM = {
        "RoleWorkspace": "Workspace",
        "MessageTemplateProperty": "Attribute",
        "RoleConfiguration": "Role",
        "SimplePage": "Page",
    }

    folder_to_type = {
        "AccountTemplates": "AccountTemplate",
        "Carts": "Cart",
        "MessageTemplates": "MessageTemplate",
        "OrgStructureTemplates": "OrgStructureTemplate",
        "Pages": "Page",
        "ProcessTemplates": "ProcessTemplate",
        "RecordTemplates": "RecordTemplate",
        "RoleTemplates": "RoleTemplate",
        "Roles": "Role",
        "Routes": "Routes",
        "Streams": "Stream",
        "Triggers": "Trigger",
        "WidgetConfigs": "WidgetConfig",
        "Workspaces": "Workspace",
    }

    unique_types = set(TYPE_TO_PLATFORM.keys())
    for t in TYPE_TO_PLATFORM.values():
        unique_types.add(t)

    aliases_files = list(output_dir.glob(f"{args.app}_*_aliases.json"))
    for af in aliases_files:
        try:
            with open(af) as f:
                data = json.load(f)
            for obj in data.get("aliases", []):
                unique_types.add(obj["type"])
        except (json.JSONDecodeError, OSError):
            continue

    types_list = sorted(unique_types)

    print(f"Querying {len(types_list)} types: {types_list}")

    cache = {}
    start = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(query_type, args.app, t): t for t in types_list}

        for future in as_completed(futures):
            obj_type = futures[future]
            try:
                t, alias_map = future.result()
                cache[t] = alias_map
                completed += 1
                print(f"  [{completed}/{len(types_list)}] {t}: {len(alias_map)} objects")
            except Exception as e:
                print(f"  [{completed}/{len(types_list)}] {obj_type}: ERROR - {e}")
                cache[obj_type] = {}
                completed += 1

    elapsed = time.time() - start

    output_data = {
        "app": args.app,
        "types": types_list,
        "cache": cache,
        "collected_at": datetime.now().isoformat(),
        "collection_time_seconds": elapsed,
        "workers": args.workers
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    total_objects = sum(len(v) for v in cache.values())

    print(f"\n=== Collection Complete ===")
    print(f"Total types: {len(cache)}")
    print(f"Total objects: {total_objects}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Output: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
