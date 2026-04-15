#!/usr/bin/env python3
"""
Paginated query with in-code filtering for complex conditions.

Usage:
    python query_with_filter.py --app <app> --template <template> --filter-attr <attr> --filter-op gt --filter-value 0
    python query_with_filter.py --app Volga --template Arendator --filter-attr Debt --filter-op gt --filter-value 0
"""
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.templates_tools.tool_list_records import list_template_records


def query_with_filter(
    app_name: str,
    template_name: str,
    filter_attr: str,
    filter_op: str = "gt",
    filter_value: float = 0,
    attributes: list[str] | None = None,
    limit: int = 100,
):
    if attributes is None:
        attributes = ["id", filter_attr]

    all_matching = []
    offset = 0

    print(f"Querying {app_name}.{template_name} where {filter_attr} {filter_op} {filter_value}")

    while True:
        result = list_template_records.invoke({
            "application_system_name": app_name,
            "template_system_name": template_name,
            "attributes": attributes,
            "limit": limit,
            "offset": offset
        })

        if not result["success"]:
            print(f"Error at offset {offset}: {result.get('error')}")
            break

        records = result["data"]
        if not records:
            break

        for rec in records:
            val = rec.get(filter_attr)
            matches = False

            if filter_op == "gt" and val is not None and val > filter_value:
                matches = True
            elif filter_op == "lt" and val is not None and val < filter_value:
                matches = True
            elif filter_op == "eq" and val == filter_value:
                matches = True
            elif filter_op == "ne" and val != filter_value:
                matches = True
            elif filter_op == "ge" and val is not None and val >= filter_value:
                matches = True
            elif filter_op == "le" and val is not None and val <= filter_value:
                matches = True

            if matches:
                all_matching.append(rec)

        print(f"Offset {offset}: checked {len(records)}, total matches: {len(all_matching)}")

        if len(records) < limit:
            break
        offset += limit

    print(f"\n=== Result: {len(all_matching)} matching records ===")
    for rec in all_matching[:20]:
        print(f"  ID: {rec.get('id')} | {filter_attr}: {rec.get(filter_attr)}")

    return all_matching


def main():
    parser = argparse.ArgumentParser(description="Query with in-code filtering")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--template", required=True, help="Template system name")
    parser.add_argument("--filter-attr", required=True, help="Attribute to filter on")
    parser.add_argument("--filter-op", default="gt", choices=["gt", "lt", "eq", "ne", "ge", "le"], help="Filter operator")
    parser.add_argument("--filter-value", type=float, default=0, help="Filter value")
    parser.add_argument("--limit", type=int, default=100, help="Page size")
    args = parser.parse_args()

    query_with_filter(
        args.app,
        args.template,
        args.filter_attr,
        args.filter_op,
        args.filter_value,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
