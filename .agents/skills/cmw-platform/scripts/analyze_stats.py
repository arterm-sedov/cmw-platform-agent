#!/usr/bin/env python3
"""
Statistical analysis of numeric attributes.

Usage:
    python analyze_stats.py --app <app> --template <template> --attr <attribute> --top N
    python analyze_stats.py --app Volga --template Schetaioplaty --attr DebtSumm --top 10
"""
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.templates_tools.tool_list_records import list_template_records


def analyze_stats(
    app_name: str,
    template_name: str,
    attr_name: str,
    group_by: str | None = None,
    limit: int = 100,
    top_n: int = 10,
):
    values = []
    offset = 0

    print(f"Analyzing {app_name}.{template_name}.{attr_name}")

    attrs_to_fetch = ["id", attr_name]
    if group_by:
        attrs_to_fetch.append(group_by)

    while True:
        result = list_template_records.invoke({
            "application_system_name": app_name,
            "template_system_name": template_name,
            "attributes": attrs_to_fetch,
            "limit": limit,
            "offset": offset
        })

        if not result["success"]:
            print(f"Error at offset {offset}")
            break

        records = result["data"]
        if not records:
            break

        for rec in records:
            val = rec.get(attr_name)
            if val is not None:
                group_val = rec.get(group_by) if group_by else None
                values.append((rec.get("id"), val, group_val))

        if len(records) < limit:
            break
        offset += limit

    if not values:
        print("No values found")
        return

    numeric_vals = [v[1] for v in values if v[1] is not None]

    print(f"Total records: {len(values)}")
    print(f"Records with {attr_name}: {len(numeric_vals)}")

    if numeric_vals:
        print(f"Min: {min(numeric_vals)}")
        print(f"Max: {max(numeric_vals)}")
        print(f"Sum: {sum(numeric_vals)}")
        print(f"Avg: {sum(numeric_vals) / len(numeric_vals):.2f}")

    if top_n > 0:
        print(f"\nTop {top_n} by {attr_name}:")
        if group_by:
            sorted_vals = sorted(values, key=lambda x: x[1] or 0, reverse=True)
        else:
            sorted_vals = sorted(values, key=lambda x: x[1] or 0, reverse=True)

        for rec_id, val, group_val in sorted_vals[:top_n]:
            group_str = f" | {group_by}: {group_val}" if group_by else ""
            print(f"  ID: {rec_id} | {attr_name}: {val}{group_str}")

    return values


def main():
    parser = argparse.ArgumentParser(description="Analyze numeric attribute stats")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--template", required=True, help="Template system name")
    parser.add_argument("--attr", required=True, help="Attribute to analyze")
    parser.add_argument("--group-by", help="Optional attribute to group by")
    parser.add_argument("--limit", type=int, default=100, help="Page size")
    parser.add_argument("--top", type=int, default=10, help="Show top N results")
    args = parser.parse_args()

    analyze_stats(
        args.app,
        args.template,
        args.attr,
        args.group_by,
        args.limit,
        args.top
    )


if __name__ == "__main__":
    main()
