#!/usr/bin/env python3
"""
Explore multiple templates in an application.

Usage:
    python explore_templates.py --app <application_system_name> --templates Template1,Template2
    python explore_templates.py --app Volga --templates Arendator,RentLots
"""
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.templates_tools.tool_list_attributes import list_attributes
from tools.templates_tools.tool_list_records import list_template_records


def explore_templates(app_name: str, template_names: list[str], limit: int = 2):
    results = {}
    for tmpl in template_names:
        print(f"\n=== {tmpl} ===")

        attrs = list_attributes.invoke({
            "application_system_name": app_name,
            "template_system_name": tmpl
        })

        if attrs.get("success"):
            attrs_data = attrs.get("data", [])[:8]
            results[f"{tmpl}_attrs"] = attrs_data
            for a in attrs_data:
                sys_name = a.get("Attribute system name", "?")
                attr_type = a.get("Attribute type", "?")
                name = a.get("Name", "")
                print(f"  - {sys_name} ({attr_type}): {name}")
        else:
            print(f"  Attrs error: {attrs.get('error')}")
            results[f"{tmpl}_attrs"] = None

        records = list_template_records.invoke({
            "application_system_name": app_name,
            "template_system_name": tmpl,
            "limit": limit
        })

        if records.get("success"):
            data = records.get("data", [])
            results[f"{tmpl}_records"] = data
            print(f"  Records: {len(data)} (showing {limit})")
            for r in data[:limit]:
                print(f"    {json.dumps(r, ensure_ascii=False)[:200]}")
        else:
            print(f"  Records error: {records.get('error')}")
            results[f"{tmpl}_records"] = None

    return results


def main():
    parser = argparse.ArgumentParser(description="Explore templates in an application")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--templates", required=True, help="Comma-separated template names")
    parser.add_argument("--limit", type=int, default=2, help="Records per template (default: 2)")
    args = parser.parse_args()

    templates = [t.strip() for t in args.templates.split(",")]
    explore_templates(args.app, templates, args.limit)


if __name__ == "__main__":
    main()
