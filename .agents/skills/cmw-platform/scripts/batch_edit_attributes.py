#!/usr/bin/env python3
"""
Batch edit template attributes using the proper tool layer.

Handles each attribute type correctly through the appropriate tool,
preserving type-specific fields. One script call vs many tool calls = token savings.

Usage:
    # Dry-run (preview changes)
    python batch_edit_attributes.py --app FacilityManagement --template Buildings --mapping edits.json

    # Execute changes
    python batch_edit_attributes.py --app FacilityManagement --template Buildings --mapping edits.json --execute

Mapping format (JSON):
{
    "attributes": {
        "AttrSystemName": {
            "name": "New Name",
            "description": "New Description",
            "isMandatory": true,
            "displayFormat": "PlainText"
        }
    }
}

Any field from the attribute schema can be included. Fields not provided are preserved.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv = __import__("dotenv", fromlist=["load_dotenv"]).load_dotenv
load_dotenv(REPO_ROOT / ".env")


TYPE_TO_TOOL = {
    "Decimal": ("tools.attributes_tools.tools_decimal_attribute", "edit_or_create_numeric_attribute"),
    "Enum": ("tools.attributes_tools.tools_enum_attribute", "edit_or_create_enum_attribute"),
    "Record": ("tools.attributes_tools.tools_record_attribute", "edit_or_create_record_attribute"),
    "DateTime": ("tools.attributes_tools.tools_datetime_attribute", "edit_or_create_date_time_attribute"),
    "Document": ("tools.attributes_tools.tools_document_attribute", "edit_or_create_document_attribute"),
    "Image": ("tools.attributes_tools.tools_image_attribute", "edit_or_create_image_attribute"),
    "Duration": ("tools.attributes_tools.tools_duration_attribute", "edit_or_create_duration_attribute"),
    "Account": ("tools.attributes_tools.tools_account_attribute", "edit_or_create_account_attribute"),
    "String": ("tools.attributes_tools.tools_text_attribute", "edit_or_create_text_attribute"),
    "Text": ("tools.attributes_tools.tools_text_attribute", "edit_or_create_text_attribute"),
}


def get_attribute_type(app: str, template: str, attr_alias: str) -> str | None:
    """Get the type of an attribute."""
    import base64

    import requests

    base_url = os.environ.get("CMW_BASE_URL", "").rstrip("/")
    login = os.environ.get("CMW_LOGIN", "")
    password = os.environ.get("CMW_PASSWORD", "")

    if not base_url or not login or not password:
        return None

    creds = base64.b64encode(f"{login}:{password}".encode()).decode()
    headers = {"Authorization": f"Basic {creds}", "Content-Type": "application/json"}

    try:
        resp = requests.get(
            f"{base_url}/webapi/Attribute/List/Template@{app}.{template}",
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 200 and resp.json().get("response"):
            for a in resp.json()["response"]:
                if a.get("globalAlias", {}).get("alias") == attr_alias:
                    return a.get("type")
    except Exception:
        pass

    return None


def edit_attribute_via_tool(
    app: str, template: str, attr_alias: str, attr_type: str, changes: dict[str, Any]
) -> dict[str, Any]:
    """Edit an attribute using the appropriate tool."""
    tool_info = TYPE_TO_TOOL.get(attr_type)
    if not tool_info:
        return {"success": False, "error": f"No tool for type: {attr_type}"}

    module_path, tool_name = tool_info

    try:
        module = __import__(module_path, fromlist=[tool_name])
        tool = getattr(module, tool_name)
    except (ImportError, AttributeError) as e:
        return {"success": False, "error": f"Failed to load tool {tool_name}: {e}"}

    kwargs = {
        "operation": "edit",
        "system_name": attr_alias,
        "application_system_name": app,
        "template_system_name": template,
    }
    for key, value in changes.items():
        if key not in ("system_name", "application_system_name", "template_system_name", "operation"):
            kwargs[key] = value

    try:
        result = tool.invoke(kwargs)
        return result if isinstance(result, dict) else {"success": False, "error": f"Unexpected result type: {type(result)}"}
    except Exception as e:
        return {"success": False, "error": f"Tool invocation failed: {e}"}


def batch_edit(
    app: str, template: str, mapping: dict[str, dict[str, Any]], dry_run: bool = True
) -> list[dict[str, Any]]:
    """Batch edit attributes according to the mapping."""
    results = []

    for attr_alias, changes in mapping.items():
        if not changes:
            results.append({
                "alias": attr_alias,
                "success": False,
                "error": "No changes specified",
            })
            continue

        attr_type = get_attribute_type(app, template, attr_alias)
        if not attr_type:
            results.append({
                "alias": attr_alias,
                "success": False,
                "error": "Could not determine attribute type",
            })
            continue

        if dry_run:
            results.append({
                "alias": attr_alias,
                "type": attr_type,
                "changes": changes,
                "action": "WOULD EDIT",
                "success": True,
            })
            continue

        result = edit_attribute_via_tool(app, template, attr_alias, attr_type, changes)

        results.append({
            "alias": attr_alias,
            "type": attr_type,
            "changes": changes,
            "result": result,
            "success": result.get("success", False),
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch edit template attributes using proper tool layer",
    )
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--template", required=True, help="Template system name")
    parser.add_argument("--mapping", required=True, help="Path to JSON mapping file")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Preview changes without applying")
    parser.add_argument("--execute", action="store_true", help="Execute changes (default is dry-run)")
    args = parser.parse_args()

    with open(args.mapping) as f:
        mapping = json.load(f).get("attributes", {})

    if not mapping:
        print("No attributes found in mapping file")
        sys.exit(1)

    dry_run = not args.execute

    print(f"\n Batch editing attributes for {args.app}.{args.template}")
    print(f" Mode: {'DRY-RUN' if dry_run else 'EXECUTE'}")
    print(f" Attributes: {len(mapping)}")
    print("-" * 60)

    results = batch_edit(args.app, args.template, mapping, dry_run)

    success_count = sum(1 for r in results if r.get("success"))
    fail_count = len(results) - success_count

    for r in results:
        status = "OK" if r.get("success") else "FAIL"
        if dry_run:
            changes_str = ", ".join(f"{k}={v}" for k, v in r.get("changes", {}).items())
            print(f"  [{status}] {r['alias']} ({r.get('type', '?')}): {changes_str}")
        else:
            action = r.get("action", "EDITED" if r.get("success") else "FAILED")
            print(f"  [{status}] {r['alias']}: {action}")
            if not r.get("success") and r.get("error"):
                print(f"         Error: {r.get('error')}")

    print("-" * 60)
    print(f" Summary: {success_count} succeeded, {fail_count} failed")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
