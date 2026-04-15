#!/usr/bin/env python3
"""
Safe attribute edit - READ first, then EDIT with full schema preserved.
Use this when you want to change ONLY specific fields (name, description) 
without affecting the rest of the schema.

Usage:
    python safe_edit_attribute.py --app Volga --template RentLots --attr Ploschad --name "Lot Area"
    python safe_edit_attribute.py --app Volga --template RentLots --attr Ploschad --desc "Area in sqm"
    python safe_edit_attribute.py --app Volga --template RentLots --attr Ploschad --name "Lot Area" --desc "Area in sqm"
"""
import sys
import os
import base64
import requests
import argparse
from pathlib import Path
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")


def get_credentials():
    base_url = os.environ.get("CMW_BASE_URL", "")
    login = os.environ.get("CMW_LOGIN", "")
    password = os.environ.get("CMW_PASSWORD", "")
    if not base_url or not login or not password:
        raise ValueError("Missing CMW_BASE_URL, CMW_LOGIN, or CMW_PASSWORD")
    creds = base64.b64encode(f"{login}:{password}".encode()).decode()
    return base_url.rstrip("/"), {"Authorization": f"Basic {creds}", "Content-Type": "application/json"}


def edit_attribute(app: str, template: str, attr_alias: str, 
                   new_name: str | None = None, 
                   new_description: str | None = None):
    """
    Safely edit an attribute - READ first, then EDIT with full schema preserved.

    Only updates name and/or description - preserves everything else (type, format, 
    variants, instanceGlobalAlias, isMultiValue, etc.)
    """
    base_url, headers = get_credentials()

    # Step 1: GET current complete schema
    resp = requests.get(
        f"{base_url}/webapi/Attribute/List/Template@{app}.{template}",
        headers=headers,
        timeout=30
    )

    if not resp.json().get("response"):
        return {"success": False, "error": f"Cannot read attributes: {resp.json().get('error')}"}

    attrs = resp.json()["response"]
    current = None
    for a in attrs:
        if a.get("globalAlias", {}).get("alias") == attr_alias:
            current = a
            break

    if not current:
        return {"success": False, "error": f"Attribute not found: {attr_alias}"}

    # Step 2: Modify ONLY specified fields
    if new_name:
        current["name"] = new_name
    if new_description:
        current["description"] = new_description

    # Step 3: PUT back complete schema (only changed fields will differ)
    put_resp = requests.put(
        f"{base_url}/webapi/Attribute/{app}",
        headers=headers,
        json=current,
        timeout=30
    )

    result = put_resp.json()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Safe attribute edit - READ first, preserve schema, EDIT only name/description"
    )
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--template", required=True, help="Template system name")
    parser.add_argument("--attr", required=True, help="Attribute system name (alias)")
    parser.add_argument("--name", default=None, help="New display name")
    parser.add_argument("--desc", default=None, help="New description")
    args = parser.parse_args()

    if not args.name and not args.desc:
        print("Error: Must specify either --name or --desc")
        sys.exit(1)

    result = edit_attribute(
        args.app, args.template, args.attr,
        args.name, args.desc
    )

    if result.get("success"):
        print(f"SUCCESS: {args.attr}")
        if args.name:
            print(f"  name -> {args.name}")
        if args.desc:
            print(f"  desc -> {args.desc}")
    else:
        print(f"FAILED: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
