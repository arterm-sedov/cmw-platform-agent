#!/usr/bin/env python3
"""
Safe attribute translation script - READ FIRST then EDIT.
Preserves complete original schema while updating only name/description.
"""
import sys
import os
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")


# Translation mapping: system_name -> (new_name, new_description)
TRANSLATIONS = {}


def get_credentials():
    base_url = os.environ.get("CMW_BASE_URL", "")
    login = os.environ.get("CMW_LOGIN", "")
    password = os.environ.get("CMW_PASSWORD", "")
    if not base_url or not login or not password:
        raise ValueError("Missing CMW_BASE_URL, CMW_LOGIN, or CMW_PASSWORD")
    creds = base64.b64encode(f"{login}:{password}".encode()).decode()
    return base_url.rstrip("/"), {"Authorization": f"Basic {creds}", "Content-Type": "application/json"}


def translate_attribute(base_url: str, headers: dict, app: str, template: str, 
                       attr_alias: str, new_name: str, new_description: str | None = None):
    """
    Safely translate an attribute - READ first, then EDIT with full schema preserved.
    """
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

    # Step 2: Modify ONLY name/description, preserve everything else
    current["name"] = new_name
    if new_description:
        current["description"] = new_description

    # Step 3: PUT back complete schema
    put_resp = requests.put(
        f"{base_url}/webapi/Attribute/{app}",
        headers=headers,
        json=current,
        timeout=30
    )

    result = put_resp.json()
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Safe attribute translation - READ FIRST")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--template", required=True, help="Template system name")  
    parser.add_argument("--attr", required=True, help="Attribute system name")
    parser.add_argument("--name", required=True, help="New display name")
    parser.add_argument("--desc", default=None, help="New description")
    args = parser.parse_args()

    base_url, headers = get_credentials()

    result = translate_attribute(
        base_url, headers,
        args.app, args.template,
        args.attr, args.name, args.desc
    )

    if result.get("success"):
        print(f"SUCCESS: {args.attr} -> {args.name}")
    else:
        print(f"FAILED: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
