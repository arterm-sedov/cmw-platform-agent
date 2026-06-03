#!/usr/bin/env python3
"""
Import CTF file back to CMW Platform.

Usage:
    python import_application.py --app Volga --ctf-file C:\\tmp\\cmw-transfer\\Volga_tr_renamed.ctf
"""
import argparse
import base64
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(APP_DIR))

import requests


def get_base_url() -> str:
    """Get CMW base URL from environment."""
    base_url = os.environ.get("CMW_BASE_URL", "")
    if base_url:
        return base_url.rstrip("/")

    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("CMW_BASE_URL="):
                base_url = line.split("=", 1)[1].strip().strip('"').strip("'")
                return base_url.rstrip("/")

    raise RuntimeError("CMW_BASE_URL is not set (environment or .env)")


def main(app: str, ctf_file: str | None = None, ctf_data: str | None = None) -> int:
    """Import CTF to application."""
    if not ctf_file and not ctf_data:
        print("Error: Either --ctf-file or --ctf-data must be provided")
        return 1

    if ctf_file:
        if not os.path.exists(ctf_file):
            print(f"Error: CTF file not found: {ctf_file}")
            return 1

        with open(ctf_file, "rb") as f:
            ctf_bytes = f.read()

        ctf_data = base64.b64encode(ctf_bytes).decode("utf-8")
    else:
        ctf_bytes = base64.b64decode(ctf_data)
        ctf_data = ctf_data

    base_url = get_base_url()
    endpoint = f"{base_url}/webapi/Transfer"

    import requests

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    session_token = os.environ.get("CMW_SESSION_TOKEN", "")
    if session_token:
        headers["Authorization"] = f"Bearer {session_token}"

    payload = {
        "applicationSystemName": app,
        "ctf": ctf_data,
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error: Import failed: {e}")
        return 1

    result = response.json()

    if result.get("success", False):
        print(f"Successfully imported {app}")
        if result.get("data"):
            print(json.dumps(result.get("data"), indent=2))
        return 0
    else:
        error_msg = result.get("error", "Unknown error")
        print(f"Error: Import failed: {error_msg}")
        return 1


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description="Import CTF file to CMW Platform")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--ctf-file", help="Path to CTF file")
    parser.add_argument("--ctf-data", help="Base64-encoded CTF data")
    args = parser.parse_args()
    sys.exit(main(args.app, args.ctf_file, args.ctf_data))