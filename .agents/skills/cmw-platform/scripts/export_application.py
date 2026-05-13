#!/usr/bin/env python3
"""
Export application from CMW Platform to CTF format.

Usage:
    python export_application.py --app Volga --output-dir C:\\tmp\\cmw-transfer
"""
import argparse
import base64
import json
import os
import sys
import uuid
from pathlib import Path

APP_DIR = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(APP_DIR))


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

    return "https://mz-fr.test.cbap.ru"


def main(app: str, output_dir: str, save_to_file: bool = True) -> int:
    """Export application to CTF."""
    base_url = get_base_url()
    endpoint = f"{base_url}/webapi/Transfer/{app}"

    import requests

    headers = {
        "Accept": "application/json",
    }

    session_token = os.environ.get("CMW_SESSION_TOKEN", "")
    if session_token:
        headers["Authorization"] = f"Bearer {session_token}"

    try:
        response = requests.get(endpoint, headers=headers, timeout=120)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error: Export failed: {e}")
        return 1

    result = response.json()

    if not result.get("success", False):
        error_msg = result.get("error", "Unknown error")
        print(f"Error: API export failed: {error_msg}")
        return 1

    raw_response = result.get("raw_response", {})
    if isinstance(raw_response, dict):
        response_data = raw_response.get("response", {})
    else:
        response_data = raw_response

    ctf_data = response_data.get("data", "") or response_data.get("ctf", "")
    if not ctf_data:
        ctf_data = str(response_data)

    if not ctf_data:
        print("Error: No CTF data in response")
        return 1

    if not save_to_file:
        print(ctf_data)
        return 0

    ctf_bytes = base64.b64decode(ctf_data)

    os.makedirs(output_dir, exist_ok=True)

    safe_name = "".join(c for c in app if c.isalnum() or c in "-_")
    timestamp = uuid.uuid4().hex[:8]
    filename = f"{safe_name}_{timestamp}.ctf"
    file_path = os.path.join(output_dir, filename)

    with open(file_path, "wb") as f:
        f.write(ctf_bytes)

    print(f"Exported {app} to: {file_path}")
    print(f"CTF size: {len(ctf_bytes)} bytes")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export application from CMW Platform to CTF")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--output-dir", required=True, help="Path to output directory")
    parser.add_argument("--save-to-file", default=True, help="Save to file (default: True)")
    args = parser.parse_args()
    sys.exit(main(args.app, args.output_dir, args.save_to_file))