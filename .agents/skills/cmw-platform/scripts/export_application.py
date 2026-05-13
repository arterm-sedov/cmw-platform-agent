#!/usr/bin/env python3
"""
Export application from CMW Platform to CTF format.

Usage:
    python export_application.py --app Volga --output-dir C:\\tmp\\cmw-transfer
"""
import argparse
import base64
import os
import sys
import uuid
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

APP_DIR = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(APP_DIR))


def get_credentials() -> tuple[str, str]:
    """Get login and password from .env file."""
    env_path = APP_DIR / ".env"
    login = ""
    password = ""

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("CMW_LOGIN="):
                login = line.split("=", 1)[1].strip().strip('"').strip("'")
            elif line.startswith("CMW_PASSWORD="):
                password = line.split("=", 1)[1].strip().strip('"').strip("'")

    return login, password


def get_basic_auth_header() -> dict:
    """Get headers with Basic auth."""
    login, password = get_credentials()
    if not login or not password:
        return {"Accept": "application/json"}

    credentials = base64.b64encode(f"{login}:{password}".encode("ascii")).decode("ascii")
    return {
        "Authorization": f"Basic {credentials}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def get_base_url() -> str:
    """Get CMW base URL from environment."""
    base_url = os.environ.get("CMW_BASE_URL", "")
    if base_url:
        return base_url.rstrip("/")

    env_path = APP_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("CMW_BASE_URL="):
                base_url = line.split("=", 1)[1].strip().strip('"').strip("'")
                return base_url.rstrip("/")

    return "https://mz-fr.test.cbap.ru"


def main(app: str, output_dir: str, save_to_file: bool = True) -> int:
    """Export application to CTF."""
    import requests

    base_url = get_base_url()
    endpoint = f"{base_url}/webapi/Transfer/{app}"
    headers = get_basic_auth_header()

    try:
        response = requests.get(endpoint, headers=headers, timeout=120)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error: Export failed: {e}")
        return 1

    result = response.json()

    # Extract CTF data - same logic as tool_export_application
    if not result.get("success", False):
        error_msg = result.get("error", {}).get("message", "Unknown error")
        print(f"Error: API export failed: {error_msg}")
        return 1

    raw_response = result.get("response", {})
    if isinstance(raw_response, dict):
        ctf_data = raw_response.get("data", "") or raw_response.get("ctf", "")
    else:
        ctf_data = str(raw_response) if raw_response else ""

    if not ctf_data:
        print("Error: No CTF data in response")
        return 1

    # Decode base64 and save to file
    try:
        ctf_bytes = base64.b64decode(ctf_data)
    except Exception as e:
        print(f"Error: Failed to decode CTF data: {e}")
        return 1

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