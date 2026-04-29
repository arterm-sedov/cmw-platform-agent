#!/usr/bin/env python3
"""
Connection diagnostics for cmw-platform skill.

Verifies:
1. Configuration file exists with required keys
2. Platform is accessible via HTTP
3. Credentials authenticate successfully

Usage:
    python .agents/skills/cmw-platform/scripts/diagnose_connection.py

Exit codes:
    0 = all checks passed
    1 = some checks failed
"""

import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
AGENT_ROOT = SCRIPT_DIR.parent.parent.parent.parent


def check_config_file():
    print("=" * 60)
    print("CHECKING CONFIGURATION FILE")
    print("=" * 60)

    env_file = AGENT_ROOT / ".env"

    if not env_file.exists():
        print(f"FAIL: .env not found at {env_file}")
        return False

    print(f"OK: .env found at {env_file}")

    required_keys = ["CMW_BASE_URL", "CMW_LOGIN", "CMW_PASSWORD"]
    found_keys = []

    with open(env_file, encoding="utf-8") as f:
        content = f.read()
        for key in required_keys:
            if key in content:
                found_keys.append(key)
                print(f"OK: {key} present")
            else:
                print(f"FAIL: {key} missing")

    missing = set(required_keys) - set(found_keys)
    if missing:
        print(f"FAIL: Missing required keys: {missing}")
        return False

    print("OK: All required keys present")
    return True


def check_connection():
    print("\n" + "=" * 60)
    print("TESTING API CONNECTION")
    print("=" * 60)

    try:
        from dotenv import load_dotenv
        env_file = AGENT_ROOT / ".env"
        load_dotenv(env_file)
    except ImportError:
        print("FAIL: python-dotenv not installed")
        print("  Run: pip install python-dotenv")
        return False

    base_url = os.environ.get("CMW_BASE_URL", "").strip().rstrip("/")
    login = os.environ.get("CMW_LOGIN", "").strip()
    password = os.environ.get("CMW_PASSWORD", "").strip()

    if not base_url:
        print("FAIL: CMW_BASE_URL is empty")
        return False

    if not login or not password:
        print("FAIL: CMW_LOGIN or CMW_PASSWORD is empty")
        return False

    print(f"  URL: {base_url}")
    print(f"  Login: {login}")
    print(f"  Password: {'*' * len(password)}")

    import base64
    import requests

    credentials = base64.b64encode(f"{login}:{password}".encode()).decode()
    headers = {
        "Authorization": f"Basic {credentials}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        print("  Connecting to webapi/Solution...")
        response = requests.get(
            f"{base_url}/webapi/Solution",
            headers=headers,
            timeout=30
        )
        print(f"  HTTP Status: {response.status_code}")

        if response.status_code == 200:
            print("OK: Connection successful - platform is accessible")
            return True
        elif response.status_code == 401:
            print("FAIL: Authentication failed - check credentials")
            return False
        else:
            print(f"WARN: Unexpected status code {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return False

    except requests.exceptions.Timeout:
        print("FAIL: Request timeout - platform may be slow or unreachable")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"FAIL: Connection error - {e}")
        return False
    except Exception as e:
        print(f"FAIL: Unexpected error - {e}")
        return False


def main():
    print("=" * 60)
    print("cmw-platform Connection Diagnostics")
    print("=" * 60)
    print()

    env_ok = check_config_file()
    conn_ok = check_connection() if env_ok else False

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if env_ok and conn_ok:
        print("ALL CHECKS PASSED")
        print("Platform connection is healthy.")
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED")
        print("Review errors above and fix before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()