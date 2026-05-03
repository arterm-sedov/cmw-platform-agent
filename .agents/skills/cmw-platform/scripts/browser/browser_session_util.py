#!/usr/bin/env python3
"""
Browser automation utility: Session management

This script demonstrates browser session management patterns:
1. Create/restore sessions
2. Save/load browser state
3. Cleanup sessions

Usage:
    python browser_session_util.py list
    python browser_session_util.py save --session-name <name>
    python browser_session_util.py load --session-name <name>
    python browser_session_util.py cleanup --session-name <name>
"""

import subprocess
import sys
import os
from pathlib import Path
import json
from datetime import datetime

STATE_DIR = Path(".browser-states")


def run_browser_command(session_name: str, command: str) -> dict:
    """Execute agent-browser command."""
    full_cmd = f"agent-browser --session-name {session_name} {command}"

    result = subprocess.run(
        full_cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=60
    )

    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


def list_sessions():
    """List all saved browser sessions."""
    print("\n=== Browser Sessions ===\n")

    if not STATE_DIR.exists():
        print("No sessions found (directory doesn't exist)")
        return

    state_files = list(STATE_DIR.glob("cmw-*.json"))

    if not state_files:
        print("No sessions found")
        return

    for state_file in sorted(state_files):
        session_name = state_file.stem.replace("cmw-", "")
        size = state_file.stat().st_size
        modified = datetime.fromtimestamp(state_file.stat().st_mtime)

        print(f"Session: {session_name}")
        print(f"  File: {state_file}")
        print(f"  Size: {size:,} bytes")
        print(f"  Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")

        # Try to read state info
        try:
            with open(state_file) as f:
                state = json.load(f)
                if "cookies" in state:
                    print(f"  Cookies: {len(state['cookies'])}")
                if "origins" in state:
                    print(f"  Origins: {len(state['origins'])}")
        except Exception as e:
            print(f"  Error reading state: {e}")

        print()


def save_session(session_name: str):
    """Save browser session state."""
    print(f"\n=== Saving Session: {session_name} ===\n")

    STATE_DIR.mkdir(exist_ok=True)
    state_file = STATE_DIR / f"cmw-{session_name}.json"

    result = run_browser_command(session_name, f'state save "{state_file}"')

    if result["success"]:
        print(f"✓ Session saved: {state_file}")

        # Show state info
        if state_file.exists():
            size = state_file.stat().st_size
            print(f"  Size: {size:,} bytes")

            with open(state_file) as f:
                state = json.load(f)
                if "cookies" in state:
                    print(f"  Cookies: {len(state['cookies'])}")
                if "origins" in state:
                    print(f"  Origins: {len(state['origins'])}")
        return True
    else:
        print(f"✗ Failed to save session: {result['stderr']}")
        return False


def load_session(session_name: str):
    """Load browser session state."""
    print(f"\n=== Loading Session: {session_name} ===\n")

    state_file = STATE_DIR / f"cmw-{session_name}.json"

    if not state_file.exists():
        print(f"✗ Session not found: {state_file}")
        return False

    result = run_browser_command(session_name, f'state load "{state_file}"')

    if result["success"]:
        print(f"✓ Session loaded: {state_file}")

        # Verify by getting current URL
        result = run_browser_command(session_name, "get url")
        if result["success"]:
            print(f"  Current URL: {result['stdout'].strip()}")
        return True
    else:
        print(f"✗ Failed to load session: {result['stderr']}")
        return False


def cleanup_session(session_name: str):
    """Cleanup browser session."""
    print(f"\n=== Cleaning Up Session: {session_name} ===\n")

    # Close browser
    print("Closing browser...")
    result = run_browser_command(session_name, "close")
    if result["success"]:
        print("✓ Browser closed")
    else:
        print(f"⚠ Browser close warning: {result['stderr']}")

    # Remove state file
    state_file = STATE_DIR / f"cmw-{session_name}.json"
    if state_file.exists():
        state_file.unlink()
        print(f"✓ State file removed: {state_file}")

    # Remove screenshots
    screenshot_dir = STATE_DIR / "screenshots" / session_name
    if screenshot_dir.exists():
        import shutil
        shutil.rmtree(screenshot_dir)
        print(f"✓ Screenshots removed: {screenshot_dir}")

    print("\n✓ Cleanup complete")
    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage browser sessions")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # List command
    subparsers.add_parser("list", help="List all sessions")

    # Save command
    save_parser = subparsers.add_parser("save", help="Save session state")
    save_parser.add_argument("--session-name", required=True, help="Session name")

    # Load command
    load_parser = subparsers.add_parser("load", help="Load session state")
    load_parser.add_argument("--session-name", required=True, help="Session name")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Cleanup session")
    cleanup_parser.add_argument("--session-name", required=True, help="Session name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "list":
        list_sessions()
    elif args.command == "save":
        success = save_session(args.session_name)
        sys.exit(0 if success else 1)
    elif args.command == "load":
        success = load_session(args.session_name)
        sys.exit(0 if success else 1)
    elif args.command == "cleanup":
        success = cleanup_session(args.session_name)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
