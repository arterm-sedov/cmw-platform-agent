#!/usr/bin/env python3
"""List backup configurations, create session, optional poll until terminal.

Environment (see .env.example): CMW_BASE_URL, CMW_LOGIN, CMW_PASSWORD

Example:
    python .agents/skills/cmw-platform/scripts/backup_configuration_session.py
    python .agents/skills/cmw-platform/scripts/backup_configuration_session.py --poll
    python .agents/skills/cmw-platform/scripts/backup_configuration_session.py --base-url https://your-host/
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from _cmw_cli import add_repo_to_path, configure_env, require_credentials

add_repo_to_path()

from tools import requests_  # noqa: E402
from tools.cmw_webapi import unwrap_webapi_payload  # noqa: E402

BACKUP_CONFIGS = "webapi/Backup/Configuration"
TERMINAL = frozenset({"Completed", "Failed", "Aborted"})


def _unwrap_list(r: dict) -> list[dict]:
    if not r.get("success"):
        return []
    raw = r.get("raw_response") or {}
    resp = unwrap_webapi_payload(raw)
    if isinstance(resp, dict):
        return [x for x in resp.values() if isinstance(x, dict)]
    if isinstance(resp, list):
        return [x for x in resp if isinstance(x, dict)]
    return []


def _unwrap_obj(r: dict) -> dict:
    raw = r.get("raw_response") or {}
    if isinstance(raw, dict):
        resp = unwrap_webapi_payload(raw)
        if isinstance(resp, dict):
            return resp
        if resp is not None and not isinstance(resp, (dict, list)):
            return {"id": resp}
    return raw if isinstance(raw, dict) else {}


def _pick_config(configs: list[dict], prefer: str | None) -> dict:
    if prefer:
        for c in configs:
            if str(c.get("id")) == prefer:
                return c
            desc = str(c.get("description") or "")
            if prefer.lower() in desc.lower():
                return c
    for c in configs:
        desc = str(c.get("description") or "")
        if "умолчан" in desc or "default" in desc.lower():
            return c
    return configs[0] if configs else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch configuration backup session")
    parser.add_argument("--base-url", help="Override CMW_BASE_URL from .env")
    parser.add_argument(
        "--config-id",
        help="Backup configuration id (default: default-named row)",
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Poll session until terminal status",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Seconds between poll requests (default 5)",
    )
    parser.add_argument(
        "--poll-max",
        type=int,
        default=60,
        help="Max poll iterations (default 60)",
    )
    args = parser.parse_args()
    configure_env(base_url=args.base_url)
    base, _, _ = require_credentials()

    configs_r = requests_._get_request(BACKUP_CONFIGS)
    if not configs_r.get("success"):
        print(json.dumps({"error": "list configurations failed", "detail": configs_r}))
        return 1
    configs = _unwrap_list(configs_r)
    if not configs:
        print(json.dumps({"error": "no backup configurations"}))
        return 1

    config = _pick_config(configs, args.config_id)
    config_id = config.get("id")
    if not config_id:
        print(json.dumps({"error": "configuration missing id"}))
        return 1

    create_r = requests_._post_request({}, f"webapi/Backup/Session/{config_id}")
    if not create_r.get("success"):
        print(json.dumps({"error": "create session failed", "detail": create_r}))
        return 1
    session = _unwrap_obj(create_r)
    session_id = str(session.get("id") or "")
    status = str(session.get("sessionStatus") or "Unknown")

    if args.poll and session_id:
        for _ in range(args.poll_max):
            if status in TERMINAL:
                break
            time.sleep(args.poll_interval)
            poll = requests_._get_request(f"webapi/Backup/Session/{session_id}")
            if poll.get("success"):
                status = str(_unwrap_obj(poll).get("sessionStatus") or status)

    out = {
        "base_url": base,
        "configuration_id": config_id,
        "configuration_description": config.get("description"),
        "configuration_file_name": config.get("fileName"),
        "session_id": session_id,
        "session_status": status,
        "configurations_count": len(configs),
    }
    print(json.dumps(out, indent=2))
    return 0 if status == "Completed" or session_id else 1


if __name__ == "__main__":
    sys.exit(main())
