#!/usr/bin/env python3
"""Compare AccountService List vs Get — List Username may be stale after renames.

Environment: CMW_BASE_URL, CMW_LOGIN, CMW_PASSWORD

Example:
    python .agents/skills/cmw-platform/scripts/verify_account_usernames.py
    python .agents/skills/cmw-platform/scripts/verify_account_usernames.py --account-id account.5
    python .agents/skills/cmw-platform/scripts/verify_account_usernames.py --only-stale
"""
from __future__ import annotations

import argparse
import json
import sys

from _cmw_cli import configure_env, parse_raw, require_credentials, system_core_post


def list_accounts() -> list[dict]:
    raw = system_core_post("Base/AccountService/List")
    parsed = parse_raw(raw)
    if not isinstance(parsed, list):
        raise RuntimeError("List returned unexpected shape")
    return [x for x in parsed if isinstance(x, dict)]


def get_account(account_id: str) -> dict:
    raw = system_core_post("Base/AccountService/Get", {"id": account_id})
    parsed = parse_raw(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Get {account_id} failed")
    return parsed


def compare_row(row: dict) -> dict:
    aid = str(row.get("Id") or row.get("id") or "")
    live = get_account(aid)
    list_user = (row.get("Username") or row.get("username") or "").strip()
    get_user = (live.get("Username") or live.get("username") or "").strip()
    list_mbox = row.get("Mbox") or row.get("mbox") or ""
    get_mbox = live.get("Mbox") or live.get("mbox") or ""
    stale = list_user != get_user
    return {
        "account_id": aid,
        "list_username": list_user,
        "get_username": get_user,
        "username_stale": stale,
        "list_mbox": list_mbox,
        "get_mbox": get_mbox,
        "mbox_stale": list_mbox != get_mbox and bool(get_mbox),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify List vs Get account fields (staleness check)",
    )
    parser.add_argument("--base-url", help="Override CMW_BASE_URL")
    parser.add_argument(
        "--account-id",
        action="append",
        dest="account_ids",
        help="Check only these ids (repeatable)",
    )
    parser.add_argument(
        "--only-stale",
        action="store_true",
        help="Print only rows where List username != Get username",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()
    configure_env(base_url=args.base_url)
    base, _, _ = require_credentials()

    rows = list_accounts()
    if args.account_ids:
        wanted = set(args.account_ids)
        rows = [r for r in rows if str(r.get("Id") or r.get("id")) in wanted]

    results = [compare_row(r) for r in rows]
    if args.only_stale:
        results = [r for r in results if r["username_stale"]]

    stale_count = sum(1 for r in results if r["username_stale"])
    summary = {
        "base_url": base,
        "checked": len(results),
        "username_stale_count": stale_count,
        "rows": results,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"base={base} checked={len(results)} stale_usernames={stale_count}")
        for r in results:
            flag = "STALE" if r["username_stale"] else "ok"
            print(
                f"{r['account_id']:<14} list={r['list_username']:<20} "
                f"get={r['get_username']:<20} {flag}"
            )
    return 1 if stale_count else 0


if __name__ == "__main__":
    sys.exit(main())
