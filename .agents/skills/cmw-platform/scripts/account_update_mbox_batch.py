#!/usr/bin/env python3
"""Batch-update account Mbox to {username}@{CMW_EMAIL_DOMAIN} (Mbox only).

Uses AccountService List + Get (authoritative username) + Edit.
Environment: CMW_BASE_URL, CMW_LOGIN, CMW_PASSWORD, optional CMW_EMAIL_DOMAIN.

Example:
    python .agents/skills/cmw-platform/scripts/account_update_mbox_batch.py --dry-run
    python .agents/skills/cmw-platform/scripts/account_update_mbox_batch.py --apply
    python .agents/skills/cmw-platform/scripts/account_update_mbox_batch.py --apply --output results.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from _cmw_cli import configure_env, parse_raw, require_credentials, system_core_post

CYRILLIC = re.compile(r"[а-яА-ЯёЁ]")
DEFAULT_DOMAIN = "facility-demo.example"
BAD_DOMAIN_FRAGMENTS = (
    "mycompanyname.org",
    "test.ru",
    "@mail.",
    "@yandex.",
    "@gmail.",
    "cmwlab.com",
    "comindware.ru",
    "@example.test",
)


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def norm_account(raw: dict) -> dict:
    key_map = {
        "id": "Id",
        "username": "Username",
        "mbox": "Mbox",
        "fullName": "FullName",
        "isSystemAdministrator": "IsSystemAdministrator",
        "isAnonymous": "IsAnonymous",
        "isActive": "IsActive",
        "isDisabled": "IsDisabled",
        "role": "Role",
    }
    out: dict = {}
    for k, v in raw.items():
        pk = key_map.get(k, k[:1].upper() + k[1:] if k else k)
        if isinstance(v, str) and pk == "Role" and v:
            v = v[:1].upper() + v[1:]
        out[pk if pk in key_map.values() else k] = v
    return out


def list_accounts() -> list[dict]:
    raw = system_core_post("Base/AccountService/List")
    parsed = parse_raw(raw)
    if not isinstance(parsed, list):
        raise RuntimeError("List returned unexpected shape")
    return [norm_account(x) for x in parsed if isinstance(x, dict)]


def get_account(account_id: str) -> dict:
    raw = system_core_post("Base/AccountService/Get", {"id": account_id})
    parsed = parse_raw(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Get {account_id} unexpected")
    return norm_account(parsed)


def current_username(list_row: dict, get_row: dict) -> str:
    return (get_row.get("Username") or list_row.get("Username") or "").strip()


def target_mbox(username: str, domain: str) -> str:
    return f"{username}@{domain}"


def needs_mbox_update(username: str, mbox: str, target: str) -> bool:
    if mbox == target:
        return False
    if not mbox:
        return True
    if CYRILLIC.search(mbox):
        return True
    low = mbox.lower()
    if any(frag in low for frag in BAD_DOMAIN_FRAGMENTS):
        return True
    return bool(username and not low.startswith(username.lower() + "@"))


def process_account(
    list_row: dict, *, domain: str, dry_run: bool
) -> dict:
    account_id = str(list_row.get("Id") or list_row.get("id") or "")
    get_row = get_account(account_id)
    username = current_username(list_row, get_row)
    old_mbox = list_row.get("Mbox") or get_row.get("Mbox") or ""
    target = target_mbox(username, domain) if username else ""

    row = {
        "account_id": account_id,
        "username": username,
        "list_username": list_row.get("Username"),
        "old_mbox": old_mbox,
        "new_mbox": target,
        "target_mbox": target,
    }
    if not username:
        row["status"] = "skipped"
        row["reason"] = "missing_username"
        return row
    if not needs_mbox_update(username, old_mbox, target):
        row["status"] = "skipped"
        row["reason"] = "already_correct"
        row["new_mbox"] = old_mbox
        return row
    if dry_run:
        row["status"] = "dry_run"
        return row

    payload = dict(list_row)
    payload["Id"] = account_id
    payload["Mbox"] = target
    system_core_post("Base/AccountService/Edit", payload)
    list_after = next(
        (a for a in list_accounts() if str(a.get("Id")) == account_id),
        {},
    )
    live = list_after.get("Mbox") or ""
    row["status"] = "updated" if live == target else "updated_unverified"
    row["verified_mbox"] = live
    row["verified"] = live == target
    row["api_method"] = "Base/AccountService/Edit"
    row["timestamp"] = _ts()
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch Mbox refresh for all accounts")
    parser.add_argument("--base-url", help="Override CMW_BASE_URL")
    parser.add_argument(
        "--domain",
        default=os.environ.get("CMW_EMAIL_DOMAIN", DEFAULT_DOMAIN),
        help=f"Mbox domain (default env CMW_EMAIL_DOMAIN or {DEFAULT_DOMAIN})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    parser.add_argument("--apply", action="store_true", help="Apply edits")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON results to this path (project repo, not committed)",
    )
    args = parser.parse_args()
    dry_run = not args.apply
    if args.dry_run:
        dry_run = True

    configure_env(base_url=args.base_url)
    base, _, _ = require_credentials()
    domain = args.domain.strip()

    list_rows = list_accounts()
    list_rows.sort(key=lambda a: (a.get("Username") or "").lower())
    rows = [process_account(r, domain=domain, dry_run=dry_run) for r in list_rows]

    payload = {
        "meta": {
            "base_url": base,
            "mbox_domain": domain,
            "mbox_pattern": f"{{username}}@{domain}",
            "dry_run": dry_run,
            "api_method": "Base/AccountService/Edit",
            "changes": sum(1 for r in rows if r.get("status") == "updated"),
        },
        "results": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")

    print(f"accounts={len(rows)} domain={domain} dry_run={dry_run}")
    for row in rows:
        print(
            f"{row.get('username', ''):<22} "
            f"{(row.get('old_mbox') or ''):<36} "
            f"{(row.get('new_mbox') or ''):<36} "
            f"{row.get('status')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
