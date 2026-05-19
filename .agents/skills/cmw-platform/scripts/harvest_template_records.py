#!/usr/bin/env python3
"""Read-only harvest: list records for a template (field inventory + slim rows).

Environment: CMW_BASE_URL, CMW_LOGIN, CMW_PASSWORD (per host run).

Example:
    python .agents/skills/cmw-platform/scripts/harvest_template_records.py \\
        --app Volga --template Sobytiya --limit 20
    python .agents/skills/cmw-platform/scripts/harvest_template_records.py \\
        --app Volga --template GodRabochihChasov \\
        --base-url https://reference-host/ \\
        --compare-url https://target-host/ \\
        --output harvest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _cmw_cli import add_repo_to_path, configure_env, require_credentials

add_repo_to_path()

from tools.requests_ import _get_request  # noqa: E402


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch_records(
    app: str,
    template: str,
    *,
    limit: int,
    attributes: list[str] | None,
) -> tuple[list[dict[str, Any]], str | None]:
    attrs_q = ""
    if attributes:
        attrs_q = "&" + "&".join(f"attributes={a}" for a in attributes)
    ep = (
        f"webapi/Records/Template@{app}.{template}"
        f"?limit={limit}&offset=0&sortBy=creationDate&sortDesc=true{attrs_q}"
    )
    r = _get_request(ep)
    if not r.get("success"):
        return [], str(r.get("error") or r.get("status_code"))
    raw = r.get("raw_response")
    if isinstance(raw, dict):
        resp = raw.get("response")
        if isinstance(resp, list):
            return resp, None
        if isinstance(resp, dict):
            return [x for x in resp.values() if isinstance(x, dict)], None
    return [], "unexpected response shape"


def _field_inventory(rows: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    return sorted(keys)


def _slim_row(row: dict[str, Any], *, keys: list[str] | None) -> dict[str, Any]:
    if keys:
        return {k: row.get(k) for k in keys if k in row}
    keep = ("id", "name", "creationDate", "lastWriteDate")
    out = {k: row[k] for k in keep if k in row}
    for k, v in row.items():
        if k in out:
            continue
        if v in (None, "", [], {}):
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
    return out


def harvest_one(
    label: str,
    base_url: str,
    app: str,
    template: str,
    *,
    limit: int,
    attributes: list[str] | None,
    slim_keys: list[str] | None,
) -> dict[str, Any]:
    configure_env(base_url=base_url)
    rows, err = _fetch_records(app, template, limit=limit, attributes=attributes)
    slim = [_slim_row(r, keys=slim_keys) for r in rows]
    return {
        "label": label,
        "base_url": base_url.rstrip("/") + "/",
        "application_system_name": app,
        "template_system_name": template,
        "error": err,
        "count": len(slim),
        "field_inventory": _field_inventory(rows),
        "records": slim,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Harvest template records (read-only)")
    parser.add_argument("--app", required=True, help="Application system name")
    parser.add_argument("--template", required=True, help="Template system name")
    parser.add_argument("--base-url", help="Source host (default CMW_BASE_URL)")
    parser.add_argument(
        "--compare-url",
        help="Optional second host (e.g. target instance for TR/FR diff)",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--attribute",
        action="append",
        dest="attributes",
        help="Restrict attributes in list query (repeatable)",
    )
    parser.add_argument(
        "--slim-key",
        action="append",
        dest="slim_keys",
        help="Keep only these keys in output rows",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write harvest JSON (use project repo path; do not commit PII)",
    )
    args = parser.parse_args()

    configure_env(base_url=args.base_url)
    primary_url, _, _ = require_credentials()

    out: dict[str, Any] = {
        "meta": {
            "harvested_at": _ts(),
            "application_system_name": args.app,
            "template_system_name": args.template,
            "limit": args.limit,
        },
        "hosts": {},
    }

    out["hosts"]["primary"] = harvest_one(
        "primary",
        primary_url,
        args.app,
        args.template,
        limit=args.limit,
        attributes=args.attributes,
        slim_keys=args.slim_keys,
    )

    if args.compare_url:
        out["hosts"]["compare"] = harvest_one(
            "compare",
            args.compare_url,
            args.app,
            args.template,
            limit=args.limit,
            attributes=args.attributes,
            slim_keys=args.slim_keys,
        )
        p_ids = {str(r.get("id")) for r in out["hosts"]["primary"]["records"]}
        c_ids = {str(r.get("id")) for r in out["hosts"]["compare"]["records"]}
        out["diff"] = {
            "primary_count": len(p_ids),
            "compare_count": len(c_ids),
            "note": "Record ids are host-specific; compare field_inventory and counts only",
        }

    text = json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    err = out["hosts"]["primary"].get("error")
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
