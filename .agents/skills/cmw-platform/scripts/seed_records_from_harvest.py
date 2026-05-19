#!/usr/bin/env python3
"""Apply create/edit operations from a harvest or seed JSON file.

JSON shapes:
  - ``operations`` array (preferred): each item has operation, values, optional record_id
  - ``hosts.primary.records`` + ``map`` (harvest file): edit/create using map entries

Environment: CMW_BASE_URL, CMW_LOGIN, CMW_PASSWORD (target instance).

Example:
    python .agents/skills/cmw-platform/scripts/seed_records_from_harvest.py \\
        --file ./my-project/seed_ops.json --dry-run
    python .agents/skills/cmw-platform/scripts/seed_records_from_harvest.py \\
        --file harvest.json --apply --base-url https://target-host/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from _cmw_cli import add_repo_to_path, configure_env, require_credentials

add_repo_to_path()

from tools.templates_tools.tool_create_edit_record import create_edit_record  # noqa: E402


def _load_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("JSON root must be an object")
    return data


def _meta(data: dict[str, Any]) -> dict[str, Any]:
    meta = data.get("meta") or {}
    hosts = data.get("hosts") or {}
    primary = hosts.get("primary") or {}
    return {
        "application_system_name": (
            meta.get("application_system_name")
            or primary.get("application_system_name")
            or data.get("application_system_name")
        ),
        "template_system_name": (
            meta.get("template_system_name")
            or primary.get("template_system_name")
            or data.get("template_system_name")
        ),
    }


def _operations_from_file(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data.get("operations"), list):
        return data["operations"]

    ops: list[dict[str, Any]] = []
    for entry in data.get("map") or []:
        if not isinstance(entry, dict):
            continue
        fr_id = entry.get("fr_record_id") or entry.get("record_id")
        values = entry.get("values") or entry.get("target_values")
        if fr_id and values:
            ops.append({"operation": "edit", "record_id": str(fr_id), "values": values})
        elif values and not fr_id:
            ops.append({"operation": "create", "values": values})
    return ops


def _run_operation(
    app: str,
    template: str,
    op: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    operation = (op.get("operation") or "edit").lower()
    values = op.get("values") or {}
    record_id = op.get("record_id")
    row = {
        "operation": operation,
        "record_id": record_id,
        "keys": sorted(values.keys()),
        "status": "pending",
    }
    if dry_run:
        row["status"] = "dry_run"
        return row
    if operation not in ("create", "edit"):
        row["status"] = "error"
        row["error"] = f"unsupported operation: {operation}"
        return row
    if operation == "edit" and not record_id:
        row["status"] = "error"
        row["error"] = "edit requires record_id"
        return row

    payload: dict[str, Any] = {
        "operation": operation,
        "application_system_name": app,
        "template_system_name": template,
        "values": values,
    }
    if record_id:
        payload["record_id"] = str(record_id)

    result = create_edit_record.invoke(payload)
    row["success"] = bool(result.get("success"))
    row["status"] = "ok" if result.get("success") else "error"
    if not result.get("success"):
        row["error"] = result.get("error") or result.get("status_code")
    else:
        row["data"] = result.get("data")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed records from JSON operations file")
    parser.add_argument("--file", type=Path, required=True, help="Harvest or seed JSON path")
    parser.add_argument("--base-url", help="Target instance (default CMW_BASE_URL)")
    parser.add_argument("--app", help="Override application system name")
    parser.add_argument("--template", help="Override template system name")
    parser.add_argument("--dry-run", action="store_true", help="Preview (default)")
    parser.add_argument("--apply", action="store_true", help="Execute create/edit")
    args = parser.parse_args()

    dry_run = not args.apply
    if args.dry_run:
        dry_run = True

    data = _load_file(args.file)
    meta = _meta(data)
    app = args.app or meta.get("application_system_name")
    template = args.template or meta.get("template_system_name")
    if not app or not template:
        raise SystemExit("application and template required (--app/--template or JSON meta)")

    configure_env(base_url=args.base_url)
    base, _, _ = require_credentials()

    operations = _operations_from_file(data)
    if not operations:
        raise SystemExit("no operations found (use operations[] or map[] with values)")

    results = [
        _run_operation(app, template, op, dry_run=dry_run) for op in operations
    ]
    ok = sum(1 for r in results if r.get("status") in ("ok", "dry_run"))
    err = sum(1 for r in results if r.get("status") == "error")
    summary = {
        "base_url": base,
        "application_system_name": app,
        "template_system_name": template,
        "dry_run": dry_run,
        "operations": len(operations),
        "ok": ok,
        "errors": err,
        "results": results,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
