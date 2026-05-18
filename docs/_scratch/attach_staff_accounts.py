"""Attach FR Staff rows to platform accounts via record edit (discover + apply)."""
from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)
os.environ["CMW_USE_DOTENV"] = "true"
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from tools import requests_  # noqa: E402
from tools.cmw_webapi import unwrap_webapi_payload  # noqa: E402
from tools.platform_record_document import fetch_record_field_values  # noqa: E402
from tools.templates_tools.tool_create_edit_record import create_edit_record  # noqa: E402

FR = "https://mz-fr.test.cbap.ru/"
TR = "https://mz-tr.test.cbap.ru/"
APP = "Volga"
TEMPLATE = "Sotrudniki"

PERSONAS = [
    ("dispatcher01", "account.3", "account.182"),
    ("engineer", "account.4", "account.183"),
    ("technician", "account.5", "account.184"),
    ("cleaning_manager", "account.8", "account.185"),
    ("manager", "account.9", "account.186"),
    ("chief_engineer", "account.10", "account.187"),
    ("operations_head", "account.12", "account.188"),
    ("smirnova", "account.13", "account.189"),
    ("serov", "account.14", "account.190"),
    ("isaeva", "account.15", "account.191"),
]

LINK_FIELDS = (
    "username",
    "cmw_account_username",
    "cmw_account_fullName",
    "cmw_account_mbox",
    "cmw_account_active",
)


def set_host(url: str) -> None:
    os.environ["CMW_BASE_URL"] = url.rstrip("/") + "/"


def fetch_list() -> list[dict]:
    r = requests_._get_request(f"webapi/Records/AccountTemplate@{APP}.{TEMPLATE}")
    if not r.get("success"):
        return []
    resp = unwrap_webapi_payload(r.get("raw_response"))
    if isinstance(resp, dict):
        return [x for x in resp.values() if isinstance(x, dict)]
    if isinstance(resp, list):
        return [x for x in resp if isinstance(x, dict)]
    return []


def is_linked(rec: dict) -> bool:
    un = (rec.get("username") or rec.get("cmw_account_username") or "").strip()
    return bool(un)


def probe_record(host: str, record_id: str) -> dict:
    set_host(host)
    out: dict = {"record_id": record_id}
    r = requests_._get_request(f"webapi/Record/{record_id}")
    out["get_ok"] = r.get("success")
    out["get_status"] = r.get("status_code")
    raw = r.get("raw_response")
    inner = unwrap_webapi_payload(raw)
    if isinstance(inner, dict):
        out["get_fields"] = {
            k: inner[k]
            for k in LINK_FIELDS
            if k in inner and inner[k] not in (None, "")
        }
    else:
        out["get_inner_type"] = type(inner).__name__
    gp = fetch_record_field_values(record_id, list(LINK_FIELDS))
    out["gpv"] = gp.get("data", {}).get(record_id) if gp.get("success") else gp.get("error")
    return out


def find_demo_samples(host: str, recs: list[dict]) -> list[dict]:
    hits = []
    for rec in recs:
        un = (rec.get("username") or rec.get("cmw_account_username") or "").lower()
        fn = str(rec.get("fullName") or rec.get("cmw_account_fullName") or "").lower()
        if un == "demo" or "demonstrator" in fn or "john" in fn:
            hits.append(
                {
                    "id": rec.get("id"),
                    "username": rec.get("username") or rec.get("cmw_account_username"),
                    "fullName": rec.get("fullName") or rec.get("cmw_account_fullName"),
                    "keys": sorted(
                        k
                        for k, v in rec.items()
                        if v not in (None, "", [], {}) and not str(k).startswith("_")
                    ),
                }
            )
    return hits


def try_include_single(account_id: str, container_id: str) -> dict:
    body = {"accountId": account_id, "containerId": container_id}
    r = requests_._post_request(
        body, "api/public/system/TeamNetwork/ObjectService/IncludeInContainer"
    )
    return {
        "success": r.get("success"),
        "status_code": r.get("status_code"),
        "error": r.get("error"),
    }


def try_edit(record_id: str, values: dict) -> dict:
    return create_edit_record.invoke(
        {
            "operation": "edit",
            "application_system_name": APP,
            "template_system_name": TEMPLATE,
            "record_id": record_id,
            "values": values,
        }
    )


def verify_employee(emp_id: str) -> bool:
    recs = fetch_list()
    for r in recs:
        if str(r.get("id")) == emp_id:
            return is_linked(r)
    gp = fetch_record_field_values(emp_id, ["username", "cmw_account_username"])
    if gp.get("success"):
        row = gp.get("data", {}).get(emp_id, {})
        return bool((row.get("username") or row.get("cmw_account_username") or "").strip())
    return False


def main() -> int:
    discovery: dict = {}
    for label, host in [("TR", TR), ("FR", FR)]:
        set_host(host)
        recs = fetch_list()
        discovery[label] = {
            "count": len(recs),
            "linked": sum(1 for r in recs if is_linked(r)),
            "demo_samples": find_demo_samples(host, recs),
        }
        if discovery[label]["demo_samples"]:
            rid = discovery[label]["demo_samples"][0]["id"]
            discovery[label]["demo_probe"] = probe_record(host, str(rid))

    # Pick first unattached FR row for edit experiments
    set_host(FR)
    fr_recs = fetch_list()
    engineer = next((r for r in fr_recs if str(r.get("id")) == "account.183"), None)
    discovery["FR_engineer_before"] = {
        k: engineer.get(k)
        for k in LINK_FIELDS
        if engineer and engineer.get(k) not in (None, "")
    } if engineer else None

    attach_results = []
    # Strategy order from discovery: username string, then IncludeInContainer
    for username, account_id, emp_id in PERSONAS:
        set_host(FR)
        before = verify_employee(emp_id)
        if before:
            attach_results.append(
                {
                    "username": username,
                    "employee_id": emp_id,
                    "attached": True,
                    "method": "already_linked",
                    "notes": "",
                }
            )
            continue

        notes = []
        ok = False
        method = ""

        # 1) PUT username (OpenAPI / list field name)
        for field, value in [
            ("username", username),
            ("cmw_account_username", username),
        ]:
            res = try_edit(emp_id, {field: value})
            notes.append(f"edit {field}: {res}")
            if res.get("success") and verify_employee(emp_id):
                ok = True
                method = f"create_edit_record:{field}"
                break

        # 2) IncludeInContainer single
        if not ok:
            inc = try_include_single(account_id, emp_id)
            notes.append(f"IncludeInContainer: {inc}")
            if inc.get("success") and verify_employee(emp_id):
                ok = True
                method = "IncludeInContainer"

        attach_results.append(
            {
                "username": username,
                "employee_id": emp_id,
                "account_id": account_id,
                "attached": ok,
                "method": method,
                "notes": "; ".join(str(n) for n in notes)[:500],
            }
        )

    out = {
        "discovery": discovery,
        "attach_results": attach_results,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    out_path = ROOT / "docs" / "_scratch" / "attach_staff_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if all(r["attached"] for r in attach_results) else 1


if __name__ == "__main__":
    sys.exit(main())
