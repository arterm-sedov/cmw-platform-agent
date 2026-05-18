"""Discover Staff account-link field via John Demonstrator / demo vs unattached rows."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)
os.environ["CMW_USE_DOTENV"] = "true"

from tools import requests_  # noqa: E402

APP = "Volga"
TEMPLATE = "Sotrudniki"
TEMPLATE_GA = f"Template@{APP}.{TEMPLATE}"


def set_host(url: str) -> None:
    os.environ["CMW_BASE_URL"] = url.rstrip("/") + "/"


def get_json(endpoint: str) -> tuple[dict | list | str | None, dict]:
    r = requests_._get_request(endpoint)
    return r.get("raw_response"), r


def get_record(record_id: str) -> dict | None:
    r = requests_._get_request(f"webapi/Record/{record_id}")
    if not r.get("success"):
        return None
    raw = r.get("raw_response")
    if isinstance(raw, dict) and "response" in raw:
        resp = raw["response"]
        if isinstance(resp, dict):
            return resp
    if isinstance(raw, dict):
        return raw
    return None


def list_attrs() -> list[dict]:
    r = requests_._get_request(f"webapi/Attribute/List/{TEMPLATE_GA}")
    if not r.get("success"):
        return []
    raw = r.get("raw_response") or {}
    resp = raw.get("response") if isinstance(raw, dict) else None
    return resp if isinstance(resp, list) else []


def account_type_attrs(attrs: list[dict]) -> list[dict]:
    out = []
    for item in attrs:
        t = (item.get("type") or "").lower()
        gl = item.get("globalAlias") or {}
        alias = gl.get("alias") or item.get("alias") or ""
        if t in ("account", "role") or "account" in str(alias).lower():
            out.append(
                {
                    "alias": alias,
                    "type": t,
                    "display": item.get("displayName") or item.get("name"),
                    "isSystem": item.get("isSystem"),
                }
            )
    return out


def summarize_record(rec: dict | None, label: str) -> dict:
    if not rec:
        return {"label": label, "error": "not found"}
    interesting = {}
    for k, v in rec.items():
        if v in (None, "", [], {}):
            continue
        if k.startswith("_"):
            continue
        interesting[k] = v
    return {"label": label, "id": rec.get("id"), "fields": interesting}


def main() -> int:
    hosts = {
        "FR": "https://mz-fr.test.cbap.ru/",
        "TR": "https://mz-tr.test.cbap.ru/",
    }
    record_ids = {
        "FR": ["account.2", "account.182", "account.183"],
        "TR": ["account.2", "account.3"],
    }
    out: dict = {}
    for host_label, base in hosts.items():
        set_host(base)
        attrs = list_attrs()
        out[host_label] = {
            "account_type_attrs": account_type_attrs(attrs),
            "records": [],
        }
        for rid in record_ids.get(host_label, []):
            rec = get_record(rid)
            out[host_label]["records"].append(summarize_record(rec, rid))

    # Search TR employees list for demonstrator (first page only via filter if possible)
    set_host(hosts["TR"])
    r = requests_._get_request(f"webapi/Records/AccountTemplate@{APP}.{TEMPLATE}")
    tr_hits = []
    if r.get("success"):
        raw = r.get("raw_response") or {}
        resp = raw.get("response")
        recs = list(resp.values()) if isinstance(resp, dict) else (resp or [])
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            blob = json.dumps(rec, ensure_ascii=False).lower()
            if any(
                x in blob
                for x in ("demonstrator", "john", '"username":"demo"', '"username": "demo"')
            ):
                tr_hits.append(summarize_record(rec, str(rec.get("id"))))
    out["TR_demonstrator_search"] = tr_hits[:5]

    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
