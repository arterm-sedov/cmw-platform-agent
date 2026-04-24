"""One-off: FacilityManagement / MaintenancePlans / record — document fetch smoke (delete after use)."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

# repo root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

load_dotenv(os.path.join(_ROOT, ".env"))

from tools.platform_record_document import (  # noqa: E402
    extract_platform_document_id,
    fetch_record_field_values,
    get_document_content,
    get_document_model,
)
from tools.templates_tools.tool_list_attributes import list_attributes  # noqa: E402

APP = "FacilityManagement"
TPL = "MaintenancePlans"
RECORD_ID = "4738"


def _row_from_fetch(fv: dict) -> dict:
    data = fv.get("data") or {}
    row = data.get(RECORD_ID, {})
    if isinstance(row, dict) and row:
        return row
    if len(data) == 1:
        only = next(iter(data.values()))
        if isinstance(only, dict):
            print("Note: using sole record row (id key in response differed from input).")
            return only
    return row if isinstance(row, dict) else {}


def main() -> int:
    la = list_attributes.invoke(
        {"application_system_name": APP, "template_system_name": TPL}
    )
    print("list_attributes:", "ok" if la.get("success") else "FAIL", la.get("error"))
    if not la.get("success"):
        return 1
    attrs = la.get("data") or []
    doc_aliases: list[str] = []
    for a in attrs:
        if not isinstance(a, dict):
            continue
        t = (a.get("Attribute type") or a.get("type") or "").strip()
        if t.lower() == "document":
            alias = (a.get("Attribute system name") or a.get("alias") or "").strip()
            if alias:
                doc_aliases.append(alias)
    print("Document attribute system names:", doc_aliases)
    if not doc_aliases:
        print("No Document-type attributes on template.")
        return 0

    fv = fetch_record_field_values(RECORD_ID, doc_aliases)
    print("fetch_record_field_values:", "ok" if fv.get("success") else "FAIL", fv.get("error"))
    if not fv.get("success"):
        return 1
    row = _row_from_fetch(fv)
    print("Row keys sample:", list(row.keys())[:20] if row else "(empty)")

    fetched = 0
    for name in doc_aliases:
        raw = row.get(name)
        if raw is None:
            for k, v in row.items():
                if k.lower() == name.lower():
                    raw = v
                    break
        did = extract_platform_document_id(raw)
        print(f"  {name}: doc_id={did!r} raw_type={type(raw).__name__}")
        if not did:
            continue
        gm = get_document_model(did)
        m = gm.get("model")
        if not gm.get("success") or not isinstance(m, dict):
            print(f"    get_document_model FAILED: {gm.get('error')}")
            continue
        gc = get_document_content(did, document_model=m)
        if not gc.get("success"):
            print(f"    get_document_content FAILED: {gc.get('error')}")
            continue
        c = gc.get("content")
        ln = len(c) if isinstance(c, str) else 0
        print(
            f"    OK: mime={gc.get('mime_type')!r} file={gc.get('filename')!r} "
            f"b64_len={ln}"
        )
        fetched += 1
    print("Done. Attachments successfully fetched:", fetched)
    return 0 if fetched else 2


if __name__ == "__main__":
    raise SystemExit(main())
