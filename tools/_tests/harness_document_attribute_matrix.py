"""
Agentic + direct-API live harness: FacilityManagement / MaintenancePlans / Document.

- ``get_llm_manager().get_tools()`` must expose the same @tool s as the running agent.
- **Direct (same HTTP as tools use):** ``get_document_model`` + ``get_document_content`` on
  each document id (no agent).
- **Tool path (what the agent uses):** ``fetch_record_document_file`` + ``read_text_based_file`` —
  re-reads the record and calls the same GetDocument/Content stack internally; compare
  ``display`` vs the direct model ``title``+``extension`` line.

  python tools/_tests/harness_document_attribute_matrix.py
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dotenv import load_dotenv

load_dotenv(_REPO / ".env")

from agent_ng.llm_manager import get_llm_manager

from tools.attributes_tools.tool_get_attribute import get_attribute
from tools.attributes_tools.tools_document_attribute import (
    edit_or_create_document_attribute,
)
from tools.local_path_text import read_local_path_to_plain_text
from tools.platform_record_document import (
    display_filename_for_registry,
    extract_platform_document_id,
    fetch_record_field_values,
    get_document_content,
    get_document_model,
)
from tools.templates_tools.tool_create_edit_record import create_edit_record
from tools.templates_tools.tool_record_document import (
    attach_file_to_record_document_attribute,
    fetch_record_document_file,
)
from tools.tools import read_text_based_file

APP = "FacilityManagement"
TPL = "MaintenancePlans"
DOC_SYS = "Document"

# Fixed matrix (all must exist on disk) — user-provided set for MaintenancePlans / Document
FILE_PATHS: list[Path] = [
    Path(
        r"c:\OneDrive\Documents\Вакансия Технический писатель (английский язык) в Москве, "
        r"работа в компании Comindware (вакансия в архиве c 20 мая 2021).pdf"
    ),
    Path(r"c:\OneDrive\Documents\CMW_Copilot_20250920_184805.md"),
    Path(r"c:\OneDrive\Documents\gradio-screen-recording-2025-09-09T21-02-37.mp4"),
    Path(r"c:\OneDrive\Documents\Вопросы Wildberries.docx"),
    Path(r"c:\OneDrive\Music\tni.mp3"),
    Path(r"C:\OneDrive\Pictures\photo_2024-06-30_05-25-12.jpg"),
]


def _registry_check() -> None:
    names = {t.name for t in get_llm_manager().get_tools()}
    for req in (
        "attach_file_to_record_document_attribute",
        "fetch_record_document_file",
        "create_edit_record",
        "edit_or_create_document_attribute",
        "get_attribute",
    ):
        if req not in names:
            msg = f"missing tool in LLMManager registry: {req}"
            raise RuntimeError(msg)
    print("harness: LLMManager tool set OK.")


def _row_document_field(row: dict) -> list[Any]:
    for k, v in row.items():
        if k.lower() == DOC_SYS.lower():
            return v if isinstance(v, list) else [v] if v is not None else []
    return []


def _direct_get_document_probes(dids: list[str], upload_names: list[str] | None) -> None:
    """GET ``/webapi/Document/{{id}}`` (authoritative name) and ``/Content`` (body)."""
    print("\n--- direct API: GetDocument (model) + get_document_content ---")
    for i, did in enumerate(dids):
        if not did:
            continue
        gm = get_document_model(did)
        print(f"  {did} -> GetDocument success={gm.get('success')}")
        if not gm.get("success"):
            print(f"    error: {gm.get('error')}")
            continue
        m = gm.get("model") or {}
        reg = display_filename_for_registry(m) if isinstance(m, dict) else None
        print(
            f"    model: title={m.get('title')!r} extension={m.get('extension')!r} "
            f"-> registry name={reg!r}",
        )
        if upload_names and i < len(upload_names):
            up = upload_names[i]
            up_base = up.name if isinstance(up, Path) else str(up)
            if reg and up_base != reg and Path(reg).name != up_base:
                print(f"    note: upload was {up_base!r} vs model registry {reg!r}")
        r = get_document_content(did, document_model=m)
        ok = r.get("success")
        if not ok:
            print(f"    Content: error: {r.get('error')}")
            continue
        fn = r.get("filename")
        mt = r.get("mime_type")
        cl = len(r.get("content") or "")
        # filename here is {documentId}{ext from model} — not from the HTTP body.
        print(f"    get_document_content filename: {fn!r}  (id + model extension)")
        print(f"    mime (from ext): {mt!r}  b64_str_len: {cl}")


def _local_preview(p: Path, max_c: int = 320) -> str:
    t, err, _ = read_local_path_to_plain_text(str(p), read_html_as_markdown=True)
    if err:
        return f"<local err: {err}>"
    t = (t or "").strip()
    return (t[:max_c] + "…") if len(t) > max_c else (t or "<empty>")


def main() -> int:
    _registry_check()
    files = [p for p in FILE_PATHS if p.is_file()]
    if len(files) != len(FILE_PATHS):
        for p in FILE_PATHS:
            if not p.is_file():
                print("MISSING:", p)
        return 1
    for p in files:
        print(f"  file: {p.name} ({p.stat().st_size} b)")

    ga = get_attribute.invoke(
        {
            "application_system_name": APP,
            "template_system_name": TPL,
            "system_name": DOC_SYS,
        }
    )
    if not ga.get("success"):
        print("get_attribute failed:", ga.get("error"))
        return 1
    attr_name = (ga.get("Name") or ga.get("name") or "Document").strip()
    edit_or_create_document_attribute.invoke(
        {
            "operation": "edit",
            "name": attr_name,
            "system_name": DOC_SYS,
            "application_system_name": APP,
            "template_system_name": TPL,
            "store_multiple_values": True,
        }
    )
    cr = create_edit_record.invoke(
        {
            "operation": "create",
            "application_system_name": APP,
            "template_system_name": TPL,
            "values": {},
        }
    )
    if not cr.get("success"):
        print("create failed:", cr)
        return 1
    rid = str(cr["record_id"])
    print("record_id:", rid)

    for i, p in enumerate(files):
        b64 = base64.standard_b64encode(p.read_bytes()).decode("ascii")
        ar = attach_file_to_record_document_attribute.invoke(
            {
                "record_id": rid,
                "attribute_system_name": DOC_SYS,
                "file_name": p.name,
                "file_base64": b64,
                "replace": i == 0,
            }
        )
        print(f"  attach [{i}] {p.name!r} -> {ar.get('success')} {ar.get('error') or ''}")
        if not ar.get("success"):
            return 1

    fv = fetch_record_field_values(rid, [DOC_SYS])
    if not fv.get("success"):
        print("GetPropertyValues failed:", fv.get("error"))
        return 1
    # Same shape as :func:`fetch_record_field_values` (single keyed row per record_id).
    row = (fv.get("data") or {}).get(rid) or {}
    print("  property bag keys:", list(row.keys()) if isinstance(row, dict) else row)

    raw_list = _row_document_field(row if isinstance(row, dict) else {})
    dids: list[str] = []
    for item in raw_list:
        eid = extract_platform_document_id(item)
        if eid:
            dids.append(eid)
    print("  document id list (platform order):", dids)
    order_newest_first = list(reversed(files))
    _direct_get_document_probes(
        dids, [p.name for p in order_newest_first[: len(dids)]]
    )

    n = min(len(dids), len(order_newest_first), 8)
    print("\n--- fetch_record_document_file + read_text_based_file (no agent = abs path) ---")
    for i in range(n):
        p = order_newest_first[i]
        loc = _local_preview(p, 400)
        reg = fetch_record_document_file.invoke(
            {
                "record_id": rid,
                "document_attribute_system_name": DOC_SYS,
                "multivalue_index": i,
                "agent": None,
            }
        )
        if not reg.get("success"):
            print(f"  [idx {i}] {p.name!r}  register: {reg}")
            return 1
        ref = reg.get("file_reference")
        rtxt = read_text_based_file.invoke(
            {
                "file_reference": ref,
                "read_html_as_markdown": True,
                "agent": None,
            }
        )
        tlen = len(str(rtxt or ""))
        print(
            f"  [idx {i}] {p.name!r}  local≈{len(loc)} chars  register_ok={reg.get('success')}  "
            f"read_len={tlen}  display={reg.get('display_filename')!r}  "
            f"content_fileName={reg.get('content_fileName')!r}  ref=abs"
        )
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
