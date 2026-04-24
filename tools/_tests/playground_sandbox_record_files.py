"""
**Interactive** sandbox: **app** + **template** from **``CMW_INTEGRATION_APP``** /
**``CMW_INTEGRATION_TEMPLATE``** in **``.env``**.

**Creates** **records**, **attaches** text / **PDF** / **markdown** (and optional **audio** / **video** paths
as **document** binaries), **replaces** **Document** and **Image** **fields**, and tries to **clear**
via **``PUT /webapi/Record/{id}``** with JSON **``null``** (platform-dependent).

**Run** (from repo root, venv on)::

    python tools/_tests/playground_sandbox_record_files.py
    python tools/_tests/playground_sandbox_record_files.py --video C:\\path\\to\\clip.mp4
"""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from dotenv import load_dotenv

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
load_dotenv(_REPO / ".env")

from tools import requests_  # noqa: E402
from tools._tests.cmw_integration_harness import (  # noqa: E402
    ensure_harness_image_attribute,
)
from tools._tests.integration_test_env import (  # noqa: E402
    CMW_INTEGRATION_APP,
    CMW_INTEGRATION_TEMPLATE,
    integration_app_and_template,
    integration_document_attr_system_name,
)
from tools.platform_record_document import (  # noqa: E402
    display_filename_for_registry,
    extract_platform_document_id,
    fetch_record_field_values,
    get_document_model,
)
from tools.templates_tools.tool_create_edit_record import (  # noqa: E402
    create_edit_record,
)
from tools.templates_tools.tool_record_document import (  # noqa: E402
    attach_file_to_record_document_attribute,
    fetch_record_document_file,
)
from tools.templates_tools.tool_record_image import (  # noqa: E402
    attach_file_to_record_image_attribute,
)

# Minimal 1x1 PNG (see integration image test).
TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
# Short PDF header (enough to register as a document file on many platforms).
TINY_PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 2 2]>>endobj
trailer<</Size 4/Root 1 0 R>>
%%EOF
"""


def _p(msg: str, *args: Any) -> None:
    print(f"[playground] {msg}", *args, flush=True)


def _row_val(row: dict[str, Any], key: str) -> object | None:
    v = row.get(key)
    if v is not None:
        return v
    for k, x in row.items():
        if k.lower() == key.lower():
            return x
    return None


def _put_clear_property(record_id: str, property_system_name: str) -> dict[str, Any]:
    """Try **``PUT /webapi/Record/{id}``** with **``{ "Attr": null }``** to detach."""
    return requests_._put_request(
        {property_system_name: None},
        f"webapi/Record/{record_id}",
    )


def _doc_title_hint(record_id: str, doc: str) -> str:
    fv = fetch_record_field_values(record_id, [doc])
    if not fv.get("success"):
        return f"(fetch failed: {fv.get('error')})"
    data = fv.get("data") or {}
    row = data.get(record_id) or (
        next(iter(data.values()), {}) if len(data) == 1 else {}
    )
    if not isinstance(row, dict):
        return "?"
    raw = _row_val(row, doc)
    if raw is None:
        return "<empty>"
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    did = extract_platform_document_id(raw) if raw is not None else None
    if not did:
        return f"{raw!r}"
    m = get_document_model(did)
    if m.get("success") and isinstance(m.get("model"), dict):
        mo = m["model"]
        reg = display_filename_for_registry(mo) or str(mo.get("title") or mo.get("name") or "")
        return f"{reg} id={did[:8]}…" if reg else f"id={did[:12]}…"
    return f"id={did[:12]}…"


def _write_temps() -> dict[str, Path]:
    d = Path(tempfile.mkdtemp(prefix="cmw_play_"))
    out: dict[str, Path] = {}
    out["note_txt"] = d / "play_note.txt"
    out["note_txt"].write_text("Playground: text as document\n", encoding="utf-8")
    out["note_md"] = d / "play_note.md"
    out["note_md"].write_text("# Play\n\n*markdown*\n", encoding="utf-8")
    out["tiny_pdf"] = d / "play_tiny.pdf"
    out["tiny_pdf"].write_bytes(TINY_PDF)
    p1 = d / "img_a.png"
    p1.write_bytes(base64.standard_b64decode(TINY_PNG_B64))
    out["img_a"] = p1
    p2 = d / "img_b.png"
    p2.write_bytes(
        base64.standard_b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
    )
    out["img_b"] = p2
    return out


def _attach_doc(rid: str, doc: str, path: Path, *, name: str, replace: bool) -> bool:
    r = attach_file_to_record_document_attribute.invoke(
        {
            "record_id": rid,
            "attribute_system_name": doc,
            "file_reference": str(path.resolve()),
            "file_name": name,
            "replace": replace,
        }
    )
    ok = bool(r.get("success"))
    if ok:
        _p(
            "document attached",
            name,
            "replace=",
            replace,
            "→",
            _doc_title_hint(rid, doc),
        )
    else:
        _p("document attach failed", r.get("error", r))
    return ok


def _attach_image(rid: str, img: str, path: Path) -> bool:
    r = attach_file_to_record_image_attribute.invoke(
        {
            "record_id": rid,
            "attribute_system_name": img,
            "file_reference": str(path.resolve()),
        }
    )
    ok = bool(r.get("success"))
    if ok:
        _p(
            "image attached",
            path.name,
            "new image_id=",
            r.get("image_id", "")[:20],
            "…",
        )
    else:
        _p("image attach failed", r.get("error", r))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Create records on the integration template; attach and replace files."
    )
    ap.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Optional .mp3/.wav path: attached as a **document** (not audio attribute).",
    )
    ap.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional .mp4 path: attached as a **document** (binary).",
    )
    args = ap.parse_args()
    t = integration_app_and_template()
    if not t:
        _p(
            f"set {CMW_INTEGRATION_APP} and {CMW_INTEGRATION_TEMPLATE} in .env first.",
        )
        return 1
    app, tpl = t
    doc = integration_document_attr_system_name()
    _p("app=", app, "template=", tpl, "Document attr=", doc)

    img = ensure_harness_image_attribute()
    _p("Image attr=", img)

    tmp = _write_temps()
    _p("temp files under", str(next(iter(tmp.values())).parent))

    # —— Record 1: document lifecycle ——
    cr1 = create_edit_record.invoke(
        {
            "operation": "create",
            "application_system_name": app,
            "template_system_name": tpl,
            "values": {},
        }
    )
    if not cr1.get("success"):
        _p("create record 1 failed", cr1)
        return 1
    r1 = str(cr1["record_id"])
    _p("record 1 (document playground)", r1)

    if not _attach_doc(r1, doc, tmp["note_txt"], name="play_note.txt", replace=True):
        return 1
    if not _attach_doc(r1, doc, tmp["tiny_pdf"], name="play_tiny.pdf", replace=True):
        return 1
    if not _attach_doc(r1, doc, tmp["note_md"], name="play_note.md", replace=True):
        return 1
    fdoc = fetch_record_document_file.invoke(
        {
            "record_id": r1,
            "document_attribute_system_name": doc,
            "multivalue_index": 0,
            "agent": None,
        }
    )
    if fdoc.get("success"):
        _p("fetched last document to local", fdoc.get("file_reference", "")[:64], "…")
    else:
        _p("fetch_record_document_file", fdoc.get("error", fdoc))

    if args.video and args.video.is_file():
        if not _attach_doc(r1, doc, args.video, name=args.video.name, replace=True):
            return 1
    else:
        _p("skip --video: not set or not a file (pass a small .mp4 to test)")

    if args.audio and args.audio.is_file():
        if not _attach_doc(r1, doc, args.audio, name=args.audio.name, replace=True):
            return 1
    else:
        _p("skip --audio: not set (pass an .mp3 to test as **document** binary)")

    clr = _put_clear_property(r1, doc)
    if clr.get("success") and not clr.get("error"):
        _p("PUT clear Document: OK (property may be empty now)")
    else:
        _p(
            "PUT null Document: HTTP/API result",
            clr.get("success"),
            clr.get("error", clr.get("raw_response", ""))[:200],
        )

    # —— Record 2: image replace ——
    cr2 = create_edit_record.invoke(
        {
            "operation": "create",
            "application_system_name": app,
            "template_system_name": tpl,
            "values": {},
        }
    )
    if not cr2.get("success"):
        _p("create record 2 failed", cr2)
        return 1
    r2 = str(cr2["record_id"])
    _p("record 2 (image playground)", r2)
    if not _attach_image(r2, img, tmp["img_a"]):
        return 1
    if not _attach_image(r2, img, tmp["img_b"]):
        return 1

    iclr = _put_clear_property(r2, img)
    if iclr.get("success") and not iclr.get("error"):
        _p("PUT clear Image: OK")
    else:
        _p(
            "PUT null Image",
            iclr.get("success"),
            (iclr.get("error") or str(iclr.get("raw_response")))[:200],
        )

    _p("done. You can open records in the app:", r1, r2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
