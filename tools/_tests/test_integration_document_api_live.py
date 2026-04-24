"""
Live API: *Document* **model** (title + extension) on a **sandbox** app/template from **``.env``**.

Requires a working CMW **connection** (repo **``.env``**). Enable with::

    CMW_INTEGRATION_TESTS=1
    CMW_INTEGRATION_APP=YourAppSystemName
    CMW_INTEGRATION_TEMPLATE=YourTemplateSystemName

Optional: **``CMW_INTEGRATION_DOCUMENT_ATTR``** — *Document* **field** system name (default **``Document``**).

**Document** **tests** **create** **records** and **attach** files, then re-read the same **record** id
(GetPropertyValues + GetDocument); no pre-seeded ids.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import pytest

from tools._tests.integration_test_env import (
    CMW_INTEGRATION_APP,
    CMW_INTEGRATION_TEMPLATE,
    CMW_INTEGRATION_TESTS,
    integration_app_and_template,
    integration_document_attr_system_name,
)
from tools.platform_record_document import (
    display_filename_for_registry,
    extract_platform_document_id,
    fetch_record_field_values,
    get_document_model,
)
from tools.templates_tools.tool_create_edit_record import create_edit_record
from tools.templates_tools.tool_record_document import (
    attach_file_to_record_document_attribute,
    fetch_record_document_file,
)
from tools.tools import read_text_based_file

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
load_dotenv(_REPO / ".env")

_INTEGRATION_OFF = os.environ.get(CMW_INTEGRATION_TESTS, "").strip() != "1"
_NO_TARGET = integration_app_and_template() is None

pytestmark = [
    pytest.mark.skipif(
        _INTEGRATION_OFF,
        reason="set CMW_INTEGRATION_TESTS=1 and configure .env for live API",
    ),
    pytest.mark.skipif(
        _NO_TARGET,
        reason=(
            f"set {CMW_INTEGRATION_APP} and {CMW_INTEGRATION_TEMPLATE} in .env to your "
            "sandbox application and template system names"
        ),
    ),
]


def _app_tpl_doc() -> tuple[str, str, str]:
    p = integration_app_and_template()
    assert p is not None
    a, t = p
    return a, t, integration_document_attr_system_name()


def _row_get_document_value(row: dict, doc_key: str) -> object | None:
    raw = row.get(doc_key)
    if raw is not None:
        return raw
    for k, v in row.items():
        if k.lower() == doc_key.lower():
            return v
    return None


def test_new_record_document_model_matches_upload(tmp_path: Path) -> None:
    app, tpl, doc = _app_tpl_doc()
    tiny = tmp_path / "api_probe.txt"
    tiny.write_text("cmw document model probe\n", encoding="utf-8")

    cr = create_edit_record.invoke(
        {
            "operation": "create",
            "application_system_name": app,
            "template_system_name": tpl,
            "values": {},
        }
    )
    assert cr.get("success"), cr
    rid = str(cr["record_id"])
    ar = attach_file_to_record_document_attribute.invoke(
        {
            "record_id": rid,
            "attribute_system_name": doc,
            "file_reference": str(tiny),
            "file_name": "api_probe.txt",
            "replace": True,
        }
    )
    assert ar.get("success"), ar

    model = _assert_get_document_model_for_record(rid, doc)
    reg = display_filename_for_registry(model)
    assert "api_probe" in (model.get("title") or "") or "api_probe" in reg, model


def test_fetch_record_document_file_after_attach_roundtrip(tmp_path: Path) -> None:
    """Exercises **``fetch_record_document_file``** + **``read_text_based_file``** (no agent)."""
    app, tpl, doc = _app_tpl_doc()
    tiny = tmp_path / "fetch_probe.md"
    tiny.write_text("# probe\n", encoding="utf-8")
    cr = create_edit_record.invoke(
        {
            "operation": "create",
            "application_system_name": app,
            "template_system_name": tpl,
            "values": {},
        }
    )
    assert cr.get("success"), cr
    rid = str(cr["record_id"])
    ar = attach_file_to_record_document_attribute.invoke(
        {
            "record_id": rid,
            "attribute_system_name": doc,
            "file_reference": str(tiny.resolve()),
            "file_name": "fetch_probe.md",
            "replace": True,
        }
    )
    assert ar.get("success"), ar
    out = fetch_record_document_file.invoke(
        {
            "record_id": rid,
            "document_attribute_system_name": doc,
            "multivalue_index": 0,
            "agent": None,
        }
    )
    assert out.get("success"), out
    fr = str(out.get("file_reference") or "")
    assert os.path.isabs(fr)
    assert os.path.isfile(fr)
    rtxt = read_text_based_file.invoke(
        {"file_reference": fr, "read_html_as_markdown": True, "agent": None}
    )
    s = str(rtxt or "")
    assert "probe" in s.lower()


def _assert_get_document_model_for_record(rid: str, doc: str) -> dict:
    """Re-read **record** by id: GetPropertyValues + GetDocument (no create)."""
    fv = fetch_record_field_values(rid, [doc])
    assert fv.get("success"), fv
    data = fv.get("data") or {}
    row = data.get(rid) or {}
    if not row and len(data) == 1:
        row = next(iter(data.values()))
    assert isinstance(row, dict), (rid, data)
    assert row, (rid, data)
    raw = _row_get_document_value(row, doc)
    assert raw is not None, row
    if isinstance(raw, list):
        assert raw, "empty document list"
        raw = raw[0]
    doc_id = extract_platform_document_id(raw)
    assert doc_id, raw
    gmod = get_document_model(doc_id)
    assert gmod.get("success"), gmod
    m = gmod["model"]
    assert isinstance(m, dict)
    reg = display_filename_for_registry(m)
    assert reg, m
    assert m.get("title")
    assert m.get("extension")
    return m


def test_document_get_model_re_read_by_record_id_after_attach(tmp_path: Path) -> None:
    """**Create** **record**, **attach** a file, then only **re-query** by **``record_id``** (persisted)."""
    app, tpl, doc = _app_tpl_doc()
    f = tmp_path / "reread_persisted.txt"
    f.write_text("reread by record id only after save\n", encoding="utf-8")
    cr = create_edit_record.invoke(
        {
            "operation": "create",
            "application_system_name": app,
            "template_system_name": tpl,
            "values": {},
        }
    )
    assert cr.get("success"), cr
    rid = str(cr["record_id"])
    ar = attach_file_to_record_document_attribute.invoke(
        {
            "record_id": rid,
            "attribute_system_name": doc,
            "file_reference": str(f.resolve()),
            "file_name": "reread_persisted.txt",
            "replace": True,
        }
    )
    assert ar.get("success"), ar
    _assert_get_document_model_for_record(rid, doc)
