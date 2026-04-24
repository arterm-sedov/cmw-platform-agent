"""
Live API: FacilityManagement / MaintenancePlans — ``DocumentModel`` (title + extension).

Requires a working CMW connection (repo ``.env``). Enable with::

    CMW_INTEGRATION_TESTS=1

Optional: ``CMW_FACILITY_TEST_RECORD_ID`` = existing record id with a document attribute
for a read-only model probe.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import pytest

from tools.platform_record_document import (
    display_filename_for_registry,
    extract_platform_document_id,
    fetch_record_field_values,
    get_document_model,
)
from tools.templates_tools.tool_create_edit_record import create_edit_record
from tools.templates_tools.tool_record_document import (
    attach_file_to_record_document_attribute,
)

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
load_dotenv(_REPO / ".env")

APP = "FacilityManagement"
TPL = "MaintenancePlans"
DOC = "Document"

pytestmark = pytest.mark.skipif(
    os.environ.get("CMW_INTEGRATION_TESTS", "").strip() != "1",
    reason="set CMW_INTEGRATION_TESTS=1 and configure .env for live API",
)


def _row_get_document_value(row: dict) -> object | None:
    raw = row.get(DOC)
    if raw is not None:
        return raw
    for k, v in row.items():
        if k.lower() == DOC.lower():
            return v
    return None


def test_new_record_document_model_matches_upload(tmp_path: Path) -> None:
    tiny = tmp_path / "api_probe.txt"
    tiny.write_text("cmw document model probe\n", encoding="utf-8")
    b64 = base64.standard_b64encode(tiny.read_bytes()).decode("ascii")

    cr = create_edit_record.invoke(
        {
            "operation": "create",
            "application_system_name": APP,
            "template_system_name": TPL,
            "values": {},
        }
    )
    assert cr.get("success"), cr
    rid = str(cr["record_id"])
    ar = attach_file_to_record_document_attribute.invoke(
        {
            "record_id": rid,
            "attribute_system_name": DOC,
            "file_name": "api_probe.txt",
            "file_base64": b64,
            "replace": True,
        }
    )
    assert ar.get("success"), ar

    fv = fetch_record_field_values(rid, [DOC])
    assert fv.get("success"), fv
    data = fv.get("data") or {}
    row = data.get(rid) or {}
    if not row and len(data) == 1:
        row = next(iter(data.values()))
    assert isinstance(row, dict)
    raw = _row_get_document_value(row)
    assert raw is not None, row
    if isinstance(raw, list):
        raw = raw[0]
    doc_id = extract_platform_document_id(raw)
    assert doc_id, raw

    gmod = get_document_model(doc_id)
    assert gmod.get("success"), gmod
    model = gmod["model"]
    assert isinstance(model, dict)
    reg = display_filename_for_registry(model)
    assert reg, model
    assert "api_probe" in (model.get("title") or "") or "api_probe" in reg, model


def test_existing_record_id_env_document_model() -> None:
    rid = os.environ.get("CMW_FACILITY_TEST_RECORD_ID", "").strip()
    if not rid:
        pytest.skip("set CMW_FACILITY_TEST_RECORD_ID to probe an existing document")

    fv = fetch_record_field_values(rid, [DOC])
    assert fv.get("success"), fv
    data = fv.get("data") or {}
    row = data.get(rid) or {}
    if not row and len(data) == 1:
        row = next(iter(data.values()))
    if not isinstance(row, dict) or not row:
        pytest.skip("no row in GetPropertyValues")
    raw = _row_get_document_value(row)
    if raw is None:
        pytest.skip("no Document on this record")
    if isinstance(raw, list):
        if not raw:
            pytest.skip("empty document list")
        raw = raw[0]
    doc_id = extract_platform_document_id(raw)
    if not doc_id:
        pytest.skip("could not parse document id")
    gmod = get_document_model(doc_id)
    assert gmod.get("success"), gmod
    m = gmod["model"]
    assert isinstance(m, dict)
    reg = display_filename_for_registry(m)
    assert reg, m
    assert m.get("title")
    assert m.get("extension")
