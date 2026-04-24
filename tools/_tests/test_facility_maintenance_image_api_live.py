"""
Live API: ``FacilityManagement`` / ``MaintenancePlans`` — image attribute (``webapi/Image``).

Requires ``.env`` and ::

    CMW_INTEGRATION_TESTS=1

Set ``CMW_MAINTENANCE_PLANS_IMAGE_ATTR`` to the **system name** of an *Image* attribute on
``MaintenancePlans`` (create the attribute in the platform first, same app as the document tests).
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import pytest

from tools.platform_record_document import (
    extract_platform_document_id,
    fetch_record_field_values,
)
from tools.platform_record_image import get_image_model
from tools.templates_tools.tool_create_edit_record import create_edit_record
from tools.templates_tools.tool_record_image import (
    attach_file_to_record_image_attribute,
)

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
load_dotenv(_REPO / ".env")

APP = "FacilityManagement"
TPL = "MaintenancePlans"
IMG = os.environ.get("CMW_MAINTENANCE_PLANS_IMAGE_ATTR", "").strip()

# Minimal 1x1 PNG (89 bytes) — no external file dependency.
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

pytestmark = pytest.mark.skipif(
    os.environ.get("CMW_INTEGRATION_TESTS", "").strip() != "1",
    reason="set CMW_INTEGRATION_TESTS=1 and configure .env for live API",
)


def _need_image_attr() -> None:
    if not IMG:
        pytest.skip(
            "set CMW_MAINTENANCE_PLANS_IMAGE_ATTR to the image attribute system name on MaintenancePlans"
        )


def test_create_record_attach_image_roundtrip() -> None:
    _need_image_attr()
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
    up = attach_file_to_record_image_attribute.invoke(
        {
            "record_id": rid,
            "attribute_system_name": IMG,
            "file_name": "probe.png",
            "file_base64": TINY_PNG_B64,
        }
    )
    assert up.get("success"), up

    fv = fetch_record_field_values(rid, [IMG])
    assert fv.get("success"), fv
    data = fv.get("data") or {}
    row = data.get(rid) or {}
    if not row and len(data) == 1:
        row = next(iter(data.values()))
    raw = row.get(IMG)
    for k, v in row.items():
        if k.lower() == IMG.lower() and raw is None:
            raw = v
            break
    assert raw is not None, row
    if isinstance(raw, list):
        raw = raw[0]
    iid = extract_platform_document_id(raw)
    assert iid, raw
    g = get_image_model(iid)
    assert g.get("success"), g
    m = g.get("model")
    assert isinstance(m, dict)
    assert m.get("content")
    c = base64.b64decode(str(m.get("content", "")), validate=False)
    assert c[:8] == b"\x89PNG\r\n\x1a\n"
