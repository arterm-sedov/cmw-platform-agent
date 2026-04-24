"""
Live API: *Image* **attribute** (``webapi/Image``) on a **sandbox** app/template from **``.env``**.

Requires a working CMW **connection** in **``.env``** and::

    CMW_INTEGRATION_TESTS=1
    CMW_INTEGRATION_APP=YourAppSystemName
    CMW_INTEGRATION_TEMPLATE=YourTemplateSystemName

Default *Image* **field** system name: **``CMW_INTEGRATION_IMAGE_ATTR``** (see
**``integration_test_env``**); test sets **``IntegrationTestImage``** via **``setdefault``**,
not in **``.env``** unless you override. **``ensure_harness_image_attribute``** may **create** the
**attribute** with **``edit_or_create_image_attribute``** on the test **template** if needed.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import pytest

from tools._tests.cmw_integration_harness import ensure_harness_image_attribute
from tools._tests.integration_test_env import (
    CMW_INTEGRATION_APP,
    CMW_INTEGRATION_IMAGE_ATTR,
    CMW_INTEGRATION_TEMPLATE,
    CMW_INTEGRATION_TESTS,
    DEFAULT_INTEGRATION_IMAGE_ATTR_SYSTEM_NAME,
    integration_app_and_template,
)
from tools.platform_record_document import (
    extract_platform_document_id,
    fetch_record_field_values,
)
from tools.platform_record_image import get_image_model
from tools.templates_tools.tool_create_edit_record import create_edit_record
from tools.templates_tools.tool_record_image import (
    attach_file_to_record_image_attribute,
    fetch_record_image_file,
)

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
load_dotenv(_REPO / ".env")
os.environ.setdefault(
    CMW_INTEGRATION_IMAGE_ATTR,
    DEFAULT_INTEGRATION_IMAGE_ATTR_SYSTEM_NAME,
)

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

# Minimal 1x1 PNG (89 bytes) — no external file dependency.
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_create_record_attach_image_fetch_roundtrip(tmp_path: Path) -> None:
    t = integration_app_and_template()
    assert t is not None
    app, tpl = t
    img = ensure_harness_image_attribute()
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
    png = tmp_path / "probe.png"
    png.write_bytes(base64.standard_b64decode(TINY_PNG_B64))
    up = attach_file_to_record_image_attribute.invoke(
        {
            "record_id": rid,
            "attribute_system_name": img,
            "file_reference": str(png),
        }
    )
    assert up.get("success"), up

    fetch = fetch_record_image_file.invoke(
        {
            "record_id": rid,
            "image_attribute_system_name": img,
            "multivalue_index": 0,
            "agent": None,
        }
    )
    assert fetch.get("success"), fetch
    fr = str(fetch.get("file_reference") or "")
    assert os.path.isabs(fr)
    assert os.path.isfile(fr)

    fv = fetch_record_field_values(rid, [img])
    assert fv.get("success"), fv
    data = fv.get("data") or {}
    row = data.get(rid) or {}
    if not row and len(data) == 1:
        row = next(iter(data.values()))
    raw = row.get(img)
    for k, v in row.items():
        if k.lower() == img.lower() and raw is None:
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
