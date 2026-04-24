"""
Live API helpers: **app** + **template** from **``CMW_INTEGRATION_APP``** / **``CMW_INTEGRATION_TEMPLATE``**
(see **``integration_test_env``**), and **Image** field handling for tests.

Create an *Image* **attribute** with **``edit_or_create_image_attribute``** if needed. The
*Image* **field** system name is **``CMW_INTEGRATION_IMAGE_ATTR``**; unit tests
**``setdefault``** **``IntegrationTestImage``** when unset, while **app** / **template** are only
**from** the environment.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from tools._tests.integration_test_env import (
    CMW_INTEGRATION_IMAGE_ATTR,
    DEFAULT_INTEGRATION_IMAGE_ATTR_SYSTEM_NAME,
    integration_app_and_template,
)
from tools.attributes_tools.tool_get_attribute import get_attribute
from tools.attributes_tools.tools_image_attribute import edit_or_create_image_attribute
from tools.templates_tools.tool_list_attributes import list_attributes

logger = logging.getLogger(__name__)


def _require_app_template() -> tuple[str, str]:
    p = integration_app_and_template()
    if not p:
        msg = (
            "set CMW_INTEGRATION_APP and CMW_INTEGRATION_TEMPLATE in the environment "
            "(e.g. .env) to your sandbox application and template system names"
        )
        raise RuntimeError(msg)
    return p


def integration_image_attr_system_name() -> str:
    """Resolved *Image* **attribute** system name: env, else default (never empty)."""
    v = (os.environ.get(CMW_INTEGRATION_IMAGE_ATTR) or "").strip()
    return v or DEFAULT_INTEGRATION_IMAGE_ATTR_SYSTEM_NAME


def _attr_system_name(a: dict[str, Any]) -> str:
    for k in (
        "Attribute system name",
        "SystemName",
        "System name",
        "systemName",
    ):
        v = a.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _attr_type(a: dict[str, Any]) -> str:
    for k in ("Attribute type", "Type", "type", "AttributeType"):
        v = a.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def first_image_attribute_system_name() -> str | None:
    """First *Image* **attribute** on the **template** (if any), by list API."""
    app, tpl = _require_app_template()
    out = list_attributes.invoke(
        {"application_system_name": app, "template_system_name": tpl}
    )
    if not out.get("success") or not isinstance(out.get("data"), list):
        return None
    for a in out["data"]:
        if not isinstance(a, dict):
            continue
        t = _attr_type(a).lower()
        sn = _attr_system_name(a)
        if sn and "image" in t and "document" not in t:
            return sn
    return None


def ensure_harness_image_attribute() -> str:
    """
    Ensure a usable *Image* **attribute** for tests.

    - If **get_attribute** for the **integration** system name (env/default) **succeeds**, return
      it.
    - Else if any *Image* **attribute** **exists** on the **template**, return that.
    - Else **create** an *Image* **attribute** with **``edit_or_create_image_attribute``** using
      the **integration** system name and a neutral display name.
    """
    app, tpl = _require_app_template()
    target = integration_image_attr_system_name()
    got = get_attribute.invoke(
        {
            "application_system_name": app,
            "template_system_name": tpl,
            "system_name": target,
        }
    )
    if got.get("success"):
        return target
    ext = first_image_attribute_system_name()
    if ext:
        return ext
    cr = edit_or_create_image_attribute.invoke(
        {
            "operation": "create",
            "name": "Integration test image",
            "system_name": target,
            "application_system_name": app,
            "template_system_name": tpl,
            "rendering_color_mode": "Original",
            "description": "Image attribute for API integration tests (safe to delete in modeler).",
            "use_to_search_records": False,
        }
    )
    if cr.get("success"):
        return target
    ext2 = first_image_attribute_system_name()
    if ext2:
        return ext2
    gr = get_attribute.invoke(
        {
            "application_system_name": app,
            "template_system_name": tpl,
            "system_name": target,
        }
    )
    if gr.get("success"):
        return target
    msg = f"Could not get or create Image attribute on {app}.{tpl}: {cr.get('error', cr)}"
    raise RuntimeError(msg)
