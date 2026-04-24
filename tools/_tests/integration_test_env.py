"""
Environment **variable names** for CMW **integration** tests (live API).

- **``CMW_INTEGRATION_TESTS=1``** — gate; often set in **``.env``** for local runs.
- **``CMW_INTEGRATION_APP``** / **``CMW_INTEGRATION_TEMPLATE``** — application and template
  **system** **names** for a sandbox; **required** for live **tests** (no NDA **names** in the repo).
- **``CMW_INTEGRATION_DOCUMENT_ATTR``** (optional) — *Document* **field** system name; defaults to
  **``Document``** if unset.
- **``CMW_INTEGRATION_IMAGE_ATTR``** (optional) — *Image* **field** to use or **create**; see
  :func:`tools._tests.cmw_integration_harness.ensure_harness_image_attribute`.

Tests use **``os.environ.setdefault``** for image defaults only, so a shell or **``.env``** can
override, e.g. ``$env:CMW_INTEGRATION_IMAGE_ATTR = "MyImageAttr"`` (PowerShell).
"""

from __future__ import annotations

import os

# Gate: set to **``1``** in the environment (often via **``.env``** for local runs).
CMW_INTEGRATION_TESTS = "CMW_INTEGRATION_TESTS"

# Sandbox **target** (required for live app/template **tests**).
CMW_INTEGRATION_APP = "CMW_INTEGRATION_APP"
CMW_INTEGRATION_TEMPLATE = "CMW_INTEGRATION_TEMPLATE"

# *Document* **attribute** system name on the template (optional; common platform default).
CMW_INTEGRATION_DOCUMENT_ATTR = "CMW_INTEGRATION_DOCUMENT_ATTR"
DEFAULT_INTEGRATION_DOCUMENT_ATTR_SYSTEM_NAME = "Document"

# System name of the *Image* **attribute** to use, or to **create** on the test app/template
# (see :func:`tools._tests.cmw_integration_harness.ensure_harness_image_attribute`).
CMW_INTEGRATION_IMAGE_ATTR = "CMW_INTEGRATION_IMAGE_ATTR"
DEFAULT_INTEGRATION_IMAGE_ATTR_SYSTEM_NAME = "IntegrationTestImage"


def integration_app_and_template() -> tuple[str, str] | None:
    """**Application** and **template** system names from the environment, or **``None``** if either missing."""
    a = (os.environ.get(CMW_INTEGRATION_APP) or "").strip()
    t = (os.environ.get(CMW_INTEGRATION_TEMPLATE) or "").strip()
    if not a or not t:
        return None
    return (a, t)


def integration_document_attr_system_name() -> str:
    """*Document* **field** system name: **env** or :data:`DEFAULT_INTEGRATION_DOCUMENT_ATTR_SYSTEM_NAME`."""
    v = (os.environ.get(CMW_INTEGRATION_DOCUMENT_ATTR) or "").strip()
    return v or DEFAULT_INTEGRATION_DOCUMENT_ATTR_SYSTEM_NAME
