"""CMW **image** HTTP helpers (``webapi/Image``; see ``cmw_open_api/web_api_v1.json``).

Images are **not** documents: there is no ``SetObjectImage`` in TeamNetwork. Upload is
``POST /webapi/Image/Create`` with :class:`FileContentModel` (name + base64). The response is
the new image id (string). A record image attribute stores that id; read bytes via
``GET /webapi/Image?imageId=…``, which returns :class:`ImageContentModel` (name, format, content).
"""

from __future__ import annotations

import base64
import binascii
import mimetypes
import tempfile
from typing import Any
import urllib.parse

from tools import requests_
from tools.platform_record_document import unwrap_webapi_payload

# Relative to server base (same convention as :mod:`tools.platform_record_document`).
WEBAPI_IMAGE = "webapi/Image"
WEBAPI_IMAGE_CREATE = "webapi/Image/Create"


def image_get_path(image_id: str) -> str:
    """``GET /webapi/Image?imageId=…`` (JSON body, not raw bytes for image)."""
    q = urllib.parse.urlencode({"imageId": image_id})
    return f"{WEBAPI_IMAGE}?{q}"


def get_image_model(image_id: str) -> dict[str, Any]:
    """``GET /webapi/Image?imageId=`` — image metadata and base64 ``content`` in JSON."""
    result = requests_._get_request(image_get_path(image_id))
    if not result.get("success"):
        return {
            "success": False,
            "error": result.get("error", "Get Image failed"),
            "model": None,
        }
    raw = result.get("raw_response")
    inner = unwrap_webapi_payload(raw) if raw is not None else None
    if not isinstance(inner, dict):
        return {
            "success": False,
            "error": "Unexpected Get Image response shape",
            "model": None,
        }
    if "name" in inner and "content" in inner:
        return {"success": True, "error": None, "model": inner}
    return {
        "success": False,
        "error": "Get Image model missing name or content",
        "model": None,
    }


def display_filename_for_image_model(model: dict[str, Any] | None) -> str | None:
    """
    File name for session/registry: prefer API ``name``, else ``image.{format}`` when
    ``format`` is set.
    """
    if not model:
        return None
    name = (str(model.get("name") or "")).strip()
    if name:
        return name
    fmt = (str(model.get("format") or "")).strip().lstrip(".")
    if not fmt:
        return None
    return f"image.{fmt.lower()}"


def get_image_file_payload(
    image_id: str,
    *,
    image_model: dict[str, Any],
) -> dict[str, Any]:
    """
    Return base64 file payload for tools (same idea as :func:`get_document_content`).

    Uses ``name`` / ``format`` from ``image_model`` (from :func:`get_image_model`); no byte sniffing.
    """
    content_b64 = image_model.get("content")
    if not isinstance(content_b64, str) or not str(content_b64).strip():
        return {
            "success": False,
            "error": "Get Image model has no base64 content field.",
        }
    display = display_filename_for_image_model(image_model) or f"{image_id}.img"
    suf = _suffix_for_display_name(display, image_model)
    mt, _ = mimetypes.guess_type(f"x{suf}")
    return {
        "success": True,
        "content": str(content_b64).strip(),
        "mime_type": mt,
        "filename": f"{image_id}{suf}",
    }


def _suffix_for_display_name(display: str, model: dict[str, Any]) -> str:
    d = display.strip()
    if "." in d:
        return d[d.rindex(".") :]
    fmt = (str(model.get("format") or "")).strip().lstrip(".")
    if not fmt:
        return ""
    return f".{fmt.lower()}"


def extract_created_id(result: dict[str, Any]) -> str | None:
    """New image or document id from a ``webapi/…/Create``-style JSON response."""
    if not result.get("success"):
        return None
    raw = result.get("raw_response")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if not isinstance(raw, dict):
        return None
    inner = unwrap_webapi_payload(raw)
    if isinstance(inner, str) and inner.strip():
        return inner.strip()
    if isinstance(inner, dict):
        s = inner.get("response")
        if isinstance(s, str) and s.strip():
            return s.strip()
    return None


def create_image_file(file_name: str, file_bytes: bytes) -> dict[str, Any]:
    """
    ``POST /webapi/Image/Create`` — :class:`FileContentModel` (name, content as base64 string).

    New id: use :func:`extract_created_id` on the result.
    """
    body: dict[str, Any] = {
        "name": file_name,
        "content": base64.b64encode(file_bytes).decode("ascii"),
    }
    return requests_._post_request(body, WEBAPI_IMAGE_CREATE)


def put_record_image_attribute_value(
    record_id: str,
    image_attribute_system_name: str,
    image_id: str,
) -> dict[str, Any]:
    """``PUT /webapi/Record/{id}`` with a single key — no TeamNetwork attach for image."""
    return requests_._put_request(
        {image_attribute_system_name: image_id},
        f"webapi/Record/{record_id}",
    )


def b64_to_temp_image_file(b64: str, suffix: str) -> tuple[str, str | None]:
    """Write base64 image body to a temp file (suffix includes a dot, e.g. ``.png``)."""
    try:
        data = base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError) as e:
        return "", f"Invalid base64: {str(e)}"
    suf = suffix if suffix.startswith(".") else f".{suffix}"
    try:
        fd, path = tempfile.mkstemp(suffix=suf)
        with open(fd, "wb") as f:
            f.write(data)
    except OSError as e:
        return "", str(e)
    return path, None


__all__ = [
    "WEBAPI_IMAGE",
    "WEBAPI_IMAGE_CREATE",
    "b64_to_temp_image_file",
    "create_image_file",
    "display_filename_for_image_model",
    "extract_created_id",
    "get_image_file_payload",
    "get_image_model",
    "image_get_path",
    "put_record_image_attribute_value",
]
