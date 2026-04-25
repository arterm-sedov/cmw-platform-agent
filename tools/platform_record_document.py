"""CMW document HTTP helpers (for tools, tests, and direct API checks — not ad-hoc agent HTTP).

**Metadata** — ``GET /webapi/Document/{id}`` returns ``DocumentModel`` with **title** and
**extension** (see ``cmw_open_api/web_api_v1.json``). In the product UI, the document file name
is not split into "title" vs "name"; the API still exposes *two* string fields. **Use only**
:func:`display_filename_for_registry` to get one logical file name: it is **fully determined**
by those two values (no byte sniffing, no heuristics on Content URL).

**Rules for** :func:`display_filename_for_registry` — normalize ``extension`` to start with
``.``; if ``title`` already ends with that suffix (case-insensitive), return ``title``; else
return ``{title}{extension}``. Same inputs always yield the same result.

**Content** — ``GET /webapi/Document/{id}/Content`` returns the **raw file** body with no
``Content-Type`` in our client flow. :func:`get_document_content` names bytes as
``{id}{extension}``; ``mime_type`` in the result is only ``mimetypes.guess_type`` for a dummy
name ending in that suffix (not sniffed, not from the server). On the CMW test tenant,
``GetDocument`` returned both ``title`` and ``extension`` for every file—see
``tools/_tests/harness_document_attribute_matrix.py``.
"""

from __future__ import annotations

import base64
import mimetypes
from typing import Any

from tools import requests_
from tools.cmw_webapi import extract_platform_document_id, unwrap_webapi_payload

# TeamNetwork + webapi paths (relative to server base; no leading slash). Only this module
# builds the bodies for them; :mod:`tools.requests_` is transport.
# Response parsing (unwrap, ID extraction) is now in shared :mod:`tools.cmw_webapi`
# to eliminate duplication and cross-module imports.
GET_PROPERTY_VALUES = "api/public/system/TeamNetwork/ObjectService/GetPropertyValues"
SET_OBJECT_DOCUMENT = "api/public/system/TeamNetwork/DocumentService/SetObjectDocument"


def document_content_get_path(document_id: str) -> str:
    """``GET`` ``webapi/Document/{id}/Content`` (raw body via :func:`requests_._get_url_binary`)."""
    return f"webapi/Document/{document_id}/Content"


def fetch_record_field_values(
    record_id: str,
    attribute_system_names: list[str],
) -> dict[str, Any]:
    """
    Load selected attribute values for a record (TeamNetwork GetPropertyValues).

    **Empty** ``propertiesByAlias`` (``attribute_system_names == []``) is not a “return all
    fields” mode: on a live tenant, the row often contains only ``id`` (and no template
    attributes). Pass explicit system names (e.g. from **list_attributes**).

    Returns:
        success, data: ``{ record_id: { attr_alias: value, ... } }`` or error.
    """
    body: dict[str, Any] = {
        "objects": [record_id],
        "propertiesByAlias": list(attribute_system_names),
    }
    result = requests_._post_request(body, GET_PROPERTY_VALUES)
    if not result.get("success"):
        return {
            "success": False,
            "status_code": int(result.get("status_code", 0) or 0),
            "data": None,
            "error": result.get("error") or "Request failed",
        }
    raw = result.get("raw_response")
    inner = unwrap_webapi_payload(raw)
    if not isinstance(inner, dict):
        return {
            "success": False,
            "status_code": int(result.get("status_code", 0) or 0),
            "data": None,
            "error": "Unexpected GetPropertyValues response shape",
        }
    row = inner.get(record_id, {})
    if not isinstance(row, dict):
        row = {}
    return {
        "success": True,
        "status_code": int(result.get("status_code", 0) or 0),
        "data": {record_id: row},
        "error": None,
    }


def resolve_id_from_record_property(
    record_id: str,
    attribute_system_name: str,
    multivalue_index: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Return the stored file / media id for a single property (document, image, etc.).

    Uses the same GetPropertyValues row shape as the UI: value may be a string id, ``{"id": ...}``,
    or a list for multivalue attributes. The API may key the row with a different casing than
    ``attribute_system_name``; lookup is case-insensitive.

    On failure, return (error_dict, None). On success, return (None, id string).
    """
    g = fetch_record_field_values(record_id, [attribute_system_name])
    if not g.get("success"):
        return (
            {
                "success": False,
                "error": g.get("error") or "Failed to read record",
            },
            None,
        )
    data = (g.get("data") or {}).get(record_id) or {}
    raw_val = data.get(attribute_system_name)
    if raw_val is None and isinstance(data, dict):
        for k, v in data.items():
            if k.lower() == attribute_system_name.lower():
                raw_val = v
                break
    if raw_val in (None, ""):
        return (
            {
                "success": False,
                "error": "Attribute is empty or missing on this record.",
            },
            None,
        )
    if isinstance(raw_val, list):
        if multivalue_index >= len(raw_val):
            return (
                {
                    "success": False,
                    "error": (
                        f"multivalue_index {multivalue_index} out of range "
                        f"(len={len(raw_val)})."
                    ),
                },
                None,
            )
        raw_val = raw_val[multivalue_index]
    ref = extract_platform_document_id(raw_val)
    if not ref:
        return (
            {
                "success": False,
                "error": "Could not resolve a file id from the attribute value.",
            },
            None,
        )
    return (None, ref)


def get_document_model(document_id: str) -> dict[str, Any]:
    """
    ``GET /webapi/Document/{id}`` — :class:`Comindware.Platform.Contracts.DocumentModel``.

    The wire format stores **title** and **extension** as separate strings. Do not assume
    whether **title** already contains a file suffix; always combine with
    :func:`display_filename_for_registry` for the one string used in the app registry.
    """
    result = requests_._get_request(f"webapi/Document/{document_id}")
    if not result.get("success"):
        return {
            "success": False,
            "error": result.get("error", "Request failed"),
            "model": None,
        }
    raw = result.get("raw_response")
    inner = unwrap_webapi_payload(raw)
    if not isinstance(inner, dict):
        return {
            "success": False,
            "error": "Unexpected GetDocument response shape",
            "model": None,
        }
    return {"success": True, "error": None, "model": inner}


def display_filename_for_registry(model: dict[str, Any] | None) -> str | None:
    """
    Build the full logical **filename** for chat/session file registry from ``title`` and ``extension``.

    **Deterministic:** normalize ``extension`` to ``.*suffix*``; if ``title`` already ends
    with the same suffix (ignoring case), return ``title`` unchanged; else return
    ``{title}{extension}``.

    Returns ``None`` if either field is missing/blank. Callers (e.g. :func:`fetch_record_document_file`)
    treat that as a **hard error**; there is no filename fallback. CMW file-backed documents
    in our live checks always returned both fields.
    """
    if not model:
        return None
    title = (str(model.get("title") or "")).strip()
    ext = (str(model.get("extension") or "")).strip()
    if not title or not ext:
        return None
    if not ext.startswith("."):
        ext = f".{ext}"
    if title.lower().endswith(ext.lower()):
        return title
    return f"{title}{ext}"


def _extension_suffix_from_model(model: dict[str, Any] | None) -> str | None:
    """``DocumentModel.extension`` as a suffix including the dot, or None."""
    if not model:
        return None
    ext = (str(model.get("extension") or "")).strip()
    if not ext:
        return None
    return ext if ext.startswith(".") else f".{ext}"


def get_document_content(
    document_id: str,
    *,
    document_model: dict[str, Any],
) -> dict[str, Any]:
    """
    ``GET /webapi/Document/{id}/Content`` — body is the **raw** file (same bytes as the
    uploaded or stored file on the CMW test tenant). Pass ``document_model`` from
    :func:`get_document_model`. If the model has no **extension**, this fails without
    calling ``/Content`` (no id+suffix file name; no extension string to run ``mimetypes`` on).

    ``mime_type`` is the first return value of :func:`mimetypes.guess_type` for a name ending
    in the model's suffix (e.g. ``"x" + ext``; ``None`` if the stdlib has no type for that
    suffix). CMW file-backed
    documents on the test tenant all returned a non-empty ``extension``
    (pdf, md, mp4, docx, mp3, jpg).
    """
    ext = _extension_suffix_from_model(document_model)
    if not ext:
        return {
            "success": False,
            "error": "GetDocument model has no extension; cannot set filename or MIME type.",
        }
    rel = document_content_get_path(document_id)
    g = requests_._get_url_binary(rel)
    if not g.get("success") or not g.get("content"):
        return {
            "success": False,
            "error": g.get("error", "Failed to fetch document content"),
        }
    content_bytes: bytes = g["content"]
    b64 = base64.b64encode(content_bytes).decode("ascii")
    mime_type, _ = mimetypes.guess_type(f"x{ext}")
    return {
        "success": True,
        "content": b64,
        "mime_type": mime_type,
        "filename": f"{document_id}{ext}",
    }


# b64_to_temp_file has been extracted to tools.file_utils.FileUtils.b64_to_temp_file(b64, suffix, context="document")
# This removes duplication with the image module and tool layer. See FileUtils for the unified, lean implementation.
# Callers should import from FileUtils (or keep re-export if needed for backward compat).


def set_object_document(
    record_id: str,
    property_system_name: str,
    file_name: str,
    file_bytes: bytes,
    *,
    replace: bool = True,
) -> dict[str, Any]:
    """
    POST SetObjectDocument — attach or replace a file on a document attribute.
    """
    body: dict[str, Any] = {
        "objectId": record_id,
        "propertyAlias": property_system_name,
        "fileName": file_name,
        "fileData": list(file_bytes),
        "replace": bool(replace),
    }
    return requests_._post_request(body, SET_OBJECT_DOCUMENT)


__all__ = [
    "GET_PROPERTY_VALUES",
    "SET_OBJECT_DOCUMENT",
    "_extension_suffix_from_model",
    "display_filename_for_registry",
    "document_content_get_path",
    "extract_platform_document_id",
    "fetch_record_field_values",
    "get_document_content",
    "get_document_model",
    "resolve_id_from_record_property",
    "set_object_document",
]
