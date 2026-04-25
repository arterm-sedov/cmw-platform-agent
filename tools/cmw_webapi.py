"""CMW/WebAPI common helpers for response parsing and ID extraction.

Central place for patterns common to Document, Image, Record, and other webapi/* endpoints.
Avoids duplication and the awkward cross-import from platform_record_image to platform_record_document.

See cmw_open_api/web_api_v1.json for the standard {"response": X} wrapper and Create response shapes.

**Key patterns:**
- Many endpoints wrap payload in {"response": data}.
- Attribute values for media/reference fields may be plain str id or {"id": "..." } dict.
- Create endpoints return the new id as string or wrapped.

All functions are pure, defensive, with safe defaults and comprehensive docstrings per AGENTS.md.
No unnecessary try/except; errors are explicit.

Used by platform_record_* modules, tool_record_* , tool_create_edit_record, and tests.
"""

from __future__ import annotations

from typing import Any


def unwrap_webapi_payload(raw: Any) -> Any:
    """Unwrap ``{"response": X}`` from WebApi-style JSON.

    This is the standard shape for most CMW webapi responses (see OpenAPI spec).
    If no "response" key, return as-is. Used by get_model, fetch_record_field_values,
    extract_created_id, etc.
    """
    if isinstance(raw, dict) and "response" in raw:
        return raw["response"]
    return raw


def extract_platform_document_id(value: Any) -> str | None:
    """
    Return the platform id from an attribute value (used for documents, images, references).

    Accepts a plain id string, or a reference dict containing ``id`` (common CMW style).
    Generalized for use by both document and image record tools (despite the name).
    Strips whitespace; returns None for empty/missing.

    See resolve_id_from_record_property for multivalue and case-insensitive lookup.
    """
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        raw = value.get("id")
        if raw is None or raw == "":
            return None
        return str(raw).strip() or None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    return str(value).strip() or None


def extract_created_id(result: dict[str, Any]) -> str | None:
    """New id from a ``webapi/.../Create``-style JSON response (string, {"response": str}, etc.).

    Handles the variants seen in Image/Create and similar endpoints.
    Returns None on failure or no id. Used by attach_file_to_record_image_attribute.
    """
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


__all__ = [
    "extract_created_id",
    "extract_platform_document_id",
    "unwrap_webapi_payload",
]
