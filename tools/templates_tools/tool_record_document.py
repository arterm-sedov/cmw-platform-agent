"""Template/record document tools: read fields, load files for analysis, upload attachments."""

from __future__ import annotations

import base64
import binascii
import logging
import os
from pathlib import Path
import tempfile
from typing import Annotated, Any

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field, field_validator

from tools.platform_record_document import (
    display_filename_for_registry,
    extract_platform_document_id,
    fetch_record_field_values,
    get_document_content,
    get_document_model,
    set_object_document,
)

logger = logging.getLogger(__name__)


class GetRecordFieldValuesSchema(BaseModel):
    record_id: str = Field(description="Record id (UUID) to read.")
    attribute_system_names: list[str] = Field(
        min_length=1,
        description="List of attribute system names to return.",
    )

    @field_validator("record_id", mode="before")
    @classmethod
    def strip_rid(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            msg = "record_id must be a non-empty string"
            raise ValueError(msg)
        return v.strip()


@tool(
    "get_record_field_values",
    return_direct=False,
    args_schema=GetRecordFieldValuesSchema,
)
def get_record_field_values(
    record_id: str, attribute_system_names: list[str]
) -> dict[str, Any]:
    """
    Read selected attribute values for a record (by system name).

    Use to see what is stored on the record, including document attribute ids, before
    loading or attaching files.
    """
    return fetch_record_field_values(record_id, attribute_system_names)


def _document_id_for_attribute(
    record_id: str,
    document_attribute_system_name: str,
    multivalue_index: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Read one document id from GetPropertyValues for ``document_attribute_system_name``.

    **Production use:** :func:`fetch_record_document_file` only. The API row may key the
    property by a different string casing than the template system name, so a case-insensitive
    lookup is used. Multivalue document attributes are lists; ``multivalue_index`` picks the
    element. Values are a plain id string or ``{"id": "..."}``; :func:`extract_platform_document_id`
    matches both.
    On failure, return (error_dict, None). On success, return (None, document_id).
    """
    g = fetch_record_field_values(record_id, [document_attribute_system_name])
    if not g.get("success"):
        return (
            {
                "success": False,
                "error": g.get("error") or "Failed to read record",
            },
            None,
        )
    data = (g.get("data") or {}).get(record_id) or {}
    raw_val = data.get(document_attribute_system_name)
    if raw_val is None and isinstance(data, dict):
        for k, v in data.items():
            if k.lower() == document_attribute_system_name.lower():
                raw_val = v
                break
    if raw_val in (None, ""):
        return (
            {
                "success": False,
                "error": "Document attribute is empty or missing on this record.",
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
    doc_id = extract_platform_document_id(raw_val)
    if not doc_id:
        return (
            {
                "success": False,
                "error": "Could not resolve a document id from the attribute value.",
            },
            None,
        )
    return (None, doc_id)


@tool("fetch_record_document_file", return_direct=False)
def fetch_record_document_file(
    record_id: str,
    document_attribute_system_name: str,
    multivalue_index: int = 0,
    agent: Annotated[Any | None, InjectedToolArg] = None,
) -> dict[str, Any]:
    """
    Load a file from a record's document attribute into the session like a chat attachment.
    On success, `file_reference` is the logical file name to pass into file tools (same as an
    upload; from GetDocument: API field ``title`` is the stored filename, plus ``extension``).
    ``display_filename`` in the result is that full name. Multiple files: use ``multivalue_index``
    (0-based; platform order, often newest first).
    """
    record_id = record_id.strip() if isinstance(record_id, str) else ""
    document_attribute_system_name = (
        document_attribute_system_name.strip()
        if isinstance(document_attribute_system_name, str)
        else ""
    )
    if not record_id or not document_attribute_system_name:
        return {
            "success": False,
            "error": "record_id and document_attribute_system_name must be non-empty.",
            "file_reference": None,
            "document_id": None,
        }
    err, doc_id = _document_id_for_attribute(
        record_id, document_attribute_system_name, multivalue_index
    )
    if err is not None:
        return {**err, "file_reference": None, "document_id": None}
    gmod = get_document_model(doc_id)
    if not gmod.get("success"):
        return {
            "success": False,
            "error": gmod.get("error", "GetDocument (metadata) failed"),
            "file_reference": None,
            "document_id": doc_id,
        }
    model = gmod.get("model")
    if not isinstance(model, dict):
        return {
            "success": False,
            "error": "GetDocument returned no model object.",
            "file_reference": None,
            "document_id": doc_id,
        }
    display = display_filename_for_registry(model)
    if not display:
        return {
            "success": False,
            "error": (
                "Could not determine the file name for this document on the platform."
            ),
            "file_reference": None,
            "document_id": doc_id,
        }
    dres = get_document_content(doc_id, document_model=model)
    if not dres.get("success"):
        return {
            "success": False,
            "error": dres.get("error", "Document download failed"),
            "file_reference": None,
            "document_id": doc_id,
        }
    b64c = dres.get("content")
    if not isinstance(b64c, str):
        return {
            "success": False,
            "error": "Document body was not base64 text.",
            "file_reference": None,
            "document_id": doc_id,
        }
    # :func:`get_document_content` encodes raw ``/Content`` bytes; CMW test runs used non-empty files.
    raw = base64.b64decode(b64c, validate=False)
    # Local temp name uses the same suffix as the logical filename (``title``+``extension`` on the model).
    tmp_suffix = Path(display).suffix
    try:
        fd, tpath = tempfile.mkstemp(suffix=tmp_suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
    except OSError as e:
        return {
            "success": False,
            "error": str(e),
            "file_reference": None,
            "document_id": doc_id,
        }

    content_transport_filename = dres.get("filename")
    logical = display
    # With agent: LLM only sees the session file name (like chat); paths stay in registry.
    if agent is not None and callable(getattr(agent, "register_file", None)):
        try:
            agent.register_file(logical, tpath)
        except Exception as e:
            logger.warning("register_file failed: %s", e)
            try:
                os.unlink(tpath)
            except OSError as oe:
                logger.debug("temp cleanup after register failure: %s", oe)
            return {
                "success": False,
                "error": f"register_file failed: {e!s}",
                "file_reference": None,
                "document_id": doc_id,
            }
        return {
            "success": True,
            "error": None,
            "file_reference": logical,
            "document_id": doc_id,
            "display_filename": display,
            "content_fileName": content_transport_filename,
            "message": "Use file_reference as the attachment name for follow-up file tools.",
        }
    # No agent (tests/headless): path resolution only; normal chat agent always has agent.
    return {
        "success": True,
        "error": None,
        "file_reference": os.path.abspath(tpath),
        "document_id": doc_id,
        "display_filename": display,
        "content_fileName": content_transport_filename,
        "message": "Headless: file_reference is a local path for tooling without a session.",
    }


class AttachFileToRecordDocumentSchema(BaseModel):
    record_id: str = Field(description="Record id to attach the file to.")
    attribute_system_name: str = Field(
        description="System name of the document attribute.",
    )
    file_name: str = Field(description="Original file name (e.g. report.pdf).")
    file_base64: str = Field(
        description="File contents as base64 (not a data: URL; raw base64).",
    )
    replace: bool = Field(
        default=True,
        description="If true, replace an existing file on a single-value attribute.",
    )

    @field_validator("record_id", "attribute_system_name", "file_name", mode="before")
    @classmethod
    def strip_req(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            msg = "must be a non-empty string"
            raise ValueError(msg)
        return v.strip()


@tool(
    "attach_file_to_record_document_attribute",
    return_direct=False,
    args_schema=AttachFileToRecordDocumentSchema,
)
def attach_file_to_record_document_attribute(
    record_id: str,
    attribute_system_name: str,
    file_name: str,
    file_base64: str,
    replace: bool = True,
) -> dict[str, Any]:
    """
    Upload or replace a file on a record's document attribute. Supply the file name
    and content as base64. Use this when the user or workflow needs a new file stored
    on the record.
    """
    try:
        raw = base64.b64decode(file_base64, validate=False)
    except (ValueError, binascii.Error) as e:
        return {
            "success": False,
            "status_code": 400,
            "error": f"Invalid base64: {e!s}",
            "raw_response": None,
        }
    res = set_object_document(
        record_id,
        attribute_system_name,
        file_name,
        raw,
        replace=replace,
    )
    return {
        "success": res.get("success", False),
        "status_code": res.get("status_code"),
        "error": res.get("error"),
        "raw_response": res.get("raw_response"),
    }
