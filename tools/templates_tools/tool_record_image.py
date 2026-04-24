"""Template/record **image** tools: read image id from a property, load bytes, upload new image."""

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

from tools.platform_record_document import resolve_id_from_record_property
from tools.platform_record_image import (
    create_image_file,
    display_filename_for_image_model,
    extract_created_id,
    get_image_file_payload,
    get_image_model,
    put_record_image_attribute_value,
)

logger = logging.getLogger(__name__)


@tool("fetch_record_image_file", return_direct=False)
def fetch_record_image_file(
    record_id: str,
    image_attribute_system_name: str,
    multivalue_index: int = 0,
    agent: Annotated[Any | None, InjectedToolArg] = None,
) -> dict[str, Any]:
    """
    Load an image from a record's **image** attribute into the session like a chat attachment.
    Values are image ids; bytes come from ``GET /webapi/Image`` (not raw ``/Content`` like documents).
    ``display_filename`` uses the API ``name`` or ``image.{format}``. Multivalue: ``multivalue_index``.
    """
    record_id = record_id.strip() if isinstance(record_id, str) else ""
    image_attribute_system_name = (
        image_attribute_system_name.strip()
        if isinstance(image_attribute_system_name, str)
        else ""
    )
    if not record_id or not image_attribute_system_name:
        return {
            "success": False,
            "error": "record_id and image_attribute_system_name must be non-empty.",
            "file_reference": None,
            "image_id": None,
        }
    err, img_id = resolve_id_from_record_property(
        record_id, image_attribute_system_name, multivalue_index
    )
    if err is not None:
        return {**err, "file_reference": None, "image_id": None}
    gimg = get_image_model(img_id)
    if not gimg.get("success"):
        return {
            "success": False,
            "error": gimg.get("error", "Get Image (metadata) failed"),
            "file_reference": None,
            "image_id": img_id,
        }
    model = gimg.get("model")
    if not isinstance(model, dict):
        return {
            "success": False,
            "error": "Get Image returned no model object.",
            "file_reference": None,
            "image_id": img_id,
        }
    display = display_filename_for_image_model(model)
    if not display:
        return {
            "success": False,
            "error": "Could not determine a file name from the image metadata.",
            "file_reference": None,
            "image_id": img_id,
        }
    pres = get_image_file_payload(img_id, image_model=model)
    if not pres.get("success"):
        return {
            "success": False,
            "error": pres.get("error", "Image payload failed"),
            "file_reference": None,
            "image_id": img_id,
        }
    b64c = pres.get("content")
    if not isinstance(b64c, str):
        return {
            "success": False,
            "error": "Image body was not base64 text.",
            "file_reference": None,
            "image_id": img_id,
        }
    raw = base64.b64decode(b64c, validate=False)
    tmp_suffix = Path(display).suffix or ".png"
    try:
        fd, tpath = tempfile.mkstemp(suffix=tmp_suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
    except OSError as e:
        return {
            "success": False,
            "error": str(e),
            "file_reference": None,
            "image_id": img_id,
        }

    content_transport_filename = pres.get("filename")
    logical = display
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
                "error": f"register_file failed: {str(e)}",
                "file_reference": None,
                "image_id": img_id,
            }
        return {
            "success": True,
            "error": None,
            "file_reference": logical,
            "image_id": img_id,
            "display_filename": display,
            "content_fileName": content_transport_filename,
            "message": "Use file_reference as the attachment name for follow-up file tools.",
        }
    return {
        "success": True,
        "error": None,
        "file_reference": os.path.abspath(tpath),
        "image_id": img_id,
        "display_filename": display,
        "content_fileName": content_transport_filename,
        "message": "Headless: file_reference is a local path for tooling without a session.",
    }


class AttachImageToRecordSchema(BaseModel):
    record_id: str = Field(description="Record id to attach the image to.")
    attribute_system_name: str = Field(
        description="System name of the image attribute.",
    )
    file_name: str = Field(description="Original file name (e.g. photo.png).")
    file_base64: str = Field(
        description="File contents as base64 (not a data: URL; raw base64).",
    )

    @field_validator("record_id", "attribute_system_name", "file_name", mode="before")
    @classmethod
    def strip_req(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            msg = "must be a non-empty string"
            raise ValueError(msg)
        return v.strip()


@tool(
    "attach_file_to_record_image_attribute",
    return_direct=False,
    args_schema=AttachImageToRecordSchema,
)
def attach_file_to_record_image_attribute(
    record_id: str,
    attribute_system_name: str,
    file_name: str,
    file_base64: str,
) -> dict[str, Any]:
    """
    Upload an image file to a record's **image** attribute: ``POST /webapi/Image/Create``,
    then ``PUT /webapi/Record/{id}`` with the new image id (no TeamNetwork ``SetObject…`` for image).
    """
    try:
        raw = base64.b64decode(file_base64, validate=False)
    except (ValueError, binascii.Error) as e:
        return {
            "success": False,
            "status_code": 400,
            "error": f"Invalid base64: {str(e)}",
            "raw_response": None,
            "image_id": None,
        }
    cres = create_image_file(file_name, raw)
    new_id = extract_created_id(cres)
    if not new_id:
        return {
            "success": False,
            "status_code": cres.get("status_code", 500),
            "error": cres.get("error") or "Image Create did not return an id",
            "raw_response": cres.get("raw_response"),
            "image_id": None,
        }
    pres = put_record_image_attribute_value(record_id, attribute_system_name, new_id)
    return {
        "success": pres.get("success", False),
        "status_code": pres.get("status_code"),
        "error": pres.get("error"),
        "raw_response": pres.get("raw_response"),
        "image_id": new_id,
    }


__all__ = [
    "attach_file_to_record_image_attribute",
    "fetch_record_image_file",
]
