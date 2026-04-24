"""Record tools for **image** attributes: load stored images, upload new ones (same pattern as document tools)."""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
import tempfile
from typing import Annotated, Any

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field, field_validator

from tools.file_reference_tool_text import (
    CHAT_FILE_REFERENCE_DESCRIPTION,
    CHAT_FILE_REFERENCE_RESULT_HINT,
)
from tools.file_utils import FileUtils
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

_FETCH_RECORD_IMAGE_FILE_DESCRIPTION = (
    "Load an **image** that is **already stored** on an **image** attribute of a **record** so it "
    "can be read or sent to a vision or file tool like a user attachment.\n\n"
    + CHAT_FILE_REFERENCE_RESULT_HINT
    + "\n\nThe result also includes **``image_id``** (platform id of the image).\n\n"
    "If the **attribute** is **multivalue** (more than one file can be stored), set "
    "**``multivalue_index``** to pick one image (0-based; list order, often newest first in the "
    "app)."
)


@tool(
    "fetch_record_image_file",
    return_direct=False,
    description=_FETCH_RECORD_IMAGE_FILE_DESCRIPTION,
)
def fetch_record_image_file(
    record_id: str,
    image_attribute_system_name: str,
    multivalue_index: int = 0,
    agent: Annotated[Any | None, InjectedToolArg] = None,
) -> dict[str, Any]:
    """Load a stored image; see the tool **description** for what **``file_reference``** is."""
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

    if agent is not None and callable(getattr(agent, "register_file", None)):
        try:
            agent.register_file(display, tpath)
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
            "file_reference": display,
            "image_id": img_id,
        }
    return {
        "success": True,
        "error": None,
        "file_reference": os.path.abspath(tpath),
        "image_id": img_id,
        "display_filename": display,
    }


class AttachImageToRecordSchema(BaseModel):
    record_id: str = Field(description="Which record to update (record id).")
    attribute_system_name: str = Field(
        description="System name of the image attribute in the record template.",
    )
    file_reference: str = Field(
        description=CHAT_FILE_REFERENCE_DESCRIPTION,
    )
    file_name: str | None = Field(
        default=None,
        description="Name in the app (e.g. photo.png). Defaults to the resolved file's base name.",
    )

    @field_validator("record_id", "attribute_system_name", "file_reference", mode="before")
    @classmethod
    def strip_req(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            msg = "must be a non-empty string"
            raise ValueError(msg)
        return v.strip()

    @field_validator("file_name", mode="before")
    @classmethod
    def opt_file_name(cls, v: Any) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str) or not v.strip():
            return None
        return v.strip()


@tool(
    "attach_file_to_record_image_attribute",
    return_direct=False,
    args_schema=AttachImageToRecordSchema,
)
def attach_file_to_record_image_attribute(
    record_id: str,
    attribute_system_name: str,
    file_reference: str,
    file_name: str | None = None,
    agent: Annotated[Any | None, InjectedToolArg] = None,
) -> dict[str, Any]:
    """
    **Upload** a new **image** onto an **image** **attribute** of a **record** using
    **``file_reference``** (see the parameter **description**).     **``file_name``** is optional; omit
    it to keep the same name as the file in **``file_reference``**. On a **single-value** **attribute**, the new
    image **replaces** the previous one. The result includes **``success``**, platform
    **``raw_response``** / **``error``**, and **``image_id``** for the new image.
    """
    raw, rerr, rpath = FileUtils.read_file_reference_bytes(file_reference, agent)
    if rerr is not None or raw is None or not rpath:
        return {
            "success": False,
            "status_code": 400,
            "error": rerr or "Failed to read file for upload",
            "raw_response": None,
            "image_id": None,
        }
    display_name = file_name or os.path.basename(rpath)
    if not display_name:
        return {
            "success": False,
            "status_code": 400,
            "error": "Could not determine a file name for the upload",
            "raw_response": None,
            "image_id": None,
        }
    cres = create_image_file(display_name, raw)
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
