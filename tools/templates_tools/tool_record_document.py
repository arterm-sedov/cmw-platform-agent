"""Record tools for document attributes: fetch stored files, upload attachments. For generic field reads, see tool_get_record_values."""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
import tempfile
from typing import Annotated, Any

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tools.file_reference_tool_text import (
    CHAT_FILE_REFERENCE_DESCRIPTION,
    CHAT_FILE_REFERENCE_RESULT_HINT,
)
from tools.file_utils import FileUtils
from tools.platform_record_document import (
    display_filename_for_registry,
    get_document_content,
    get_document_model,
    resolve_id_from_record_property,
    set_object_document,
)

logger = logging.getLogger(__name__)

# LangChain ``description=`` for :func:`fetch_record_document_file` (not f-strings: must be a real
# string at decoration time; shares only the result hint with the image tool).
_FETCH_RECORD_DOCUMENT_FILE_DESCRIPTION = (
    "Load a file that is **already stored** on a **document** attribute of a **record** so it "
    "can be read or passed along like a user attachment.\n\n"
    + CHAT_FILE_REFERENCE_RESULT_HINT
    + "\n\nThe result also includes **``document_id``** (platform id of the file).\n\n"
    "If the **attribute** is **multivalue** (more than one file can be stored), set "
    "**``multivalue_index``** to pick one file to load (0-based; list order, often newest first "
    "in the app)."
)


@tool(
    "fetch_record_document_file",
    return_direct=False,
    description=_FETCH_RECORD_DOCUMENT_FILE_DESCRIPTION,
)
def fetch_record_document_file(
    record_id: str,
    document_attribute_system_name: str,
    multivalue_index: int = 0,
    agent: Annotated[Any | None, InjectedToolArg] = None,
) -> dict[str, Any]:
    """Load a stored document file; see the tool description for file_reference."""
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
    err, doc_id = resolve_id_from_record_property(
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
    # Temp file suffix matches the file name (including extension) from the model.
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

    # register_file uses the product file name (``display``), not the id+suffix from
    # :func:`get_document_content` (``dres["filename"]``).
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
                "document_id": doc_id,
            }
        return {
            "success": True,
            "error": None,
            "file_reference": display,
            "document_id": doc_id,
        }
    # No agent (tests/headless): path resolution only; normal chat agent always has agent.
    return {
        "success": True,
        "error": None,
        "file_reference": os.path.abspath(tpath),
        "document_id": doc_id,
        "display_filename": display,
    }


class AttachFileToRecordDocumentSchema(BaseModel):
    # ``agent`` must be a model field (InjectedToolArg) so LangChain _parse_input keeps it;
    # plain @tool tools infer args and do not need this. arbitrary_types_allowed: live agent object.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    record_id: str = Field(description="Record id to attach the file to.")
    attribute_system_name: str = Field(
        description="System name of the document attribute in the record template.",
    )
    file_reference: str = Field(
        description=CHAT_FILE_REFERENCE_DESCRIPTION,
    )
    replace: bool = Field(
        default=True,
        description=(
            "true = replace (default), false = add. false only changes behavior for multivalue "
            "document attributes; single-file attributes still end up with one file."
        ),
    )
    agent: Annotated[Any | None, InjectedToolArg] = Field(
        default=None,
        description="Session agent for chat file registry; injected by the app, not the LLM.",
    )

    @field_validator("record_id", "attribute_system_name", "file_reference", mode="before")
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
    file_reference: str,
    replace: bool = True,
    agent: Annotated[Any | None, InjectedToolArg] = None,
) -> dict[str, Any]:
    """
    Upload a file to a document attribute (file_reference). The stored filename is the path or URL
    basename. Returns success, status_code, error, raw_response.
    """
    raw, rerr = FileUtils.read_file_reference_bytes(file_reference, agent)
    if rerr is not None or raw is None:
        return {
            "success": False,
            "status_code": 400,
            "error": rerr or "Failed to read file for upload",
            "raw_response": None,
        }
    display_name = FileUtils.upload_basename_from_reference(file_reference)
    if not display_name:
        return {
            "success": False,
            "status_code": 400,
            "error": "Could not determine a file name for the upload",
            "raw_response": None,
        }
    res = set_object_document(
        record_id,
        attribute_system_name,
        display_name,
        raw,
        replace=replace,
    )
    return {
        "success": res.get("success", False),
        "status_code": res.get("status_code"),
        "error": res.get("error"),
        "raw_response": res.get("raw_response"),
    }
