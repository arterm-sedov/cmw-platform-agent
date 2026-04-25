"""
Proves the **agent** is forwarded into **attach** / **read_file_reference_bytes** when the
**args_schema** includes **``agent``** ( :class:`InjectedToolArg` on the Pydantic model ).

**Regression:** If **agent** is only on the @tool function but **omitted** from
``AttachFileToRecordDocumentSchema`` / ``AttachImageToRecordSchema``,
:meth:`langchain_core.tools.base.BaseTool._parse_input` **drops** the injected
``"agent"`` from ``invoke(…, agent=… )``, so the chat **file_reference** never
resolves for upload while **read_text_based_file** still works.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest

from tools.templates_tools.tool_record_document import (
    attach_file_to_record_document_attribute,
)
from tools.templates_tools.tool_record_image import (
    attach_file_to_record_image_attribute,
)
from tools.tools import read_text_based_file


def _make_registry_agent():
    class _A:
        session_id = "sess-attach-1"

        def __init__(self) -> None:
            self.file_registry: dict[tuple[str, str], str] = {}

        def get_file_path(self, original_filename: str) -> str | None:
            k = (self.session_id, original_filename)
            p = self.file_registry.get(k)
            return p if p and os.path.isfile(p) else None

        def register_file(self, name: str, pth: str) -> None:
            self.file_registry[(self.session_id, name)] = pth

    return _A()


def test_attach_file_to_record_document_receives_injected_agent_for_basename() -> None:
    ag = _make_registry_agent()
    d = tempfile.mkdtemp()
    p = os.path.join(d, "upl.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("upload body")
    ag.register_file("upl.txt", p)
    with patch(
        "tools.templates_tools.tool_record_document.set_object_document",
        return_value={"success": True, "status_code": 200, "error": None, "raw_response": {}},
    ) as sdoc:
        r = attach_file_to_record_document_attribute.invoke(
            {
                "record_id": "r1",
                "attribute_system_name": "Document",
                "file_reference": "upl.txt",
                "agent": ag,
            }
        )
    assert r.get("success") is True, r
    sdoc.assert_called()
    b = sdoc.call_args[0][3]
    assert b == b"upload body", b


def test_platform_upload_uses_chat_name_not_gradio_temp_basename() -> None:
    """**fileName** follows the logical **file_reference**, not the on-disk **gradio_…** name."""
    ag = _make_registry_agent()
    d = tempfile.mkdtemp()
    gradio_like = os.path.join(
        d, "gradio_crm4r5er04_SGR_1777075419411_93d327c9.docx"
    )
    with open(gradio_like, "wb") as f:
        f.write(b"docx")
    ag.register_file("SGR工具链验证.docx", gradio_like)
    with patch(
        "tools.templates_tools.tool_record_document.set_object_document",
        return_value={"success": True, "status_code": 200, "error": None, "raw_response": {}},
    ) as sdoc:
        r = attach_file_to_record_document_attribute.invoke(
            {
                "record_id": "r1",
                "attribute_system_name": "Document",
                "file_reference": "SGR工具链验证.docx",
                "agent": ag,
            }
        )
    assert r.get("success") is True, r
    sdoc.assert_called()
    _rid, _attr, display, raw = sdoc.call_args[0]
    assert display == "SGR工具链验证.docx", display
    assert raw == b"docx"


def test_read_text_and_attach_parity_on_same_reference() -> None:
    ag = _make_registry_agent()
    d = tempfile.mkdtemp()
    p = os.path.join(d, "same_name.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("roundtrip line")
    ag.register_file("same_name.txt", p)
    t = read_text_based_file.invoke(
        {"file_reference": "same_name.txt", "read_html_as_markdown": True, "agent": ag}
    )
    assert isinstance(t, str)
    assert "roundtrip line" in t, t
    with patch(
        "tools.templates_tools.tool_record_document.set_object_document",
        return_value={"success": True, "status_code": 200, "error": None, "raw_response": {}},
    ):
        r = attach_file_to_record_document_attribute.invoke(
            {
                "record_id": "r1",
                "attribute_system_name": "Document",
                "file_reference": "same_name.txt",
                "agent": ag,
            }
        )
    assert r.get("success") is True, r


def test_attach_file_to_record_image_receives_injected_agent() -> None:
    ag = _make_registry_agent()
    p = os.path.join(tempfile.gettempdir(), f"im_{os.getpid()}.png")
    with open(p, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
    ag.register_file("z.png", p)
    with (
        patch(
            "tools.templates_tools.tool_record_image.create_image_file",
            return_value={"success": True, "raw_response": {"response": "newid"}},
        ) as cif,
        patch(
            "tools.templates_tools.tool_record_image.put_record_image_attribute_value",
            return_value={"success": True, "status_code": 200, "error": None},
        ),
    ):
        r = attach_file_to_record_image_attribute.invoke(
            {
                "record_id": "r1",
                "attribute_system_name": "Img",
                "file_reference": "z.png",
                "agent": ag,
            }
        )
    assert r.get("success") is True, r
    cif.assert_called()


def test_image_platform_upload_uses_chat_name_not_gradio_temp_basename() -> None:
    ag = _make_registry_agent()
    d = tempfile.mkdtemp()
    gradio_like = os.path.join(
        d, f"gradio_x_img_{os.getpid()}_abc12345.png"
    )
    with open(gradio_like, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
    ag.register_file("z.png", gradio_like)
    with (
        patch(
            "tools.templates_tools.tool_record_image.create_image_file",
            return_value={"success": True, "raw_response": {"response": "newid"}},
        ) as cif,
        patch(
            "tools.templates_tools.tool_record_image.put_record_image_attribute_value",
            return_value={"success": True, "status_code": 200, "error": None},
        ),
    ):
        r = attach_file_to_record_image_attribute.invoke(
            {
                "record_id": "r1",
                "attribute_system_name": "Img",
                "file_reference": "z.png",
                "agent": ag,
            }
        )
    assert r.get("success") is True, r
    cif.assert_called()
    name_arg, _raw = cif.call_args[0]
    assert name_arg == "z.png", name_arg


# --- Optional: local one-off path (e.g. user's PDF); skipped in CI. -----------------

_USER_PDF = r"c:\OneDrive\Documents\Якшин cv Ai Architect.pdf"


@pytest.mark.skipif(
    not os.path.isfile(_USER_PDF),
    reason="optional local file for manual harness-style check",
)
def test_attach_with_optional_user_pdf_path_if_present() -> None:
    ag = _make_registry_agent()
    p = _USER_PDF
    name = os.path.basename(p)
    ag.register_file(name, p)
    with patch(
        "tools.templates_tools.tool_record_document.set_object_document",
        return_value={"success": True, "status_code": 200, "error": None, "raw_response": {}},
    ) as sdoc:
        r = attach_file_to_record_document_attribute.invoke(
            {
                "record_id": "r-local-harness",
                "attribute_system_name": "Document",
                "file_reference": name,
                "agent": ag,
            }
        )
    assert r.get("success") is True, r
    sdoc.assert_called()
