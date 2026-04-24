"""Tests: platform id extraction, record fields, document download, record coercion, local text."""

from __future__ import annotations

import base64
import contextlib
import os
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

from tools import requests_
from tools.local_path_text import read_local_path_to_plain_text
from tools.platform_record_document import (
    SET_OBJECT_DOCUMENT,
    display_filename_for_registry,
    document_content_get_path,
    extract_platform_document_id,
    fetch_record_field_values,
    get_document_content,
    get_document_model,
    resolve_id_from_record_property,
    set_object_document,
    unwrap_webapi_payload,
)
from tools.platform_record_image import (
    WEBAPI_IMAGE_CREATE,
    create_image_file,
    display_filename_for_image_model,
    extract_created_id,
    get_image_file_payload,
    image_get_path,
)
from tools.templates_tools.tool_create_edit_record import _coerce_scalar_value
from tools.templates_tools.tool_record_document import fetch_record_document_file
from tools.templates_tools.tool_record_image import fetch_record_image_file
from tools.tools import read_text_based_file


class TestExtractPlatformDocumentId:
    def test_none_and_empty(self) -> None:
        assert extract_platform_document_id(None) is None
        assert extract_platform_document_id("") is None

    def test_dict_with_id(self) -> None:
        assert extract_platform_document_id({"id": "doc-1"}) == "doc-1"

    def test_dict_missing_id(self) -> None:
        assert extract_platform_document_id({"x": 1}) is None

    def test_string(self) -> None:
        assert extract_platform_document_id("  ab-cd  ") == "ab-cd"

    def test_int_coerced(self) -> None:
        assert extract_platform_document_id(99) == "99"


class TestUnwrap:
    def test_unwraps_response(self) -> None:
        assert unwrap_webapi_payload({"response": {"a": 1}}) == {"a": 1}

    def test_no_wrapper(self) -> None:
        assert unwrap_webapi_payload({"b": 2}) == {"b": 2}


class TestGetRecordFieldValues:
    def test_success_shape(self) -> None:
        rid = "rec-1"
        raw = {
            "success": True,
            "status_code": 200,
            "raw_response": {"response": {rid: {"FileRef": {"id": "d1"}}}},
        }
        with patch("tools.requests_._post_request", return_value=raw):
            out = fetch_record_field_values(rid, ["FileRef"])
        assert out["success"] is True
        assert out["data"] == {rid: {"FileRef": {"id": "d1"}}}

    def test_request_failed(self) -> None:
        with patch(
            "tools.requests_._post_request",
            return_value={"success": False, "status_code": 500, "error": "e"},
        ):
            out = fetch_record_field_values("r", ["x"])
        assert out["success"] is False
        assert "error" in out


class TestDisplayFilenameForRegistry:
    def test_title_and_ext_joined(self) -> None:
        out = display_filename_for_registry(
            {"title": "My File", "extension": "pdf"},
        )
        assert out == "My File.pdf"

    def test_title_already_has_ext(self) -> None:
        assert (
            display_filename_for_registry(
                {"title": "A.pdf", "extension": "pdf"},
            )
            == "A.pdf"
        )

    def test_extension_with_leading_dot(self) -> None:
        assert (
            display_filename_for_registry(
                {"title": "B", "extension": ".docx"},
            )
            == "B.docx"
        )
        assert (
            display_filename_for_registry(
                {"title": "B.docx", "extension": "docx"},
            )
            == "B.docx"
        )

    def test_case_insensitive_when_title_endswith_ext(self) -> None:
        assert (
            display_filename_for_registry(
                {"title": "C.PDF", "extension": "pdf"},
            )
            == "C.PDF"
        )

    def test_missing_stuff(self) -> None:
        assert display_filename_for_registry({}) is None
        assert display_filename_for_registry({"title": "x"}) is None
        assert display_filename_for_registry({"extension": "pdf"}) is None


class TestGetDocumentModel:
    def test_unwraps_response(self) -> None:
        http = {
            "success": True,
            "raw_response": {
                "response": {
                    "id": "d1",
                    "title": "R",
                    "extension": "txt",
                }
            },
        }
        with patch(
            "tools.platform_record_document.requests_._get_request", return_value=http
        ):
            out = get_document_model("d1")
        assert out["success"] is True
        m = out["model"]
        assert m is not None
        assert m["title"] == "R"
        assert m["extension"] == "txt"


class TestGetDocumentContent:
    def test_raw_body_uses_model_extension_and_mime(self) -> None:
        """CMW test tenant: Content is raw file bytes; names/types come from GetDocument model."""
        payload = b"%PDF-1.4 test"
        b64 = base64.b64encode(payload).decode("ascii")
        model = {"title": "R.pdf", "extension": "pdf"}
        with patch("tools.requests_._get_url_binary") as u:
            u.return_value = {"success": True, "content": payload}
            out = get_document_content("d2", document_model=model)
        u.assert_called_once()
        assert out["success"] is True
        assert out["content"] == b64
        assert out["filename"] == "d2.pdf"
        assert out["mime_type"] == "application/pdf"

    def test_fails_if_model_has_no_extension_before_http(self) -> None:
        with patch("tools.requests_._get_url_binary") as u:
            out = get_document_content("d2", document_model={"title": "OnlyTitle"})
        u.assert_not_called()
        assert out.get("success") is False
        assert "extension" in (out.get("error") or "").lower()

    def test_path_helper(self) -> None:
        assert "webapi/Document" in document_content_get_path("x")
        assert "Content" in document_content_get_path("x")


class TestResolveIdFromRecordProperty:
    """Used by :func:`fetch_record_document_file` and :func:`fetch_record_image_file`."""

    def test_resolves_property_value_by_exact_key(self) -> None:
        rid = "rec-a"
        with patch(
            "tools.platform_record_document.fetch_record_field_values",
            return_value={"success": True, "data": {rid: {"FileAttr": "doc-1"}}},
        ):
            err, did = resolve_id_from_record_property(rid, "FileAttr", 0)
        assert err is None
        assert did == "doc-1"

    def test_resolves_when_api_key_differs_in_case(self) -> None:
        """GetPropertyValues may return keys that do not match the requested name byte-for-byte."""
        rid = "rec-b"
        with patch(
            "tools.platform_record_document.fetch_record_field_values",
            return_value={"success": True, "data": {rid: {"MyDocAttr": {"id": "d-x"}}}},
        ):
            err, did = resolve_id_from_record_property(rid, "mydocattr", 0)
        assert err is None
        assert did == "d-x"

    def test_multivalue_list_index(self) -> None:
        rid = "rec-c"
        with patch(
            "tools.platform_record_document.fetch_record_field_values",
            return_value={
                "success": True,
                "data": {rid: {"F": ["a", "b"]}},
            },
        ):
            err, did = resolve_id_from_record_property(rid, "F", 1)
        assert err is None
        assert did == "b"

    def test_multivalue_index_out_of_range(self) -> None:
        rid = "rec-d"
        with patch(
            "tools.platform_record_document.fetch_record_field_values",
            return_value={"success": True, "data": {rid: {"F": ["only"]}}},
        ):
            err, _ = resolve_id_from_record_property(rid, "F", 1)
        assert err is not None
        assert err.get("success") is False
        assert "multivalue" in (err.get("error") or "").lower()

    def test_get_property_values_failure(self) -> None:
        with patch(
            "tools.platform_record_document.fetch_record_field_values",
            return_value={"success": False, "error": "nope"},
        ):
            err, did = resolve_id_from_record_property("r", "F", 0)
        assert err is not None
        assert did is None


class TestSetObjectDocument:
    def test_post_body_shape(self) -> None:
        with patch("tools.requests_._post_request") as p:
            p.return_value = {"success": True, "raw_response": {}}
            set_object_document("r1", "DocAttr", "f.txt", b"ab", replace=True)
        p.assert_called_once()
        _body, path = p.call_args[0]
        assert path == SET_OBJECT_DOCUMENT
        assert _body["objectId"] == "r1"
        assert _body["propertyAlias"] == "DocAttr"
        assert _body["fileName"] == "f.txt"
        assert _body["fileData"] == [ord("a"), ord("b")]
        assert _body["replace"] is True


class TestCoerceDocument:
    def test_coerce_dict_to_id(self) -> None:
        assert _coerce_scalar_value("Document", {"id": "z9"}) == "z9"

    def test_coerce_string(self) -> None:
        assert _coerce_scalar_value("Document", "  id1  ") == "id1"


class TestCoerceImage:
    def test_coerce_image_dict_to_id(self) -> None:
        assert _coerce_scalar_value("image", {"id": "i9"}) == "i9"


class TestReadLocalPathPlainText:
    def test_txt_file(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("hello world", encoding="utf-8")
        text, err, _enc = read_local_path_to_plain_text(str(f))
        assert err is None
        assert "hello" in text
        assert text is not None


class TestFetchRecordDocumentFile:
    def test_no_agent_returns_absolute_path_with_bytes(self) -> None:
        rid = "rec-1"
        body = b"register tool test"
        b64s = base64.b64encode(body).decode("ascii")
        with (
            patch(
                "tools.templates_tools.tool_record_document.resolve_id_from_record_property",
                return_value=(None, "d-7"),
            ),
            patch(
                "tools.templates_tools.tool_record_document.get_document_model",
                return_value={
                    "success": True,
                    "model": {
                        "id": "d-7",
                        "title": "Note",
                        "extension": "txt",
                    },
                },
            ),
            patch(
                "tools.templates_tools.tool_record_document.get_document_content",
                return_value={
                    "success": True,
                    "content": b64s,
                    "mime_type": "text/plain",
                    "filename": "note.txt",
                },
            ),
        ):
            out = fetch_record_document_file.invoke(
                {
                    "record_id": rid,
                    "document_attribute_system_name": "DocAttr",
                    "multivalue_index": 0,
                    "agent": None,
                }
            )
        assert out.get("success") is True
        ref = out.get("file_reference")
        assert isinstance(ref, str)
        assert os.path.isabs(ref)
        try:
            with open(ref, "rb") as f:
                assert f.read() == body
        finally:
            if isinstance(ref, str) and os.path.isfile(ref):
                os.unlink(ref)

    def test_with_in_memory_agent_resolves_read_text(self) -> None:
        class _Agent:
            session_id = "sess-1"

            def __init__(self) -> None:
                self.file_registry: dict[tuple[str, str], str] = {}

            def register_file(self, original_filename: str, file_path: str) -> None:
                self.file_registry[(self.session_id, original_filename)] = file_path

            def get_file_path(self, original_filename: str) -> str | None:
                return self.file_registry.get((self.session_id, original_filename))

        rid = "rec-2"
        body = b"registry line\n"
        b64s = base64.b64encode(body).decode("ascii")
        ag = _Agent()
        with (
            patch(
                "tools.templates_tools.tool_record_document.resolve_id_from_record_property",
                return_value=(None, "d-99"),
            ),
            patch(
                "tools.templates_tools.tool_record_document.get_document_model",
                return_value={
                    "success": True,
                    "model": {
                        "id": "d-99",
                        "title": "RegistryLine",
                        "extension": "txt",
                    },
                },
            ),
            patch(
                "tools.templates_tools.tool_record_document.get_document_content",
                return_value={
                    "success": True,
                    "content": b64s,
                    "mime_type": "text/plain",
                    "filename": "x.txt",
                },
            ),
        ):
            out = fetch_record_document_file.invoke(
                {
                    "record_id": rid,
                    "document_attribute_system_name": "Doc",
                    "multivalue_index": 0,
                    "agent": ag,
                }
            )
        assert out.get("success") is True
        fr = out.get("file_reference")
        assert isinstance(fr, str)
        assert fr == "RegistryLine.txt"
        rjson = read_text_based_file.invoke(
            {"file_reference": fr, "read_html_as_markdown": True, "agent": ag}
        )
        assert isinstance(rjson, str)
        assert "registry line" in rjson
        for pth in list(ag.file_registry.values()):
            if os.path.isfile(pth):
                with contextlib.suppress(OSError):
                    os.unlink(pth)


class TestImageHttpHelpers:
    def test_image_get_path_has_id_query(self) -> None:
        s = image_get_path("x-y")
        assert "webapi/Image" in s
        assert "imageId" in s

    def test_create_image_posts(self) -> None:
        with patch("tools.platform_record_image.requests_._post_request") as p:
            p.return_value = {
                "success": True,
                "raw_response": {"response": "new-img-id"},
            }
            out = create_image_file("a.png", b"ab")
        assert out.get("success") is True
        body, path = p.call_args[0]
        assert path == WEBAPI_IMAGE_CREATE
        assert body["name"] == "a.png"
        assert "content" in body

    def test_extract_created_id(self) -> None:
        assert (
            extract_created_id(
                {
                    "success": True,
                    "raw_response": {"response": "z1"},
                }
            )
            == "z1"
        )

    def test_display_name_and_payload(self) -> None:
        m = {"name": "p.png", "format": "PNG", "content": "QUJD"}
        assert display_filename_for_image_model(m) == "p.png"
        p = get_image_file_payload("i1", image_model=m)
        assert p.get("success") is True
        assert p.get("content") == "QUJD"
        assert p.get("filename") == "i1.png"


class TestFetchRecordImageFile:
    def test_no_agent_writes_bytes(self) -> None:
        rid = "r1"
        b64c = base64.b64encode(b"img-bytes").decode("ascii")
        with (
            patch(
                "tools.templates_tools.tool_record_image.resolve_id_from_record_property",
                return_value=(None, "img-1"),
            ),
            patch(
                "tools.templates_tools.tool_record_image.get_image_model",
                return_value={
                    "success": True,
                    "model": {
                        "name": "f.png",
                        "format": "png",
                        "content": b64c,
                    },
                },
            ),
        ):
            out = fetch_record_image_file.invoke(
                {
                    "record_id": rid,
                    "image_attribute_system_name": "ImAttr",
                    "multivalue_index": 0,
                    "agent": None,
                }
            )
        assert out.get("success") is True
        ref = out.get("file_reference")
        assert isinstance(ref, str)
        assert os.path.isabs(ref)
        try:
            with open(ref, "rb") as f:
                assert f.read() == b"img-bytes"
        finally:
            if isinstance(ref, str) and os.path.isfile(ref):
                os.unlink(ref)
