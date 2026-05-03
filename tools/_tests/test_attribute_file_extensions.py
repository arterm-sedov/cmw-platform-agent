"""Tests for case-insensitive file extensions validation in attribute schemas."""

import pytest

from tools.attributes_tools.tools_image_attribute import (
    EditOrCreateImageAttributeSchema,
    ALLOWED_EXTENSIONS_LIST as IMAGE_EXTENSIONS,
)
from tools.attributes_tools.tools_document_attribute import (
    EditOrCreateDocumentAttributeSchema,
    ALLOWED_EXTENSIONS_LIST as DOC_EXTENSIONS,
)


class TestImageAttributeExtensions:
    """Test case-insensitive and JSON array parsing for image attribute extensions."""

    def test_uppercase_extensions(self):
        """Uppercase extensions should be accepted."""
        schema = EditOrCreateImageAttributeSchema(
            operation="edit",
            name="Test",
            system_name="Test",
            application_system_name="Test",
            template_system_name="Test",
            file_extensions_filter=["PNG", "JPG", "BMP"],
        )
        assert schema.file_extensions_filter == ["PNG", "JPG", "BMP"]

    def test_lowercase_extensions(self):
        """Lowercase extensions should be accepted (case-insensitive)."""
        schema = EditOrCreateImageAttributeSchema(
            operation="edit",
            name="Test",
            system_name="Test",
            application_system_name="Test",
            template_system_name="Test",
            file_extensions_filter=["png", "jpg", "bmp"],
        )
        assert schema.file_extensions_filter == ["PNG", "JPG", "BMP"]

    def test_mixed_case_extensions(self):
        """Mixed case extensions should be accepted."""
        schema = EditOrCreateImageAttributeSchema(
            operation="edit",
            name="Test",
            system_name="Test",
            application_system_name="Test",
            template_system_name="Test",
            file_extensions_filter=["Png", "JPG", "bmp"],
        )
        assert schema.file_extensions_filter == ["PNG", "JPG", "BMP"]

    def test_json_array_string(self):
        """JSON array string should be parsed correctly."""
        schema = EditOrCreateImageAttributeSchema(
            operation="edit",
            name="Test",
            system_name="Test",
            application_system_name="Test",
            template_system_name="Test",
            file_extensions_filter='["PNG", "JPG", "BMP"]',
        )
        assert schema.file_extensions_filter == ["PNG", "JPG", "BMP"]

    def test_json_array_lowercase(self):
        """JSON array string with lowercase should be accepted."""
        schema = EditOrCreateImageAttributeSchema(
            operation="edit",
            name="Test",
            system_name="Test",
            application_system_name="Test",
            template_system_name="Test",
            file_extensions_filter='["png", "jpg", "bmp"]',
        )
        assert schema.file_extensions_filter == ["PNG", "JPG", "BMP"]

    def test_comma_separated_string(self):
        """Comma-separated string should be parsed correctly."""
        schema = EditOrCreateImageAttributeSchema(
            operation="edit",
            name="Test",
            system_name="Test",
            application_system_name="Test",
            template_system_name="Test",
            file_extensions_filter="PNG, JPG, BMP",
        )
        assert schema.file_extensions_filter == ["PNG", "JPG", "BMP"]

    def test_comma_separated_lowercase(self):
        """Comma-separated lowercase string should be accepted."""
        schema = EditOrCreateImageAttributeSchema(
            operation="edit",
            name="Test",
            system_name="Test",
            application_system_name="Test",
            template_system_name="Test",
            file_extensions_filter="png, jpg, bmp",
        )
        assert schema.file_extensions_filter == ["PNG", "JPG", "BMP"]

    def test_invalid_extension_rejected(self):
        """Invalid extensions should be rejected."""
        with pytest.raises(ValueError, match="Invalid file extensions"):
            EditOrCreateImageAttributeSchema(
                operation="edit",
                name="Test",
                system_name="Test",
                application_system_name="Test",
                template_system_name="Test",
                file_extensions_filter=["PNG", "GIF"],
            )

    def test_none_accepted(self):
        """None should be accepted."""
        schema = EditOrCreateImageAttributeSchema(
            operation="edit",
            name="Test",
            system_name="Test",
            application_system_name="Test",
            template_system_name="Test",
            file_extensions_filter=None,
        )
        assert schema.file_extensions_filter is None


class TestDocumentAttributeExtensions:
    """Test case-insensitive and JSON array parsing for document attribute extensions."""

    def test_uppercase_extensions(self):
        """Uppercase extensions should be accepted."""
        schema = EditOrCreateDocumentAttributeSchema(
            operation="edit",
            name="Test",
            system_name="Test",
            application_system_name="Test",
            template_system_name="Test",
            file_extensions_filter=["TXT", "PNG", "DOCX"],
        )
        assert schema.file_extensions_filter == ["TXT", "PNG", "DOCX"]

    def test_lowercase_extensions(self):
        """Lowercase extensions should be accepted."""
        schema = EditOrCreateDocumentAttributeSchema(
            operation="edit",
            name="Test",
            system_name="Test",
            application_system_name="Test",
            template_system_name="Test",
            file_extensions_filter=["txt", "png", "docx"],
        )
        assert schema.file_extensions_filter == ["TXT", "PNG", "DOCX"]

    def test_json_array_string(self):
        """JSON array string should be parsed correctly."""
        schema = EditOrCreateDocumentAttributeSchema(
            operation="edit",
            name="Test",
            system_name="Test",
            application_system_name="Test",
            template_system_name="Test",
            file_extensions_filter='["TXT", "DOCX", "XLSX"]',
        )
        assert schema.file_extensions_filter == ["TXT", "DOCX", "XLSX"]

    def test_comma_separated_string(self):
        """Comma-separated string should be parsed correctly."""
        schema = EditOrCreateDocumentAttributeSchema(
            operation="edit",
            name="Test",
            system_name="Test",
            application_system_name="Test",
            template_system_name="Test",
            file_extensions_filter="txt, docx, xlsx",
        )
        assert schema.file_extensions_filter == ["TXT", "DOCX", "XLSX"]

    def test_invalid_extension_rejected(self):
        """Invalid extensions should be rejected."""
        with pytest.raises(ValueError, match="Invalid file extensions"):
            EditOrCreateDocumentAttributeSchema(
                operation="edit",
                name="Test",
                system_name="Test",
                application_system_name="Test",
                template_system_name="Test",
                file_extensions_filter=["TXT", "EXE"],
            )
