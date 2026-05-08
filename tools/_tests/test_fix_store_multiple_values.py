"""
Test that store_multiple_values is only on reference-type attribute schemas,
not leaked to non-reference types (text, boolean, numeric, etc.).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from pydantic import ValidationError

from tools.models import CommonAttributeFields, RefAttributeFields


class TestRefAttributeFieldsSeparation:
    """Reference-only attributes have store_multiple_values; non-reference don't."""

    def test_common_attribute_fields_no_store_multiple_values(self):
        """CommonAttributeFields must NOT have store_multiple_values."""
        fields = CommonAttributeFields.model_fields
        assert "store_multiple_values" not in fields, (
            "store_multiple_values must not be in CommonAttributeFields - "
            "only reference types support multi-value storage"
        )

    def test_ref_attribute_fields_has_store_multiple_values(self):
        """RefAttributeFields MUST have store_multiple_values."""
        fields = RefAttributeFields.model_fields
        assert "store_multiple_values" in fields
        field_info = fields["store_multiple_values"]
        assert field_info.default is False

    def test_common_fields_instantiate_without_store_multiple(self):
        """Instantiating CommonAttributeFields must not require store_multiple_values."""
        obj = CommonAttributeFields(
            operation="create",
            name="Test",
            system_name="TestAttr",
            application_system_name="App",
            template_system_name="Tpl",
        )
        assert obj.operation == "create"
        assert obj.name == "Test"

    def test_ref_fields_accept_store_multiple(self):
        """RefAttributeFields must accept and store the store_multiple_values parameter."""
        obj = RefAttributeFields(
            operation="create",
            name="Test",
            system_name="TestAttr",
            application_system_name="App",
            template_system_name="Tpl",
            store_multiple_values=True,
        )
        assert obj.store_multiple_values is True

    def test_ref_fields_default_store_multiple_false(self):
        """RefAttributeFields defaults store_multiple_values to False."""
        obj = RefAttributeFields(
            operation="create",
            name="Test",
            system_name="TestAttr",
            application_system_name="App",
            template_system_name="Tpl",
        )
        assert obj.store_multiple_values is False


class TestNonReferenceAttributeSchemas:
    """Verify non-reference attribute schemas inherit CommonAttributeFields
    and NOT RefAttributeFields."""

    def test_text_attribute_schema_no_store_multiple(self):
        from tools.attributes_tools.tools_text_attribute import (
            EditOrCreateTextAttributeSchema,
        )
        assert issubclass(EditOrCreateTextAttributeSchema, CommonAttributeFields)
        assert not issubclass(
            EditOrCreateTextAttributeSchema, RefAttributeFields
        )
        fields = EditOrCreateTextAttributeSchema.model_fields
        assert "store_multiple_values" not in fields

    def test_boolean_attribute_no_store_multiple(self):
        from tools.attributes_tools.tools_boolean_attribute import (
            EditOrCreateBooleanAttributeSchema,
        )
        fields = EditOrCreateBooleanAttributeSchema.model_fields
        assert "store_multiple_values" not in fields

    def test_numeric_attribute_no_store_multiple(self):
        from tools.attributes_tools.tools_decimal_attribute import (
            EditOrCreateDecimalAttributeSchema,
        )
        fields = EditOrCreateDecimalAttributeSchema.model_fields
        assert "store_multiple_values" not in fields

    def test_datetime_attribute_no_store_multiple(self):
        from tools.attributes_tools.tools_datetime_attribute import (
            EditOrCreateDateTimeAttributeSchema,
        )
        fields = EditOrCreateDateTimeAttributeSchema.model_fields
        assert "store_multiple_values" not in fields

    def test_duration_attribute_no_store_multiple(self):
        from tools.attributes_tools.tools_duration_attribute import (
            EditOrCreateDurationAttributeSchema,
        )
        fields = EditOrCreateDurationAttributeSchema.model_fields
        assert "store_multiple_values" not in fields

    def test_enum_attribute_no_store_multiple(self):
        from tools.attributes_tools.tools_enum_attribute import (
            EditOrCreateEnumAttributeSchema,
        )
        fields = EditOrCreateEnumAttributeSchema.model_fields
        assert "store_multiple_values" not in fields


class TestReferenceAttributeSchemas:
    """Verify reference-type attribute schemas inherit RefAttributeFields."""

    def test_role_attribute_has_store_multiple(self):
        from tools.attributes_tools.tools_role_attribute import (
            edit_or_create_role_attribute,
        )
        schema = edit_or_create_role_attribute.args_schema
        assert issubclass(schema, RefAttributeFields)
        fields = schema.model_fields
        assert "store_multiple_values" in fields

    def test_record_attribute_has_store_multiple(self):
        from tools.attributes_tools.tools_record_attribute import (
            EditOrCreateRecordAttributeSchema,
        )
        assert issubclass(EditOrCreateRecordAttributeSchema, RefAttributeFields)
        fields = EditOrCreateRecordAttributeSchema.model_fields
        assert "store_multiple_values" in fields

    def test_document_attribute_has_store_multiple(self):
        from tools.attributes_tools.tools_document_attribute import (
            EditOrCreateDocumentAttributeSchema,
        )
        assert issubclass(EditOrCreateDocumentAttributeSchema, RefAttributeFields)
        fields = EditOrCreateDocumentAttributeSchema.model_fields
        assert "store_multiple_values" in fields

    def test_image_attribute_has_store_multiple(self):
        from tools.attributes_tools.tools_image_attribute import (
            EditOrCreateImageAttributeSchema,
        )
        assert issubclass(EditOrCreateImageAttributeSchema, RefAttributeFields)
        fields = EditOrCreateImageAttributeSchema.model_fields
        assert "store_multiple_values" in fields

    def test_account_attribute_has_store_multiple(self):
        from tools.attributes_tools.tools_account_attribute import (
            EditOrCreateAccountAttributeSchema,
        )
        assert issubclass(EditOrCreateAccountAttributeSchema, RefAttributeFields)
        fields = EditOrCreateAccountAttributeSchema.model_fields
        assert "store_multiple_values" in fields


class TestNonRefToolsDontCrashOnStoreMultiple:
    """Non-reference tools must silently ignore store_multiple_values
    (field absent from schema, Pydantic extra='ignore')."""

    def test_text_tool_ignores_unknown_store_multiple(self):
        """Text attribute schema does not have store_multiple_values field."""
        from tools.attributes_tools.tools_text_attribute import (
            EditOrCreateTextAttributeSchema,
        )
        obj = EditOrCreateTextAttributeSchema(
            operation="create",
            name="Test",
            system_name="Test",
            application_system_name="App",
            template_system_name="Tpl",
        )
        assert "store_multiple_values" not in obj.model_dump()

    def test_boolean_tool_ignores_unknown_store_multiple(self):
        from tools.attributes_tools.tools_boolean_attribute import (
            EditOrCreateBooleanAttributeSchema,
        )
        obj = EditOrCreateBooleanAttributeSchema(
            operation="create",
            name="Test",
            system_name="Test",
            application_system_name="App",
            template_system_name="Tpl",
        )
        assert "store_multiple_values" not in obj.model_dump()

    def test_numeric_tool_ignores_unknown_store_multiple(self):
        from tools.attributes_tools.tools_decimal_attribute import (
            EditOrCreateDecimalAttributeSchema,
        )
        obj = EditOrCreateDecimalAttributeSchema(
            operation="create",
            name="Test",
            system_name="Test",
            application_system_name="App",
            template_system_name="Tpl",
            number_decimal_places=2,
        )
        assert "store_multiple_values" not in obj.model_dump()
