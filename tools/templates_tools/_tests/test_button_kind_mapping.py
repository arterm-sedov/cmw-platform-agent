"""
Unit tests for button kind mapping and validation.

Tests the @field_validator that maps LLM-friendly button kind terms
to CMW Platform API enum values.

Key test scenarios:
- "Trigger scenario" → "UserEvent" (default LLM term)
- Case-insensitive mapping (trigger_scenario, TRIGGER SCENARIO)
- All 29 valid API kinds pass through unchanged
- Invalid kinds raise ValueError
- Common kinds work (Create, Edit, Delete, Archive, Unarchive)
"""

import pytest
from pydantic import ValidationError

from tools.templates_tools.tools_button import EditOrCreateButtonSchema


class TestButtonKindMapping:
    """Test button kind validator maps LLM-friendly terms to API terms."""

    def test_trigger_scenario_maps_to_user_event(self):
        """LLM default 'Trigger scenario' should map to API 'UserEvent'."""
        schema = EditOrCreateButtonSchema(
            operation="create",
            application_system_name="TestApp",
            template_system_name="TestTemplate",
            button_system_name="test_button",
            name="Test Button",
            kind="Trigger scenario",
        )
        assert schema.kind == "UserEvent"

    def test_user_event_passes_through(self):
        """Direct API term 'UserEvent' should pass through unchanged."""
        schema = EditOrCreateButtonSchema(
            operation="create",
            application_system_name="TestApp",
            template_system_name="TestTemplate",
            button_system_name="test_button",
            name="Test Button",
            kind="UserEvent",
        )
        assert schema.kind == "UserEvent"

    def test_trigger_scenario_snake_case(self):
        """Snake_case variant 'trigger_scenario' should map to UserEvent."""
        schema = EditOrCreateButtonSchema(
            operation="create",
            application_system_name="TestApp",
            template_system_name="TestTemplate",
            button_system_name="test_button",
            name="Test Button",
            kind="trigger_scenario",
        )
        assert schema.kind == "UserEvent"

    def test_trigger_scenario_uppercase(self):
        """Uppercase 'TRIGGER SCENARIO' should map to UserEvent."""
        schema = EditOrCreateButtonSchema(
            operation="create",
            application_system_name="TestApp",
            template_system_name="TestTemplate",
            button_system_name="test_button",
            name="Test Button",
            kind="TRIGGER SCENARIO",
        )
        assert schema.kind == "UserEvent"

    def test_common_kinds_pass_through(self):
        """Common button kinds should pass through unchanged."""
        common_kinds = ["Create", "Edit", "Delete", "Archive", "Unarchive", "Script"]

        for kind in common_kinds:
            schema = EditOrCreateButtonSchema(
                operation="create",
                application_system_name="TestApp",
                template_system_name="TestTemplate",
                button_system_name="test_button",
                name="Test Button",
                kind=kind,
            )
            assert schema.kind == kind, f"Expected {kind}, got {schema.kind}"

    def test_all_29_api_kinds_valid(self):
        """All 29 API enum values should be accepted."""
        valid_kinds = [
            "Undefined", "Create", "Edit", "Delete", "Archive", "Unarchive",
            "ExportObject", "ExportList", "CreateRelated", "CreateToken",
            "RetryTokens", "Migrate", "StartCase", "StartLinkedCase",
            "StartProcess", "StartLinkedProcess", "CompleteTask", "ReassignTask",
            "Defer", "Accept", "Uncomplete", "Follow", "Unfollow",
            "Exclude", "Include", "Script", "Cancel", "EditDiagram", "UserEvent"
        ]

        for kind in valid_kinds:
            schema = EditOrCreateButtonSchema(
                operation="create",
                application_system_name="TestApp",
                template_system_name="TestTemplate",
                button_system_name="test_button",
                name="Test Button",
                kind=kind,
            )
            assert schema.kind == kind, f"Expected {kind}, got {schema.kind}"

    def test_case_insensitive_common_kinds(self):
        """Common kinds should work case-insensitively."""
        test_cases = [
            ("create", "Create"),
            ("CREATE", "Create"),
            ("edit", "Edit"),
            ("EDIT", "Edit"),
            ("delete", "Delete"),
            ("archive", "Archive"),
            ("unarchive", "Unarchive"),
            ("script", "Script"),
        ]

        for input_kind, expected_kind in test_cases:
            schema = EditOrCreateButtonSchema(
                operation="create",
                application_system_name="TestApp",
                template_system_name="TestTemplate",
                button_system_name="test_button",
                name="Test Button",
                kind=input_kind,
            )
            assert schema.kind == expected_kind, (
                f"Input '{input_kind}' should map to '{expected_kind}', "
                f"got '{schema.kind}'"
            )

    def test_invalid_kind_test_rejected(self):
        """Invalid kind 'Test' should raise ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            EditOrCreateButtonSchema(
                operation="create",
                application_system_name="TestApp",
                template_system_name="TestTemplate",
                button_system_name="test_button",
                name="Test Button",
                kind="Test",
            )

        error_msg = str(exc_info.value)
        assert "Invalid button kind" in error_msg or "kind" in error_msg.lower()

    def test_invalid_kind_random_rejected(self):
        """Invalid kind 'RandomInvalidKind' should raise ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            EditOrCreateButtonSchema(
                operation="create",
                application_system_name="TestApp",
                template_system_name="TestTemplate",
                button_system_name="test_button",
                name="Test Button",
                kind="RandomInvalidKind",
            )

        error_msg = str(exc_info.value)
        assert "Invalid button kind" in error_msg or "kind" in error_msg.lower()

    def test_default_kind_is_trigger_scenario(self):
        """Default kind should be 'Trigger scenario' (maps to UserEvent)."""
        schema = EditOrCreateButtonSchema(
            operation="create",
            application_system_name="TestApp",
            template_system_name="TestTemplate",
            button_system_name="test_button",
            name="Test Button",
            # kind not provided, should use default
        )
        # Pydantic doesn't run validators on default values, so it stays as "Trigger scenario"
        # The validator will run when the tool is invoked via LangChain
        assert schema.kind == "Trigger scenario"

    def test_edit_without_kind_uses_default(self):
        """Edit operation without kind parameter should use default."""
        schema = EditOrCreateButtonSchema(
            operation="edit",
            application_system_name="TestApp",
            template_system_name="TestTemplate",
            button_system_name="test_button",
            # kind not provided, should use default "Trigger scenario"
        )
        # Default value is used (validator doesn't run on defaults)
        assert schema.kind == "Trigger scenario"


class TestButtonKindEdgeCases:
    """Test edge cases and special scenarios."""

    def test_whitespace_handling(self):
        """Kind with extra whitespace should be handled."""
        schema = EditOrCreateButtonSchema(
            operation="create",
            application_system_name="TestApp",
            template_system_name="TestTemplate",
            button_system_name="test_button",
            name="Test Button",
            kind="  Trigger scenario  ",
        )
        assert schema.kind == "UserEvent"

    def test_mixed_case_with_underscores(self):
        """Mixed case with underscores should work."""
        schema = EditOrCreateButtonSchema(
            operation="create",
            application_system_name="TestApp",
            template_system_name="TestTemplate",
            button_system_name="test_button",
            name="Test Button",
            kind="Trigger_Scenario",
        )
        assert schema.kind == "UserEvent"

    def test_export_object_variants(self):
        """ExportObject should work in various cases."""
        test_cases = [
            ("ExportObject", "ExportObject"),
            ("exportobject", "ExportObject"),
            ("export_object", "ExportObject"),
            ("EXPORT_OBJECT", "ExportObject"),
        ]

        for input_kind, expected_kind in test_cases:
            schema = EditOrCreateButtonSchema(
                operation="create",
                application_system_name="TestApp",
                template_system_name="TestTemplate",
                button_system_name="test_button",
                name="Test Button",
                kind=input_kind,
            )
            assert schema.kind == expected_kind, (
                f"Input '{input_kind}' should map to '{expected_kind}', "
                f"got '{schema.kind}'"
            )

    def test_start_case_variants(self):
        """StartCase should work in various cases."""
        test_cases = [
            ("StartCase", "StartCase"),
            ("startcase", "StartCase"),
            ("start_case", "StartCase"),
            ("START_CASE", "StartCase"),
        ]

        for input_kind, expected_kind in test_cases:
            schema = EditOrCreateButtonSchema(
                operation="create",
                application_system_name="TestApp",
                template_system_name="TestTemplate",
                button_system_name="test_button",
                name="Test Button",
                kind=input_kind,
            )
            assert schema.kind == expected_kind, (
                f"Input '{input_kind}' should map to '{expected_kind}', "
                f"got '{schema.kind}'"
            )


class TestButtonKindDocumentation:
    """Test that kind field has proper documentation."""

    def test_kind_field_has_description(self):
        """Kind field should have comprehensive description."""
        schema_fields = EditOrCreateButtonSchema.model_fields
        kind_field = schema_fields.get("kind")

        assert kind_field is not None, "kind field should exist"
        assert kind_field.description is not None, "kind should have description"

        description = kind_field.description
        assert "Trigger scenario" in description, "Should mention Trigger scenario"
        assert "Create" in description, "Should mention Create"
        assert "UserEvent" in description or "custom scenario" in description.lower(), (
            "Should explain UserEvent/custom scenario"
        )

    def test_kind_default_is_trigger_scenario(self):
        """Kind field default should be 'Trigger scenario'."""
        schema_fields = EditOrCreateButtonSchema.model_fields
        kind_field = schema_fields.get("kind")

        assert kind_field is not None
        assert kind_field.default == "Trigger scenario", (
            "Default should be 'Trigger scenario' for LLM UX"
        )
