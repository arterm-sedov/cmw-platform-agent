"""
Test that list_toolbars/list_buttons/list_datasets route through
execute_list_operation instead of manual raw parsing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from unittest.mock import MagicMock, patch


class TestListToolsUseExecuteListOperation:
    """The three previously raw-bypass list tools now delegate to
    execute_list_operation for consistent processing."""

    def test_list_toolbars_uses_execute_list_operation(self):
        with (
            patch("tools.templates_tools.tools_toolbar.requests_._get_request") as mock_get,
            patch(
                "tools.templates_tools.tools_toolbar.execute_list_operation"
            ) as mock_elo,
        ):
            mock_get.return_value = {"success": True, "raw_response": {"response": []}}
            from tools.templates_tools.tools_toolbar import list_toolbars

            list_toolbars.invoke({
                "application_system_name": "App",
                "template_system_name": "Tpl",
            })
            mock_elo.assert_called_once()

    def test_list_buttons_uses_execute_list_operation(self):
        with (
            patch("tools.templates_tools.tools_button.requests_._get_request") as mock_get,
            patch(
                "tools.templates_tools.tools_button.execute_list_operation"
            ) as mock_elo,
        ):
            mock_get.return_value = {"success": True, "raw_response": {"response": []}}
            from tools.templates_tools.tools_button import list_buttons

            list_buttons.invoke({
                "application_system_name": "App",
                "template_system_name": "Tpl",
            })
            mock_elo.assert_called_once()

    def test_list_datasets_uses_execute_list_operation(self):
        with (
            patch("tools.templates_tools.tools_dataset.requests_._get_request") as mock_get,
            patch(
                "tools.templates_tools.tools_dataset.execute_list_operation"
            ) as mock_elo,
        ):
            mock_get.return_value = {"success": True, "raw_response": {"response": []}}
            from tools.templates_tools.tools_dataset import list_datasets

            list_datasets.invoke({
                "application_system_name": "App",
                "template_system_name": "Tpl",
            })
            mock_elo.assert_called_once()


class TestResponseMappingsClean:
    """WebAPI models have no id field — mappings must not include dead entries."""

    def test_attribute_mapping_no_id(self):
        from tools.tool_utils import ATTRIBUTE_RESPONSE_MAPPING

        assert "id" not in ATTRIBUTE_RESPONSE_MAPPING

    def test_template_mapping_no_id(self):
        from tools.tool_utils import TEMPLATE_RESPONSE_MAPPING

        assert "id" not in TEMPLATE_RESPONSE_MAPPING

    def test_application_mapping_no_id(self):
        from tools.tool_utils import APPLICATION_RESPONSE_MAPPING

        assert "id" not in APPLICATION_RESPONSE_MAPPING


class TestProcessDataClean:
    """process_data must not unconditionally rename id→Platform ID."""

    def test_process_data_passes_through_non_mapped_keys(self):
        from tools.tool_utils import process_data

        data = {
            "globalAlias": {"type": "Form", "owner": "Tpl", "alias": "defaultForm"},
            "name": "Main Form",
            "isDefault": True,
        }
        result = process_data(data, "list_forms")
        assert "Platform ID" not in result
