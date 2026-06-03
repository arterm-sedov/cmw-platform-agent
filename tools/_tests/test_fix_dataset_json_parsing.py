"""
Test that dataset columns, sorting, grouping, totals accept JSON strings
via before-mode field validators (LLM may pass JSON as strings).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from pydantic import ValidationError

from tools.templates_tools.tools_dataset import EditOrCreateDatasetSchema


class TestDatasetColumnsJsonParsing:
    """columns field must accept JSON strings and parse them to dict."""

    def test_columns_accepts_dict_directly(self):
        obj = EditOrCreateDatasetSchema(
            operation="create",
            application_system_name="App",
            template_system_name="Tpl",
            dataset_system_name="defaultList",
            name="Test dataset",
            columns={"Title": {"name": "New Title"}},
        )
        assert obj.columns == {"Title": {"name": "New Title"}}

    def test_columns_accepts_json_string(self):
        json_str = '{"Title": {"name": "New Title"}, "Status": {"isHidden": true}}'
        obj = EditOrCreateDatasetSchema(
            operation="create",
            application_system_name="App",
            template_system_name="Tpl",
            dataset_system_name="defaultList",
            name="Test dataset",
            columns=json_str,
        )
        assert obj.columns == {"Title": {"name": "New Title"}, "Status": {"isHidden": True}}

    def test_columns_rejects_invalid_json(self):
        with pytest.raises(ValidationError):
            EditOrCreateDatasetSchema(
                operation="create",
                application_system_name="App",
                template_system_name="Tpl",
                dataset_system_name="defaultList",
                name="Test dataset",
                columns="not valid json {{{",
            )

    def test_columns_none_passthrough(self):
        obj = EditOrCreateDatasetSchema(
            operation="create",
            application_system_name="App",
            template_system_name="Tpl",
            dataset_system_name="defaultList",
            name="Test dataset",
        )
        assert obj.columns is None

    def test_columns_list_passthrough(self):
        """A list is not a dict — Pydantic should reject it."""
        with pytest.raises(ValidationError):
            EditOrCreateDatasetSchema(
                operation="create",
                application_system_name="App",
                template_system_name="Tpl",
                dataset_system_name="defaultList",
                name="Test dataset",
                columns=[{"Title": {"name": "X"}}],
            )


class TestDatasetSortingJsonParsing:
    """sorting field must accept JSON strings and parse them to list."""

    def test_sorting_accepts_list_directly(self):
        sorting_list = [
            {
                "propertyPath": [
                    {"type": "Attribute", "owner": "Tpl", "alias": "Title"}
                ],
                "direction": "Asc",
                "nullValuesOnTop": False,
            }
        ]
        obj = EditOrCreateDatasetSchema(
            operation="create",
            application_system_name="App",
            template_system_name="Tpl",
            dataset_system_name="defaultList",
            name="Test dataset",
            sorting=sorting_list,
        )
        assert obj.sorting == sorting_list

    def test_sorting_accepts_json_string(self):
        sorting_list = [
            {
                "propertyPath": [
                    {"type": "Attribute", "owner": "Tpl", "alias": "Title"}
                ],
                "direction": "Desc",
                "nullValuesOnTop": True,
            }
        ]
        json_str = json.dumps(sorting_list)
        obj = EditOrCreateDatasetSchema(
            operation="create",
            application_system_name="App",
            template_system_name="Tpl",
            dataset_system_name="defaultList",
            name="Test dataset",
            sorting=json_str,
        )
        assert obj.sorting == sorting_list

    def test_sorting_rejects_invalid_json(self):
        with pytest.raises(ValidationError):
            EditOrCreateDatasetSchema(
                operation="create",
                application_system_name="App",
                template_system_name="Tpl",
                dataset_system_name="defaultList",
                name="Test dataset",
                sorting="not json [",
            )

    def test_sorting_none_passthrough(self):
        obj = EditOrCreateDatasetSchema(
            operation="create",
            application_system_name="App",
            template_system_name="Tpl",
            dataset_system_name="defaultList",
            name="Test dataset",
        )
        assert obj.sorting is None


class TestDatasetGroupingJsonParsing:
    """grouping field must accept JSON strings and parse them to list."""

    def test_grouping_accepts_json_string(self):
        grouping_list = [
            {
                "propertyPath": [
                    {"type": "Attribute", "owner": "Tpl", "alias": "Category"}
                ],
                "name": "Group",
                "direction": "Asc",
                "level": 1,
            }
        ]
        json_str = json.dumps(grouping_list)
        obj = EditOrCreateDatasetSchema(
            operation="create",
            application_system_name="App",
            template_system_name="Tpl",
            dataset_system_name="defaultList",
            name="Test dataset",
            grouping=json_str,
        )
        assert obj.grouping == grouping_list


class TestDatasetTotalsJsonParsing:
    """totals field must accept JSON strings and parse them to list."""

    def test_totals_accepts_json_string(self):
        totals_list = [
            {
                "propertyPath": [
                    {"type": "Attribute", "owner": "System#", "alias": "isDisabled"}
                ],
                "aggregationMethod": "Count",
                "type": "Boolean",
            }
        ]
        json_str = json.dumps(totals_list)
        obj = EditOrCreateDatasetSchema(
            operation="create",
            application_system_name="App",
            template_system_name="Tpl",
            dataset_system_name="defaultList",
            name="Test dataset",
            totals=json_str,
        )
        assert obj.totals == totals_list
