"""
Tests for get_platform_entity_url tool.

TDD: Tests define behavior contracts before implementation.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestEntityIdResolution:
    """Tests for resolving entity_id directly via GetAxioms."""

    def test_entity_id_valid_returns_url_and_metadata(self):
        """Valid entity ID returns URL + metadata from GetAxioms."""
        with (
            patch("tools.requests_._load_server_config") as mock_cfg,
            patch("tools.applications_tools.tool_platform_entity_url._resolve_entity_id") as mock_resolve,
        ):
            mock_cfg.return_value.base_url = "https://platform.example.com"
            mock_resolve.return_value = {
                "success": True,
                "alias": "ServiceRequests",
                "name": "Service Requests",
                "container": None,
                "solution": "sln.23",
                "app_alias": "CustomerPortal",
                "rdf_type": "cmw.container",
                "kind": None,
                "raw": {},
            }

            from tools.applications_tools.tool_platform_entity_url import get_platform_entity_url

            result = get_platform_entity_url.invoke({"entity_id": "oa.193"})

            assert result["success"] is True
            assert result["entity_url"] == "https://platform.example.com/#Resolver/oa.193"
            assert result["entity_id"] == "oa.193"
            assert result["system_name"] == "ServiceRequests"
            assert result["name"] == "Service Requests"
            assert result["entity_type"] == "Template"
            assert result["application"] == "CustomerPortal"
            assert result["parent_system_name"] is None

    def test_entity_id_application_has_no_parent(self):
        """Application ID returns URL with parent_system_name=None."""
        with (
            patch("tools.requests_._load_server_config") as mock_cfg,
            patch("tools.applications_tools.tool_platform_entity_url._resolve_entity_id") as mock_resolve,
        ):
            mock_cfg.return_value.base_url = "https://platform.example.com"
            mock_resolve.return_value = {
                "success": True,
                "alias": "CustomerPortal",
                "name": "Customer Portal",
                "container": None,
                "solution": None,
                "app_alias": None,
                "rdf_type": "cmw.solution",
                "kind": None,
                "raw": {},
            }

            from tools.applications_tools.tool_platform_entity_url import get_platform_entity_url

            result = get_platform_entity_url.invoke({"entity_id": "sln.23"})

            assert result["success"] is True
            assert result["entity_url"] == "https://platform.example.com/#Resolver/sln.23"
            assert result["parent_system_name"] is None

    def test_entity_id_button_has_parent_template(self):
        """Button ID returns URL with parent_system_name from container."""
        with (
            patch("tools.requests_._load_server_config") as mock_cfg,
            patch("tools.applications_tools.tool_platform_entity_url._resolve_entity_id") as mock_resolve,
        ):
            mock_cfg.return_value.base_url = "https://platform.example.com"
            mock_resolve.return_value = {
                "success": True,
                "alias": "approve_request",
                "name": "Approve Request",
                "container": "oa.193",
                "solution": "sln.23",
                "app_alias": "CustomerPortal",
                "rdf_type": "cmw.eventTrigger",
                "kind": "Trigger scenario",
                "raw": {},
            }

            from tools.applications_tools.tool_platform_entity_url import get_platform_entity_url

            result = get_platform_entity_url.invoke({"entity_id": "event.15199"})

            assert result["success"] is True
            assert result["entity_url"] == "https://platform.example.com/#Resolver/event.15199"
            assert result["system_name"] == "approve_request"
            assert result["parent_system_name"] == "oa.193"

    def test_entity_id_not_found_returns_error(self):
        """Invalid entity ID returns error."""
        with (
            patch("tools.requests_._load_server_config") as mock_cfg,
            patch("tools.applications_tools.tool_platform_entity_url._resolve_entity_id") as mock_resolve,
        ):
            mock_cfg.return_value.base_url = "https://platform.example.com"
            mock_resolve.return_value = {"success": False, "note": "Empty response"}

            from tools.applications_tools.tool_platform_entity_url import get_platform_entity_url

            result = get_platform_entity_url.invoke({"entity_id": "nonexistent.999"})

            assert result["success"] is False
            assert "not found" in result["error"].lower()

    def test_entity_id_empty_string_returns_error(self):
        """Empty entity_id returns error."""
        from tools.applications_tools.tool_platform_entity_url import get_platform_entity_url

        with pytest.raises(ValueError, match="non-empty"):
            get_platform_entity_url.invoke({"entity_id": ""})


class TestSystemNameLookup:
    """Tests for looking up entities by system_name across all applications."""

    def test_system_name_unique_match_returns_single_match(self):
        """Unique system name returns single match with URL."""
        with (
            patch("tools.requests_._load_server_config") as mock_cfg,
            patch("tools.applications_tools.tool_platform_entity_url.requests_._post_request") as mock_post,
        ):
            mock_cfg.return_value.base_url = "https://platform.example.com"
            mock_post.side_effect = [
                {
                    "success": True,
                    "raw_response": str([
                        {
                            "id": "oa.193",
                            "alias": "ServiceRequests",
                            "name": "Service Requests",
                            "solution": "sln.23",
                            "solutionName": "Customer Portal",
                            "solutionAlias": "CustomerPortal",
                            "type": "Record",
                        },
                    ]),
                },
                {"success": True, "raw_response": str([])},
                {"success": True, "raw_response": str([])},
                {"success": True, "raw_response": str([])},
                {"success": True, "raw_response": str([])},
            ]

            from tools.applications_tools.tool_platform_entity_url import get_platform_entity_url

            result = get_platform_entity_url.invoke({"system_name": "ServiceRequests"})

            assert result["success"] is True
            assert len(result["matches"]) == 1
            match = result["matches"][0]
            assert match["entity_id"] == "oa.193"
            assert match["entity_url"] == "https://platform.example.com/#Resolver/oa.193"
            assert match["system_name"] == "ServiceRequests"
            assert match["application"] == "CustomerPortal"

    def test_system_name_multiple_matches_returns_all(self):
        """Duplicate system names across apps return all matches for audit."""
        with (
            patch("tools.requests_._load_server_config") as mock_cfg,
            patch("tools.applications_tools.tool_platform_entity_url.requests_._post_request") as mock_post,
        ):
            mock_cfg.return_value.base_url = "https://platform.example.com"
            mock_post.side_effect = [
                {
                    "success": True,
                    "raw_response": str([
                        {
                            "id": "oa.193",
                            "alias": "schedule_maintenance",
                            "name": "Schedule Maintenance",
                            "solution": "sln.23",
                            "solutionName": "Customer Portal",
                            "solutionAlias": "CustomerPortal",
                            "type": "Record",
                        },
                        {
                            "id": "oa.456",
                            "alias": "schedule_maintenance",
                            "name": "Schedule Maintenance",
                            "solution": "sln.7",
                            "solutionName": "Field Service",
                            "solutionAlias": "FieldService",
                            "type": "Record",
                        },
                    ]),
                },
                {"success": True, "raw_response": str([])},
                {"success": True, "raw_response": str([])},
                {"success": True, "raw_response": str([])},
                {"success": True, "raw_response": str([])},
            ]

            from tools.applications_tools.tool_platform_entity_url import get_platform_entity_url

            result = get_platform_entity_url.invoke({"system_name": "schedule_maintenance"})

            assert result["success"] is True
            assert len(result["matches"]) == 2
            assert result["matches"][0]["application"] == "CustomerPortal"
            assert result["matches"][1]["application"] == "FieldService"

    def test_system_name_with_application_filters_results(self):
        """System name + application returns filtered single match."""
        with (
            patch("tools.requests_._load_server_config") as mock_cfg,
            patch("tools.applications_tools.tool_platform_entity_url.requests_._post_request") as mock_post,
        ):
            mock_cfg.return_value.base_url = "https://platform.example.com"
            mock_post.side_effect = [
                {
                    "success": True,
                    "raw_response": str([
                        {
                            "id": "oa.193",
                            "alias": "schedule_maintenance",
                            "name": "Schedule Maintenance",
                            "solution": "sln.23",
                            "solutionName": "Customer Portal",
                            "solutionAlias": "CustomerPortal",
                            "type": "Record",
                        },
                        {
                            "id": "oa.456",
                            "alias": "schedule_maintenance",
                            "name": "Schedule Maintenance",
                            "solution": "sln.7",
                            "solutionName": "Field Service",
                            "solutionAlias": "FieldService",
                            "type": "Record",
                        },
                    ]),
                },
                {"success": True, "raw_response": str([])},
                {"success": True, "raw_response": str([])},
                {"success": True, "raw_response": str([])},
                {"success": True, "raw_response": str([])},
            ]

            from tools.applications_tools.tool_platform_entity_url import get_platform_entity_url

            result = get_platform_entity_url.invoke({
                "system_name": "schedule_maintenance",
                "application": "CustomerPortal",
            })

            assert result["success"] is True
            assert len(result["matches"]) == 1
            assert result["matches"][0]["application"] == "CustomerPortal"

    def test_system_name_no_match_returns_empty_matches(self):
        """Unknown system name returns success with empty matches."""
        with (
            patch("tools.requests_._load_server_config") as mock_cfg,
            patch("tools.applications_tools.tool_platform_entity_url.requests_._post_request") as mock_post,
        ):
            mock_cfg.return_value.base_url = "https://platform.example.com"
            mock_post.return_value = {"success": True, "raw_response": str([])}

            from tools.applications_tools.tool_platform_entity_url import get_platform_entity_url

            result = get_platform_entity_url.invoke({"system_name": "nonexistent_alias"})

            assert result["success"] is True
            assert result["matches"] == []

    def test_system_name_empty_string_returns_error(self):
        """Empty system_name returns error."""
        from tools.applications_tools.tool_platform_entity_url import get_platform_entity_url

        with pytest.raises(ValueError, match="non-empty"):
            get_platform_entity_url.invoke({"system_name": ""})


class TestCombinedResolution:
    """Tests for providing both entity_id and system_name."""

    def test_both_matching_returns_url(self):
        """Matching entity_id + system_name returns URL."""
        with (
            patch("tools.requests_._load_server_config") as mock_cfg,
            patch("tools.applications_tools.tool_platform_entity_url._resolve_entity_id") as mock_resolve,
        ):
            mock_cfg.return_value.base_url = "https://platform.example.com"
            mock_resolve.return_value = {
                "success": True,
                "alias": "ServiceRequests",
                "name": "Service Requests",
                "container": None,
                "solution": "sln.23",
                "app_alias": "CustomerPortal",
                "rdf_type": "cmw.container",
                "kind": None,
                "raw": {},
            }

            from tools.applications_tools.tool_platform_entity_url import get_platform_entity_url

            result = get_platform_entity_url.invoke({
                "entity_id": "oa.193",
                "system_name": "ServiceRequests",
            })

            assert result["success"] is True
            assert result["entity_url"] == "https://platform.example.com/#Resolver/oa.193"

    def test_both_mismatching_returns_error(self):
        """Mismatching entity_id + system_name returns error."""
        with (
            patch("tools.requests_._load_server_config") as mock_cfg,
            patch("tools.applications_tools.tool_platform_entity_url._resolve_entity_id") as mock_resolve,
        ):
            mock_cfg.return_value.base_url = "https://platform.example.com"
            mock_resolve.return_value = {
                "success": True,
                "alias": "ServiceRequests",
                "name": "Service Requests",
                "container": None,
                "solution": "sln.23",
                "app_alias": "CustomerPortal",
                "rdf_type": "cmw.container",
                "kind": None,
                "raw": {},
            }

            from tools.applications_tools.tool_platform_entity_url import get_platform_entity_url

            result = get_platform_entity_url.invoke({
                "entity_id": "oa.193",
                "system_name": "WrongAlias",
            })

            assert result["success"] is False
            assert "mismatch" in result["error"].lower() or "match" in result["error"].lower()

    def test_neither_provided_returns_error(self):
        """No entity_id or system_name returns error."""
        from tools.applications_tools.tool_platform_entity_url import get_platform_entity_url

        result = get_platform_entity_url.invoke({})

        assert result["success"] is False
        assert "provide" in result["error"].lower()


class TestEdgeCases:
    """Edge case tests."""

    def test_whitespace_trimming(self):
        """Handles leading/trailing whitespace in system_name."""
        with (
            patch("tools.requests_._load_server_config") as mock_cfg,
            patch("tools.applications_tools.tool_platform_entity_url.requests_._post_request") as mock_post,
        ):
            mock_cfg.return_value.base_url = "https://platform.example.com"
            mock_post.side_effect = [
                {
                    "success": True,
                    "raw_response": str([
                        {
                            "id": "oa.193",
                            "alias": "ServiceRequests",
                            "name": "Service Requests",
                            "solution": "sln.23",
                            "solutionName": "Customer Portal",
                            "solutionAlias": "CustomerPortal",
                            "type": "Record",
                        },
                    ]),
                },
                {"success": True, "raw_response": str([])},
                {"success": True, "raw_response": str([])},
                {"success": True, "raw_response": str([])},
                {"success": True, "raw_response": str([])},
            ]

            from tools.applications_tools.tool_platform_entity_url import get_platform_entity_url

            result = get_platform_entity_url.invoke({"system_name": "  ServiceRequests  "})

            assert result["success"] is True
            assert len(result["matches"]) == 1

    def test_base_url_not_configured_returns_error(self):
        """Missing base URL returns error."""
        with (
            patch("tools.requests_._load_server_config") as mock_cfg,
        ):
            mock_cfg.return_value.base_url = ""

            from tools.applications_tools.tool_platform_entity_url import get_platform_entity_url

            result = get_platform_entity_url.invoke({"entity_id": "oa.193"})

            assert result["success"] is False
            assert "base url" in result["error"].lower()
