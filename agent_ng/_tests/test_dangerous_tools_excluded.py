"""
Test that import_application and update_object_property are excluded from agent tools.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from unittest.mock import MagicMock, patch


class TestDangerousToolsExcluded:
    def test_import_application_excluded(self):
        from agent_ng.llm_manager import LLMManager

        mgr = LLMManager()
        tools = mgr.get_tools()
        tool_names = {t.name for t in tools}
        assert "import_application" not in tool_names

    def test_update_object_property_excluded(self):
        from agent_ng.llm_manager import LLMManager

        mgr = LLMManager()
        tools = mgr.get_tools()
        tool_names = {t.name for t in tools}
        assert "update_object_property" not in tool_names

    def test_export_application_still_bound(self):
        """export_application was NOT excluded — user chose to keep it."""
        from agent_ng.llm_manager import LLMManager

        mgr = LLMManager()
        tools = mgr.get_tools()
        tool_names = {t.name for t in tools}
        assert "export_application" in tool_names

    def test_other_tools_still_bound(self):
        """Verify random other tools are still present after exclusion."""
        from agent_ng.llm_manager import LLMManager

        mgr = LLMManager()
        tools = mgr.get_tools()
        tool_names = {t.name for t in tools}
        assert "list_applications" in tool_names
        assert "get_ontology_objects" in tool_names
        assert "edit_or_create_text_attribute" in tool_names
