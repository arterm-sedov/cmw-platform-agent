# ruff: noqa: N999
"""Templates Tools Package.

This package contains tools for managing Comindware Platform templates and records
and listing template attributes.
"""

# Import all tool functions
from tools.templates_tools.tool_create_edit_record import create_edit_record
from tools.templates_tools.tool_list_attributes import list_attributes
from tools.templates_tools.tool_list_records import list_template_records
from tools.templates_tools.tools_form import get_form
from tools.templates_tools.tools_record_template import edit_or_create_record_template

__all__ = [
    "create_edit_record",
    "edit_or_create_record_template",
    "get_form",
    "list_attributes",
    "list_template_records",
]
