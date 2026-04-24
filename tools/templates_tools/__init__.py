# ruff: noqa: N999
"""Templates Tools Package.

This package contains tools for managing Comindware Platform templates and records
and listing template attributes.
"""

# Import all tool functions
from tools.templates_tools.tool_create_edit_record import create_edit_record
from tools.templates_tools.tool_list_attributes import list_attributes
from tools.templates_tools.tool_list_records import list_template_records
from tools.templates_tools.tool_record_document import (
    attach_file_to_record_document_attribute,
    fetch_record_document_file,
    get_record_field_values,
)
from tools.templates_tools.tools_button import (
    archive_unarchive_button,
    edit_or_create_button,
    get_button,
    list_buttons,
)
from tools.templates_tools.tools_dataset import (
    edit_or_create_dataset,
    get_dataset,
    list_datasets,
)
from tools.templates_tools.tools_form import edit_or_create_form, get_form, list_forms
from tools.templates_tools.tools_record_template import edit_or_create_record_template
from tools.templates_tools.tools_toolbar import (
    edit_or_create_toolbar,
    get_toolbar,
    list_toolbars,
)

__all__ = [
    "archive_unarchive_button",
    "attach_file_to_record_document_attribute",
    "create_edit_record",
    "edit_or_create_button",
    "edit_or_create_dataset",
    "edit_or_create_form",
    "edit_or_create_record_template",
    "edit_or_create_toolbar",
    "fetch_record_document_file",
    "get_button",
    "get_dataset",
    "get_form",
    "get_record_field_values",
    "get_toolbar",
    "list_attributes",
    "list_buttons",
    "list_datasets",
    "list_forms",
    "list_template_records",
    "list_toolbars",
]
