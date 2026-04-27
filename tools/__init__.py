"""
Tools Package
=============

This package contains all tool-related modules for the CMW Platform agent.

Key Modules:
- applications_tools: Application and template management tools
- attributes_tools: Attribute management tools for all attribute types
- templates_tools: Template-related tools and operations
- tool_utils: Common tool utilities and helpers
- models: Data models and schemas for tools
- requests_: HTTP request utilities and helpers
- tools: Core tool functions and classes

Subpackages provide organized access to specific tool categories:
- applications_tools: list_applications, list_templates
- attributes_tools: All attribute CRUD operations (text, boolean, datetime, etc.)
- templates_tools: list_attributes
"""

# Initialize logging for tools context (idempotent)
try:
    from agent_ng.logging_config import setup_logging  # type: ignore
    setup_logging()
except Exception:
    # Tools can be used standalone; ignore if agent_ng not available
    pass

# Tools - Standalone tools
from .get_datetime import get_current_datetime

# Import all tool modules
from . import (
    applications_tools,
    attributes_tools,
    localization_tools,
    models,
    platform_entity_resolver,
    requests_,
    templates_tools,
    tool_utils,
    tools,
)

# Import key functions from subpackages for convenience
from .applications_tools import (
    get_platform_entity_url,
    get_record_url,
    list_applications,
    list_templates,
)
from .localization_tools import localize_aliases
from .platform_entity_resolver import resolve_entity
from .attributes_tools import (
    archive_or_unarchive_attribute,
    # General operations
    delete_attribute,
    # Account attributes
    edit_or_create_account_attribute,
    # Boolean attributes
    edit_or_create_boolean_attribute,
    # DateTime attributes
    edit_or_create_date_time_attribute,
    # Document attributes
    edit_or_create_document_attribute,
    # Drawing attributes
    edit_or_create_drawing_attribute,
    # Duration attributes
    edit_or_create_duration_attribute,
    # Enum attributes
    edit_or_create_enum_attribute,
    # Image attributes
    edit_or_create_image_attribute,
    # Decimal/Numeric attributes
    edit_or_create_numeric_attribute,
    # Record attributes
    edit_or_create_record_attribute,
    # Role attributes
    edit_or_create_role_attribute,
    # Text attributes
    edit_or_create_text_attribute,
    get_attribute,
)
from .templates_tools import (
    # Button tools
    archive_unarchive_button,
    edit_or_create_button,
    # Form tools
    edit_or_create_form,
    # Dataset tools
    edit_or_create_dataset,
    get_dataset,
    list_datasets,
    # Record template
    edit_or_create_record_template,
    # Toolbar tools
    edit_or_create_toolbar,
    get_button,
    get_form,
    get_toolbar,
    # General operations
    list_attributes,
    list_buttons,
    list_forms,
    list_toolbars,
)

__all__ = [
    # Datetime tool
    "get_current_datetime",

    # Module imports
    "applications_tools",
    "attributes_tools",
    "templates_tools",
    "tool_utils",
    "models",
    "requests_",
    "tools",

    # Convenience function imports
    "list_applications",
    "list_templates",
    "list_attributes",
    "get_platform_entity_url",
    "get_record_url",

    # General operations
    "delete_attribute",
    "archive_or_unarchive_attribute",
    "get_attribute",

    # Record template
    "edit_or_create_record_template",

    # Form tools
    "edit_or_create_form",
    "get_form",
    "list_forms",

    # Dataset tools
    "edit_or_create_dataset",
    "get_dataset",
    "list_datasets",

    # Toolbar tools
    "get_toolbar",
    "edit_or_create_toolbar",
    "list_toolbars",

    # Button tools
    "get_button",
    "edit_or_create_button",
    "list_buttons",
    "archive_unarchive_button",

    # Text attributes
    "edit_or_create_text_attribute",

    # Boolean attributes
    "edit_or_create_boolean_attribute",

    # DateTime attributes
    "edit_or_create_date_time_attribute",

    # Decimal/Numeric attributes
    "edit_or_create_numeric_attribute",

    # Document attributes
    "edit_or_create_document_attribute",

    # Drawing attributes
    "edit_or_create_drawing_attribute",

    # Duration attributes
    "edit_or_create_duration_attribute",

    # Image attributes
    "edit_or_create_image_attribute",

    # Record attributes
    "edit_or_create_record_attribute",

    # Role attributes
    "edit_or_create_role_attribute",

    # Account attributes
    "edit_or_create_account_attribute",

    # Enum attributes
    "edit_or_create_enum_attribute",

    # Localization tools
    "localize_aliases",

    # Platform entity resolver
    "resolve_entity",
    "platform_entity_resolver",
]
