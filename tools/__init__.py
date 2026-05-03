# ruff: noqa: N999, E402, RUF022
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

import logging

# Initialize logging for tools context (idempotent)
logger = logging.getLogger(__name__)

try:
    from agent_ng.logging_config import setup_logging  # type: ignore[import-not-found]
    setup_logging()
except Exception as exc:
    # Tools can be used standalone; ignore if agent_ng not available.
    logger.debug("Skipping tools logging setup: %s", exc)

# Tools - Standalone tools
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
from .get_datetime import get_current_datetime
from .localization_tools import localize_aliases
from .platform_entity_resolver import resolve_entity
from .templates_tools import (
    # Button tools
    archive_unarchive_button,
    edit_or_create_button,
    # Dataset tools
    edit_or_create_dataset,
    # Form tools
    edit_or_create_form,
    # Record template
    edit_or_create_record_template,
    # Toolbar tools
    edit_or_create_toolbar,
    get_button,
    get_dataset,
    get_form,
    get_toolbar,
    # General operations
    list_attributes,
    list_buttons,
    list_datasets,
    list_forms,
    list_toolbars,
)

__all__ = [
    "applications_tools",
    "archive_or_unarchive_attribute",
    "archive_unarchive_button",
    "attributes_tools",
    "delete_attribute",
    "edit_or_create_account_attribute",
    "edit_or_create_boolean_attribute",
    "edit_or_create_button",
    "edit_or_create_dataset",
    "edit_or_create_date_time_attribute",
    "edit_or_create_document_attribute",
    "edit_or_create_drawing_attribute",
    "edit_or_create_duration_attribute",
    "edit_or_create_enum_attribute",
    "edit_or_create_form",
    "edit_or_create_image_attribute",
    "edit_or_create_numeric_attribute",
    "edit_or_create_record_attribute",
    "edit_or_create_record_template",
    "edit_or_create_role_attribute",
    "edit_or_create_text_attribute",
    "edit_or_create_toolbar",
    "get_attribute",
    "get_button",
    "get_current_datetime",
    "get_dataset",
    "get_form",
    "get_platform_entity_url",
    "get_record_url",
    "get_toolbar",
    "list_applications",
    "list_attributes",
    "list_buttons",
    "list_datasets",
    "list_forms",
    "list_templates",
    "list_toolbars",
    "localize_aliases",
    "models",
    "platform_entity_resolver",
    "requests_",
    "resolve_entity",
    "templates_tools",
    "tools",
    "tool_utils",
]
