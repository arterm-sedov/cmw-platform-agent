# ruff: noqa: N999
"""Applications Tools — list apps/templates, URLs, and app management.

This package contains tools for managing Comindware Platform applications and
templates.

Available Tools:
- list_applications: List all applications in the platform
- list_templates: List all templates in a specific application
"""
from .tool_audit_process_schema import get_process_schema
from .tool_get_ontology_objects import get_ontology_objects
from .tool_list_applications import list_applications
from .tool_list_templates import list_templates
from .tool_platform_entity_url import get_platform_entity_url
from .tool_record_url import get_record_url
from .tool_update_object_property import update_object_property
from .tools_applications import create_app

__all__ = [
    "create_app",
    "get_ontology_objects",
    "get_platform_entity_url",
    "get_process_schema",
    "get_record_url",
    "list_applications",
    "list_templates",
    "update_object_property",
]
