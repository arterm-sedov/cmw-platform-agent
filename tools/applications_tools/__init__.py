"""
Applications Tools Package
=========================

This package contains tools for managing Comindware Platform applications and templates.

Available Tools:
- list_applications: List all applications in the platform
- list_templates: List all templates in a specific application
"""

# Import all tool functions
from .tool_list_applications import list_applications
from .tool_list_templates import list_templates
from .tool_platform_entity_url import get_platform_entity_url
from .tool_record_url import get_record_url
from .tool_get_process_schema import get_process_schema
from .tools_applications import create_app

__all__ = [
    'list_applications',
    'list_templates',
    'get_platform_entity_url',
    'get_record_url',
    'audit_process_schema',   
    'create_app'
]
