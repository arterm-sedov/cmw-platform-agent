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

__all__ = [
    'list_applications',
    'list_templates'
]
