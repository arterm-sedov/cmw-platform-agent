"""
Templates Tools Package
======================

This package contains tools for managing Comindware Platform templates and their attributes.

Available Tools:
- list_attributes: List all attributes in a specific template
"""

# Import all tool functions
from .tool_list_attributes import list_attributes

__all__ = [
    'list_attributes'
]
