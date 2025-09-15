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

# Import all tool modules
from . import applications_tools
from . import attributes_tools
from . import templates_tools
from . import tool_utils
from . import models
from . import requests_
from . import tools

# Import key functions from subpackages for convenience
from .applications_tools import list_applications, list_templates
from .attributes_tools import (
    # General operations
    delete_attribute, archive_or_unarchive_attribute,
    
    # Text attributes
    edit_or_create_text_attribute, get_text_attribute,
    
    # Boolean attributes
    edit_or_create_boolean_attribute, get_boolean_attribute,
    
    # DateTime attributes
    edit_or_create_date_time_attribute, get_date_time_attribute,
    
    # Decimal/Numeric attributes
    edit_or_create_numeric_attribute, get_numeric_attribute,
    
    # Document attributes
    edit_or_create_document_attribute, get_document_attribute,
    
    # Drawing attributes
    edit_or_create_drawing_attribute, get_drawing_attribute,
    
    # Duration attributes
    edit_or_create_duration_attribute, get_duration_attribute,
    
    # Image attributes
    edit_or_create_image_attribute, get_image_attribute,
    
    # Record attributes
    edit_or_create_record_attribute, get_record_attribute,
    
    # Role attributes
    edit_or_create_role_attribute, get_role_attribute,
    
    # Account attributes
    edit_or_create_account_attribute, get_account_attribute,
    
    # Enum attributes
    edit_or_create_enum_attribute, get_enum_attribute
)
from .templates_tools import list_attributes

__all__ = [
    # Module imports
    'applications_tools',
    'attributes_tools', 
    'templates_tools',
    'tool_utils',
    'models',
    'requests_',
    'tools',
    
    # Convenience function imports
    'list_applications',
    'list_templates',
    'list_attributes',
    
    # General operations
    'delete_attribute',
    'archive_or_unarchive_attribute',
    
    # Text attributes
    'edit_or_create_text_attribute',
    'get_text_attribute',
    
    # Boolean attributes
    'edit_or_create_boolean_attribute',
    'get_boolean_attribute',
    
    # DateTime attributes
    'edit_or_create_date_time_attribute',
    'get_date_time_attribute',
    
    # Decimal/Numeric attributes
    'edit_or_create_numeric_attribute',
    'get_numeric_attribute',
    
    # Document attributes
    'edit_or_create_document_attribute',
    'get_document_attribute',
    
    # Drawing attributes
    'edit_or_create_drawing_attribute',
    'get_drawing_attribute',
    
    # Duration attributes
    'edit_or_create_duration_attribute',
    'get_duration_attribute',
    
    # Image attributes
    'edit_or_create_image_attribute',
    'get_image_attribute',
    
    # Record attributes
    'edit_or_create_record_attribute',
    'get_record_attribute',
    
    # Role attributes
    'edit_or_create_role_attribute',
    'get_role_attribute',
    
    # Account attributes
    'edit_or_create_account_attribute',
    'get_account_attribute',
    
    # Enum attributes
    'edit_or_create_enum_attribute',
    'get_enum_attribute'
]
