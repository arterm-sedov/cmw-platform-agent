"""
Attributes Tools Package
=======================

This package contains tools for managing Comindware Platform attributes of all types.

Available Tools:
- Text Attributes: edit_or_create_text_attribute, get_text_attribute
- Boolean Attributes: edit_or_create_boolean_attribute, get_boolean_attribute
- DateTime Attributes: edit_or_create_date_time_attribute, get_date_time_attribute
- Decimal Attributes: edit_or_create_numeric_attribute, get_numeric_attribute
- Document Attributes: edit_or_create_document_attribute, get_document_attribute
- Drawing Attributes: edit_or_create_drawing_attribute, get_drawing_attribute
- Duration Attributes: edit_or_create_duration_attribute, get_duration_attribute
- Image Attributes: edit_or_create_image_attribute, get_image_attribute
- Instance Attributes: edit_or_create_record_attribute, get_record_attribute
- Role Attributes: edit_or_create_role_attribute, get_role_attribute
- Account Attributes: edit_or_create_account_attribute, get_account_attribute
- General Operations: delete_attribute, archive_or_unarchive_attribute
"""

# Import all tool functions
from .tool_delete_attribute import delete_attribute
from .tool_archive_or_unarchive_attribute import archive_or_unarchive_attribute
from .tools_text_attribute import edit_or_create_text_attribute, get_text_attribute
from .tools_boolean_attribute import edit_or_create_boolean_attribute, get_boolean_attribute
from .tools_datetime_attribute import edit_or_create_date_time_attribute, get_date_time_attribute
from .tools_decimal_attribute import edit_or_create_numeric_attribute, get_numeric_attribute
from .tools_document_attribute import edit_or_create_document_attribute, get_document_attribute
from .tools_drawing_attribute import edit_or_create_drawing_attribute, get_drawing_attribute
from .tools_duration_attribute import edit_or_create_duration_attribute, get_duration_attribute
from .tools_image_attribute import edit_or_create_image_attribute, get_image_attribute
from .tools_instance_attribute import edit_or_create_record_attribute, get_record_attribute
from .tools_role_attribute import edit_or_create_role_attribute, get_role_attribute
from .tools_account_attribute import edit_or_create_account_attribute, get_account_attribute
from .tools_enum_attribute import edit_or_create_enum_attribute, get_enum_attribute

__all__ = [
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
    
    # Instance/Record attributes
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
