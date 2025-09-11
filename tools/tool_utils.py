from typing import Any, Dict, List, Optional, Literal
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator
import tools.requests_ as requests_
from tools.models import (
    AttributeResult, 
    CommonAttributeFields, 
    CommonGetAttributeFields,
    normalize_operation_archive_unarchive
)

# Common constants
ATTRIBUTE_ENDPOINT = "webapi/Attribute"

def remove_nones(obj: Any) -> Any:
    """
    Recursively remove None values from dicts/lists to keep payload minimal and consistent with Platform expectations.
    
    This utility function is used across all tool modules to clean up data before sending to the API,
    ensuring that only meaningful data is transmitted and reducing payload size.
    
    Args:
        obj: The object to clean (dict, list, or any other type)
        
    Returns:
        Cleaned object with None values removed
    """
    if isinstance(obj, dict):
        return {k: remove_nones(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [remove_nones(v) for v in obj if v is not None]
    return obj
