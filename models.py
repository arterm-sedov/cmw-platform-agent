from typing import Optional
from pydantic import BaseModel, Field


class AttributeResult(BaseModel):
    """
    Common result model for attribute-related operations across all tool modules.
    
    This model standardizes the response format for attribute operations,
    providing consistent success/error handling and response structure.
    """
    success: bool
    status_code: int
    raw_response: dict | str | list | None = Field(default=None)
    error: Optional[str] = Field(default=None)
