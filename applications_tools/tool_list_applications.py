from typing import Any, Dict, List, Optional, Literal
from langchain.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.v1.types import NoneBytes
import requests_

ATTRIBUTE_ENDPOINT = "webapi/Solution"

class AttributeResult(BaseModel):
    success: bool
    status_code: int
    raw_response: dict | str | None = Field(default=None, description="Raw response for auditing or payload body")
    error: Optional[str] = Field(default=None)

@tool("list_applications", return_direct=False)
def list_applications() -> Dict[str, Any]:
    """
    List all applications.

    Returns (AttributeResult):
    - success (bool): True if applications list was fetched successfully
    - status_code (int): HTTP response status code
    - raw_response (object|null): Applications payload; sanitized (some keys may be removed)
    - error (string|null): Error message if any
    """

    result = requests_._get_request(f"{ATTRIBUTE_ENDPOINT}")

    # Check if the request was successful and has the expected structure
    if not result.get('success', False):
        return result
    
    result_body = result.get('raw_response')
    if result_body is None:
        result.update({"error": "No response data received from server"})
        return result
    
    # Check if result_body has the expected 'response' key
    if not isinstance(result_body, dict) or 'response' not in result_body:
        result.update({"error": "Unexpected response structure from server"})
        return result

    result.update({"raw_response": result_body['response']})
    validated = AttributeResult(**result)
    return validated.model_dump()

if __name__ == "__main__":
    results = list_attributes.invoke({
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test"
    })
    print(results)