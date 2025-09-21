from typing import Any, Dict, List, Optional, Literal
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.v1.types import NoneBytes
from .. import requests_ as requests_
from ..models import AttributeResult

ATTRIBUTE_ENDPOINT = "webapi/Solution"

@tool("list_applications", return_direct=False)
def list_applications() -> Dict[str, Any]:
    """
    List all applications, configured in the Platform.
    The resulting list depends on the user's access rights.
    
    Returns:
        dict: {
            "success": bool - True if application list was fetched successfully
            "status_code": int - HTTP response status code  
            "data": list|None - List of applications if successful
            "error": str|None - Error message if operation failed
        }
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

    # Extract the applications data
    applications_data = result_body['response']
    
    # Create the final result with the applications data
    final_result = {
        "success": True,
        "status_code": result.get("status_code", 200),
        "data": applications_data,
        "error": None
    }
    
    validated = AttributeResult(**final_result)
    return validated.model_dump()

if __name__ == "__main__":
    results = list_applications()
    print(results)