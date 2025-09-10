from typing import Any, Dict, List, Optional, Literal
from langchain.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator
import requests_
from models import AttributeResult

ATTRIBUTE_ENDPOINT = "webapi/Attribute"

class ArchiveOrUnarchiveAttribute(BaseModel):
    operation: Literal["archive", "unarchive"] = Field(
        description="Choose operation: Archive or Unarchive the attribute. RU: Архивировать, Разархивировать"
    )
    application_system_name: str = Field(
        description="System name of the application with the template where the attribute is archived or unarchived. RU: Системное имя приложения"
    )
    template_system_name: str = Field(
        description="System name of the template where the attribute is archived or unarchived. RU: Системное имя шаблона"
    )
    system_name: str = Field(
        description="System name of the attribute to archive or unarchive. RU: Системное имя атрибута"
    )

    @field_validator("operation", mode="before")
    @classmethod
    def normalize_operation(cls, v: str) -> str:
        if v is None:
            return v
        value = str(v).strip().lower()
        mapping = {
            "архивировать": "archive",
            "разархивировать": "unarchive"
        }
        return mapping.get(value, value)

    @field_validator("system_name", "application_system_name", "template_system_name", mode="before")
    @classmethod
    def non_empty_str(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("must be a non-empty string")
        return v    

@tool("archive_or_unarchive_attribute", return_direct=False, args_schema=ArchiveOrUnarchiveAttribute)
def archive_or_unarchive_attribute(
    operation: str,
    application_system_name: str,
    template_system_name: str,
    system_name: str
    ) -> Dict[str, Any]:
    """
    Archive or unarchive an attribute in the given template and application.
    
    Returns:
        dict: {
            "success": bool - True if the attribute was archived or unarchived successfully
            "status_code": int - HTTP response status code  
            "raw_response": dict|str|None - Raw response for auditing or payload body (sanitized)
            "error": str|None - Error message if operation failed
        }
    """

    attribute_global_alias = f"Attribute@{template_system_name}.{system_name}"

    try:
        if operation == "archive":
            result = requests_._put_request(None, f"{ATTRIBUTE_ENDPOINT}/{application_system_name}/{attribute_global_alias}/Disable")
        elif operation == "unarchive" or operation == "create":
            result = requests_._put_request(None, f"{ATTRIBUTE_ENDPOINT}/{application_system_name}/{attribute_global_alias}/Enable")
        else:
            result = {
                "success": False,
                "error": f"No such operation for attribute: {operation}. Available operations: create, edit",
                "status_code": 400
            }
    except Exception as e:
        result = {
            "success": False,
            "error": f"Tool execution failed: {str(e)}",
            "status_code": 500
        }

    # Ensure result is always a dict with proper structure
    if not isinstance(result, dict):
        result = {
            "success": False,
            "error": f"Unexpected result type: {type(result)}",
            "status_code": 500
        }
    
    # Add additional error information if the API call failed
    if not result.get("success", False) and result.get("error"):
        error_info = result.get("error", "")
        result["error"] = f"API operation failed: {error_info}"

    validated = AttributeResult(**result)
    return validated.model_dump()

if __name__ == "__main__":
    results = archive_or_unarchive_attribute.invoke({
        "operation": "unarchive",
        "system_name": "Test1",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test"
    })
    print(results)