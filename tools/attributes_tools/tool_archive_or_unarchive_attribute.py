from ..tool_utils import *


class ArchiveOrUnarchiveAttribute(CommonGetAttributeFields):
    operation: Literal["archive", "unarchive"] = Field(
        description="Choose operation: Archive or Unarchive the attribute."
                    "RU: Архивировать, Разархивировать"
    )

    @field_validator("operation", mode="before")
    @classmethod
    def normalize_operation_archive_unarchive(cls, v: str) -> str:
        return normalize_operation_archive_unarchive(v)


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