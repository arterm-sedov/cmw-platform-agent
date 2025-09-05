from typing import Any, Dict, List, Optional
from typing import Optional
from langchain.tools import tool
import requests_

ATTRIBUTE_ENDPOINT = "webapi/Attribute"

@tool("archive_or_unarchive_attribute", return_direct=False)
def archive_or_unarchive_attribute(
    operation: str,
    application_system_name: str,
    template_system_name: str,
    system_name: str
    ) -> Dict[str, Any]:
    """
    {
        "Description": "Archive or Unarchive a attribute",
        "Arguments": {
            "operation": {
                "Russian names": ["Архивировать", "Разархивировать"],
                "variants": ["archive", "unarchive"],
                "description": "Choose operation: Archive or Unarchive the attribute."
            },
            "system_name": {
                "Russian name": "Системное имя",
                "English name": "System name",
                "type": "string",
                "description": "Unique system name of the attribute"
            },
            "application_system_name": {
                "Russian name": "Системное имя приложения",
                "English name": "Application system name",
                "type": "string",
                "description": "System name of the application with the template where the attribute is created"
            },
            "template_system_name": {
                "Russian name": "Системное имя шаблона",
                "English name": "Template system name",
                "type": "string",
                "description": "System name of the template where the attribute is created"
            }
        },
        "Returns": {
            "success": {
                "type": "boolean",
                "description": "True if attribute was fetched successfully"
            },
            "status_code": {
                "type": "integer",
                "description": "HTTP response status code"
            },
            "raw_response": {
                "type": "string",
                "description": "Raw response text for auditing"
            },
            "error": {
                "type": "string",
                "default": null,
                "description": "Error message if any"
            }
        }        
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

    return result

if __name__ == "__main__":
    results = archive_or_unarchive_attribute.invoke({
        "operation": "unarchive",
        "system_name": "Test1",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test"
    })
    print(results)