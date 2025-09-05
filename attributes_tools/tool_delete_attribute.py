from typing import Any, Dict, List, Optional
from typing import Optional
from langchain.tools import tool
import requests_

ATTRIBUTE_ENDPOINT = "webapi/Attribute"

@tool("delete_attribute", return_direct=False)
def delete_attribute(
    application_system_name: str,
    template_system_name: str,
    system_name: str
    ) -> Dict[str, Any]:
    """
    {
        "Description": "Delete a attribute",
        "Arguments": {
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

    result = requests_._delete_request(f"{ATTRIBUTE_ENDPOINT}/{application_system_name}/{attribute_global_alias}")

    # Check if the request was successful and has the expected structure
    if not result.get('success', False):
        return result

    return result

if __name__ == "__main__":
    results = delete_attribute.invoke({
        "system_name": "Test2",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test"
    })
    print(results)