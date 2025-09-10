from tool_utils import *


@tool("delete_attribute", return_direct=False, args_schema=CommonGetAttributeFields)
def delete_attribute(
    application_system_name: str,
    template_system_name: str,
    system_name: str
    ) -> Dict[str, Any]:
    """
    Delete a text attribute by from the given template and application.
    
    Returns:
        dict: {
            "success": bool - True if the attribute was deleted successfully
            "status_code": int - HTTP response status code  
            "raw_response": dict|str|None - Raw response for auditing or payload body (sanitized)
            "error": str|None - Error message if operation failed
        }
    """

    attribute_global_alias = f"Attribute@{template_system_name}.{system_name}"

    result = requests_._delete_request(f"{ATTRIBUTE_ENDPOINT}/{application_system_name}/{attribute_global_alias}")

    validated = AttributeResult(**result)

    # Check if the request was successful and has the expected structure
    if not result.get('success', False):
        return validated.model_dump()

    return validated.model_dump()

if __name__ == "__main__":
    results = delete_attribute.invoke({
        "system_name": "Test2",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test"
    })
    print(results)