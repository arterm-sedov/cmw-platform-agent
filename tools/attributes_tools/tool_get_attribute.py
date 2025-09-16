from ..tool_utils import *

@tool("get_attribute", return_direct=False, args_schema=CommonGetAttributeFields)
def get_attribute(
    application_system_name: str,
    template_system_name: str,
    system_name: str
    ) -> Dict[str, Any]:
    """
    Get a attribute in a given template and application.
    
    Returns:
        dict: {
            "success": bool - True if the attribute was fetched successfully
            "status_code": int - HTTP response status code  
            "raw_response": dict|str|None - Raw response payload for auditing or payload body (sanitized)
            "error": str|None - Error message if operation failed
        }
    """

    

    attribute_global_alias = f"Attribute@{template_system_name}.{system_name}"
    endpoint = f"{ATTRIBUTE_ENDPOINT}/{application_system_name}/{attribute_global_alias}"

    return execute_get_operation(
        result_model=AttributeResult,
        response_mapping=ATTRIBUTE_RESPONE_MAPPING,
        endpoint=endpoint
    )

if __name__ == "__main__":
    print("")