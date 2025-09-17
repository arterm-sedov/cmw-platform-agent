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
            "error": str|None - Error message if operation failed
            "attribute property 1": value 1,
            "attribute property 2": value 2,
            ...
            "attribute property N": value N
        }
    """

    

    attribute_global_alias = f"Attribute@{template_system_name}.{system_name}"
    endpoint = f"{ATTRIBUTE_ENDPOINT}/{application_system_name}/{attribute_global_alias}"

    return execute_get_operation(
        result_model=AttributeResult,
        response_mapping=ATTRIBUTE_RESPONSE_MAPPING,
        endpoint=endpoint
    )

if __name__ == "__main__":
    print("")