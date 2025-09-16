from ..tool_utils import *

class EditOrCreateRoleAttributeSchema(CommonAttributeFields):
    pass


@tool("edit_or_create_role_attribute", return_direct=False, args_schema=EditOrCreateRoleAttributeSchema)
def edit_or_create_role_attribute(
    operation: str,
    name: str,
    system_name: str,
    application_system_name: str,
    template_system_name: str,
    description: Optional[str] = None,
    write_changes_to_the_log: Optional[bool] = False,
    calculate_value: Optional[bool] = False,
    expression_for_calculation: Optional[str] = None,
    store_multiple_values: Optional[bool] = False,
) -> Dict[str, Any]:
    """
    Edit or Create a role attribute.
    
    Role attribute is an attribute that is linked to roles in the system.
    
    Role attribute stores one or several linked role IDs.
    
    Returns:
        dict: {
            "success": bool - True if the attribute was created or edited successfully
            "status_code": int - HTTP response status code  
            "raw_response": dict|str|None - Raw response for auditing or payload body (sanitized)
            "error": str|None - Error message if operation failed
        }
    """

    request_body: Dict[str, Any] = {
        "globalAlias": {
            "owner": template_system_name,
            "type": "Undefined",
            "alias": system_name
        },
        "type": "Role",
        "name": name,
        "description": description,
        "isTracked": write_changes_to_the_log,
        "isMultiValue": store_multiple_values,
        "isCalculated": calculate_value if expression_for_calculation != None else False,
        "expression": expression_for_calculation
    }

        # Remove None values
    request_body = remove_nones(request_body) 

    try:
        if operation == "create":
            result = requests_._post_request(request_body, f"{ATTRIBUTE_ENDPOINT}/{application_system_name}")
        if operation == "edit" or operation == "create":
            result = requests_._put_request(request_body, f"{ATTRIBUTE_ENDPOINT}/{application_system_name}")
            print("edit is complited")
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


@tool("get_role_attribute", return_direct=False, args_schema=CommonGetAttributeFields)
def get_role_attribute(
    application_system_name: str,
    template_system_name: str,
    system_name: str
    ) -> Dict[str, Any]:
    """
    Get a role attribute in a given template and application.
    
    Role attribute is an attribute that is linked to roles in the system.
    
    Role attribute stores one or several linked role IDs.
    
    Returns:
        dict: {
            "success": bool - True if the attribute was fetched successfully
            "status_code": int - HTTP response status code  
            "raw_response": dict|str|None - Raw response payload for auditing or payload body (sanitized)
            "error": str|None - Error message if operation failed
        }
    """

    attribute_global_alias = f"Attribute@{template_system_name}.{system_name}"

    result = requests_._get_request(f"{ATTRIBUTE_ENDPOINT}/{application_system_name}/{attribute_global_alias}")

    return process_attribute_response(
        request_result=result,
        result_model=AttributeResult,
        response_mapping=ATTRIBUTE_RESPONE_MAPPING
    )

if __name__ == "__main__":
    results = edit_or_create_role_attribute.invoke({
        "operation": "create",
        "name": "Assigned Role",
        "system_name": "AssignedRole",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test",
        "description": "Role assigned to this record",
        "store_multiple_values": False
    })
    print(results)