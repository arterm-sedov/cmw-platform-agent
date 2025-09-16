from ..tool_utils import *

@tool("edit_or_create_role_attribute", return_direct=False, args_schema=CommonAttributeFields)
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

    endpoint = f"{ATTRIBUTE_ENDPOINT}/{application_system_name}"

    return execute_edit_or_create_operation(
        request_body=request_body,
        operation=operation,
        endpoint=endpoint,
        result_model=AttributeResult
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