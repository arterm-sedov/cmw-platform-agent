from ..tool_utils import *

class EditOrCreateBooleanAttributeSchema(CommonAttributeFields):
    pass


@tool("edit_or_create_boolean_attribute", return_direct=False, args_schema=EditOrCreateBooleanAttributeSchema)
def edit_or_create_boolean_attribute(
    operation: str,
    name: str,
    system_name: str,
    application_system_name: str,
    template_system_name: str,
    description: Optional[str] = None,
    write_changes_to_the_log: Optional[bool] = False,
    calculate_value: Optional[bool] = False,
    expression_for_calculation: Optional[str] = None
) -> Dict[str, Any]:
    """
    Edit or Create a boolean attribute (Логический атрибут).

    Boolean attribute stores `true` or `false`.

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
        "type": "Boolean",
        "name": name,
        "description": description,
        "isTracked": write_changes_to_the_log,
        "isCalculated": calculate_value if expression_for_calculation != None else False,
        "expression": expression_for_calculation,
    }

    endpoint = f"{ATTRIBUTE_ENDPOINT}/{application_system_name}"

    return execute_edit_or_create_operation(
        request_body=request_body,
        operation=operation,
        endpoint=endpoint,
        result_model=AttributeResult
    )

if __name__ == "__main__":
    results = edit_or_create_boolean_attribute.invoke({
        "operation": "create",
        "name": "Is Active",
        "system_name": "IsActive",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test",
        "description": "Indicates if the record is active",
        "write_changes_to_the_log": False
    })
    print(results)