from ..tool_utils import *

@tool("edit_or_create_drawing_attribute", return_direct=False, args_schema=CommonAttributeFields)
def edit_or_create_drawing_attribute(
    operation: str,
    name: str,
    system_name: str,
    application_system_name: str,
    template_system_name: str,
    description: Optional[str] = None,
    write_changes_to_the_log: Optional[bool] = False
) -> Dict[str, Any]:
    """
    Edit or Create a drawing attribute (Чертеж).

    Drawing attribute stores floor plans based on CAD files.

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
        "type": "Drawing",
        "name": name,
        "description": description,
        "isTracked": write_changes_to_the_log
    }

    endpoint = f"{ATTRIBUTE_ENDPOINT}/{application_system_name}"

    return execute_edit_or_create_operation(
        request_body=request_body,
        operation=operation,
        endpoint=endpoint,
        result_model=AttributeResult
    )

if __name__ == "__main__":
    results = edit_or_create_drawing_attribute.invoke({
        "operation": "create",
        "name": "Technical Drawing",
        "system_name": "TechnicalDrawing",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test",
        "description": "Technical drawing attachment",
        "write_changes_to_the_log": False
    })
    print(results)