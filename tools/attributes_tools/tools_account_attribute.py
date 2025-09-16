from ..tool_utils import *

class EditOrCreateAccountAttributeSchema(CommonAttributeFields):
    use_as_record_title: bool = Field(
        default=False,
        description="Set to `True` to display as a template record title."
            "RU: Использовать как заголовок записей",
    )
    related_template_system_name: str = Field(
        default="_Account",
        description="System name of the template to associate with the attribute."
            "RU: Связанный шаблон"
    )

@tool("edit_or_create_account_attribute", return_direct=False, args_schema=EditOrCreateAccountAttributeSchema)
def edit_or_create_account_attribute(
    operation: str,
    name: str,
    system_name: str,
    application_system_name: str,
    template_system_name: str,
    description: Optional[str] = None,
    use_as_record_title: Optional[bool] = False,
    write_changes_to_the_log: Optional[bool] = False,
    calculate_value: Optional[bool] = False,
    expression_for_calculation: Optional[str] = None,
    related_template_system_name: Optional[str] = "_Account",
    store_multiple_values: Optional[bool] = False
) -> Dict[str, Any]:
    """
    Edit or Create an account attribute.
    
    Account attribute is an attribute that is linked to accounts in the system.
    
    Account attribute stores one or several linked account IDs.
    
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
        "type": "Account",
        "name": name,
        "description": description,
        "isTracked": write_changes_to_the_log,
        "isMultiValue": store_multiple_values,
        "isTitle": use_as_record_title,
        "isCalculated": calculate_value if expression_for_calculation != None else False,
        "expression": expression_for_calculation,
        "instanceGlobalAlias": {
            "type": "RecordTemplate",
            "alias": related_template_system_name
        }
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


@tool("get_account_attribute", return_direct=False, args_schema=CommonGetAttributeFields)
def get_account_attribute(
    application_system_name: str,
    template_system_name: str,
    system_name: str
    ) -> Dict[str, Any]:
    """
    Get an account attribute in a given template and application.
    
    Returns:
        dict: {
            "success": bool - True if operation completed successfully
            "status_code": int - HTTP response status code  
            "raw_response": dict|str|None - Raw response payload for auditing or payload body (sanitized)
            "error": str|None - Error message if operation failed
        }
    """

    attribute_global_alias = f"Attribute@{template_system_name}.{system_name}"

    result = requests_._get_request(f"{ATTRIBUTE_ENDPOINT}/{application_system_name}/{attribute_global_alias}")

    # Check if the request was successful and has the expected structure
    if not result.get('success', False):
        return result
    
    result_body = result.get('raw_response')
    if result_body is None:
        result.update({"error": "No response data received from server"})
        return result
    
    # Check if result_body has the expected 'response' key
    if not isinstance(result_body, dict) or 'response' not in result_body:
        result.update({"error": "Unexpected response structure from server"})
        return result

    keys_to_remove = ['isUnique', 'isIndexed', 'isMandatory', 'isOwnership', 'imageColorType', 'imagePreserveAspectRatio']

    for key in keys_to_remove:
        if key in result_body['response']:
            result_body['response'].pop(key, None)

    # Extract the data
    data = result_body['response']
    
    # Create the final result with the data
    final_result = {
        "success": True,
        "status_code": result.get("status_code", 200),
        "data": data,
        "error": None
    }
    
    validated = AttributeResult(**final_result)
    return validated.model_dump()

if __name__ == "__main__":
    results = edit_or_create_account_attribute.invoke({
        "operation": "create",
        "name": "Assigned User",
        "system_name": "AssignedUser",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test",
        "description": "User assigned to this record",
        "related_template_system_name": "_Account",
        "use_as_record_title": False
    })
    print(results)