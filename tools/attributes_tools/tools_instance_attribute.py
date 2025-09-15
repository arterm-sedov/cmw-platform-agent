from ..tool_utils import *

class EditOrCreateInstanceAttributeSchema(CommonAttributeFields):
    related_template_system_name: str = Field(
        description="System name of the template to link with the attribute. "
                    "RU: Связанный шаблон"
    )
    related_attribute_system_name: Optional[str] = Field(
        default=None,
        description="System name of a record attribute in the related template to back-link with the current attribute. "
                    "RU: Взаимная связь с атрибутом"
    )

@tool("edit_or_create_record_attribute", return_direct=False, args_schema=EditOrCreateInstanceAttributeSchema)
def edit_or_create_record_attribute(
    operation: str,
    name: str,
    system_name: str,
    application_system_name: str,
    template_system_name: str,
    related_template_system_name: str,
    description: Optional[str] = None,
    write_changes_to_the_log: Optional[bool] = False,
    calculate_value: Optional[bool] = False,
    expression_for_calculation: Optional[str] = None,
    store_multiple_values: Optional[bool] = False,
    related_attribute_system_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Edit or Create a record attribute.
    
    Record attribute is an attribute that is linked to records in a related template.
    
    Record attribute stores one or several IDs of the lined records in the related template.
    
    Record attribute can be mutually linked with the attribute in the related template. Mutually linked attributes are automatically cross-linked whenever the values of one of the attributes change.
    
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
        "type": "Instance",
        "name": name,
        "description": description,
        "isTracked": write_changes_to_the_log,
        "isMultiValue": store_multiple_values,
        "isCalculated": calculate_value if expression_for_calculation != None else False,
        "expression": expression_for_calculation,
        "instanceGlobalAlias": {
            "type": "Attribute" if related_attribute_system_name != None else "RecordTemplate",
            "owner": related_attribute_system_name,
            "alias": related_attribute_system_name if related_attribute_system_name != None else related_template_system_name
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


@tool("get_record_attribute", return_direct=False, args_schema=CommonGetAttributeFields)
def get_record_attribute(
    application_system_name: str,
    template_system_name: str,
    system_name: str
    ) -> Dict[str, Any]:
    """
    Get an record attribute in a given template and application.
    
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

    keys_to_remove = ['isTitle', 'isUnique', 'isIndexed', 'isMandatory', 'isOwnership', 'imageColorType', 'imagePreserveAspectRatio']

    for key in keys_to_remove:
        if key in result_body['response']:
            result_body['response'].pop(key, None)

    result.update({"raw_response": result_body['response']})
    validated = AttributeResult(**result)
    return validated.model_dump()

if __name__ == "__main__":
    results = edit_or_create_record_attribute.invoke({
        "operation": "create",
        "name": "Related Task",
        "system_name": "RelatedTask",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test",
        "related_template_system_name": "Task",
        "description": "Related task instance",
        "store_multiple_values": False
    })
    print(results)