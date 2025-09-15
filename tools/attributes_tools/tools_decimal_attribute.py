from ..tool_utils import *

class EditOrCreateDecimalAttributeSchema(CommonAttributeFields):
    number_decimal_places: Optional[int] = Field(
        default=None,
        description="Number of decimal places of the attribute. "
                    "RU: 'Количество знаков после запятой'."
    )
    control_uniqueness: bool = Field(
        default=False,
        description="Set to `True` to control uniqueness of attribute values. "
                    "RU: 'Контролировать уникальность значения'",
    )
    use_as_record_title: bool = Field(
        default=False,
        description="Set to `True` to display attribute values as a template record title. "
                    "RU: 'Использовать как заголовок записей'",
    )
    group_digits_numbers: bool = Field(
        default=False,
        description="Set to `True` to group attribute value digits. "
                    "RU: 'Группировать разряды чисел'"
    ) 

@tool("edit_or_create_numeric_attribute", return_direct=False, args_schema=EditOrCreateDecimalAttributeSchema)
def edit_or_create_numeric_attribute(
    operation: str,
    name: str,
    system_name: str,
    application_system_name: str,
    template_system_name: str,
    number_decimal_places: int,
    description: Optional[str] = None,
    control_uniqueness: Optional[bool] = False,
    use_as_record_title: Optional[bool] = False,
    write_changes_to_the_log: Optional[bool] = False,
    calculate_value: Optional[bool] = False,
    expression_for_calculation: Optional[str] = None,
    group_digits_numbers: Optional[bool] = False
) -> Dict[str, Any]:
    """
    Edit or Create a numeroc attribute.
    
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
        "type": "Decimal",
        "name": name,
        "description": description,
        "isUnique": control_uniqueness,
        "isTracked": write_changes_to_the_log,
        "isDigitGrouping": group_digits_numbers,
        "isTitle": use_as_record_title,
        "isCalculated": calculate_value if expression_for_calculation != None else False,
        "expression": expression_for_calculation,
        "decimalPlaces": number_decimal_places
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


@tool("get_numeric_attribute", return_direct=False, args_schema=CommonGetAttributeFields)
def get_numeric_attribute(
    application_system_name: str,
    template_system_name: str,
    system_name: str
    ) -> Dict[str, Any]:
    """
    Get a numeric attribute in a given template and application.
    
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

    keys_to_remove = ['isMultiValue', 'isMandatory', 'isOwnership', 'instanceGlobalAlias', 'imageColorType', 'imagePreserveAspectRatio']

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
    results = edit_or_create_numeric_attribute.invoke({
        "operation": "create",
        "name": "Price",
        "system_name": "Price",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test",
        "number_decimal_places": 2,
        "description": "Price of the item",
        "control_uniqueness": False,
        "use_as_record_title": False
    })
    print(results)