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

    endpoint = f"{ATTRIBUTE_ENDPOINT}/{application_system_name}"

    return execute_edit_or_create_operation(
        request_body=request_body,
        operation=operation,
        endpoint=endpoint,
        result_model=AttributeResult
    )

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