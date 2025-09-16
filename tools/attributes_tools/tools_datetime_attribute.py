from ..tool_utils import *

class EditOrCreateDateTimeAttributeSchema(CommonAttributeFields):
    display_format: Literal[
        "DateISO",
        "YearMonth",
        "TimeTracker",
        "DateTimeISO",
        "ShortDateLongTime",
        "ShortDateShortTime",
        "CondensedDateTime",
        "MonthDay",
        "CondensedDate",
        "ShortDate",
        "LongDate",
        "LongDateLongTime",
        "LongDateShortTime"
    ] = Field(
        description="Attribute display format. "
                    "RU: 'Формат отображения'."
    )
    use_as_record_title: bool = Field(
        default=False,
        description="Set to `True` to display attribute values as a template record title. "
                    "RU: 'Использовать как заголовок записей'",
    )

    @field_validator("display_format", mode="before")
    def non_empty_str(cls, v: Any) -> Any:
        """
        Validate that string fields are not empty.
        
        This field validator is automatically applied to the name, system_name, 
        application_system_name, and template_system_name fields in all schemas
        that inherit from CommonAttributeFields, ensuring consistent validation.
        """
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("must be a non-empty string")
        return v

@tool("edit_or_create_date_time_attribute", return_direct=False, args_schema=EditOrCreateDateTimeAttributeSchema)
def edit_or_create_date_time_attribute(
    operation: str,
    name: str,
    system_name: str,
    application_system_name: str,
    template_system_name: str,
    display_format: str,
    description: Optional[str] = None,
    use_as_record_title: Optional[bool] = False,
    write_changes_to_the_log: Optional[bool] = False,
    calculate_value: Optional[bool] = False,
    expression_for_calculation: Optional[str] = None
) -> Dict[str, Any]:
    """
    Edit or Create a date & time attribute.
    
    Supports various display formats for date and time representation.
    
    - Strictly follow argument schema and its built-in descriptions.
    - Refer to these examples when choosing display_format:
        - DateISO: 1986-09-04
        - YearMonth: сентябрь 1986
        - TimeTracker: остался 1 д 12 ч 55 мин
        - DateTimeISO: 1986-09-04 03:30:00
        - ShortDateLongTime: 04.09.1986 г. 03:30:00
        - ShortDateShortTime: 04.09.1986 03:30
        - CondensedDateTime: 4 сент. 1986 г. 03:30
        - MonthDay: 4 сентября
        - CondensedDate: 4 сент. 1986 г.
        - ShortDate: 4.09.1986
        - LongDate: 4 сентября 1986 г.
        - LongDateLongTime: 4 сентября 1986 г. 03:30:00
        - LongDateShortTime: 4 сентября 1986 г. 03:30

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
        "type": "DateTime",
        "format": display_format,
        "name": name,
        "description": description,
        "isTracked": write_changes_to_the_log,
        "isTitle": use_as_record_title,
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


@tool("get_date_time_attribute", return_direct=False, args_schema=CommonGetAttributeFields)
def get_date_time_attribute(
    application_system_name: str,
    template_system_name: str,
    system_name: str
    ) -> Dict[str, Any]:
    """
    Get a date & time attribute in a given template and application.
    
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
        response_mapping=KEYS_TO_REMOVE_MAPPING
    )

if __name__ == "__main__":
    results = edit_or_create_date_time_attribute.invoke({
        "operation": "create",
        "name": "Created Date",
        "system_name": "CreatedDate",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test",
        "display_format": "DateTimeISO",
        "description": "Date and time when the record was created",
        "use_as_record_title": False
    })
    print(results)