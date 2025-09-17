from ..tool_utils import *

class EditOrCreateDurationAttributeSchema(CommonAttributeFields):
    display_format: Literal[
        "DurationHMS",
        "DurationHM",
        "DurationHMSTime",
        "DurationD8HM",
        "DurationD24HM",
        "DurationFullShort",
        "DurationHMTime"
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

@tool("edit_or_create_duration_attribute", return_direct=False, args_schema=EditOrCreateDurationAttributeSchema)
def edit_or_create_duration_attribute(
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
    Edit or Create a duration attribute (Длительность).
    
    Supports various duration display formats.
    
    - Strictly follow argument schema and its built-in descriptions.
    - Refer to these examples when choosing display_format:
        - DurationHMS: 2 ч 55 м 59 с
        - DurationHM: 2 ч 55 м
        - DurationHMSTime: 2:55:59 (макс 23:59:59)
        - DurationD8HM: 3 д 12 ч 55 м (8-часовой день)
        - DurationD24HM: 1 д 12 ч 55 м (24-часовой день)
        - DurationFullShort: 1 д 12 ч 55 м 59 с
        - DurationHMTime: 2:55 (макс. 23:59)

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
        "type": "Duration",
        "format": display_format,
        "name": name,
        "description": description,
        "isTracked": write_changes_to_the_log,
        "isTitle": use_as_record_title,
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
    results = edit_or_create_duration_attribute.invoke({
        "operation": "create",
        "name": "Processing Time",
        "system_name": "ProcessingTime",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test",
        "display_format": "DurationHMS",
        "description": "Time required to process the item",
        "use_as_record_title": False
    })
    print(results)