from ..tool_utils import *

def _set_input_mask(display_format: str) -> str:
    # Setting validation mask via display format
    input_mask_mapping: Dict[str, str] = {
        "PlainText": None,
        "MarkedText": None,
        "HtmlText": None,
        "LicensePlateNumberRuMask":"([АВЕКМНОРСТУХавекмнорстух]{1}[0-9]{3}[АВЕКМНОРСТУХавекмнорстух]{2} [0-9]{3})",
        "IndexRuMask": "([0-9]{6})",
        "PassportRuMask": "([0-9]{4} [0-9]{6})",
        "INNMask": "([0-9]{10})",
        "OGRNMask": "([0-9]{13})",
        "IndividualINNMask": "([0-9]{12})",
        "PhoneRuMask": "(\+7 \([0-9]{3}\) [0-9]{3}-[0-9]{2}-[0-9]{2})",
        "EmailMask": "^(([a-zа-яё0-9_-]+\.)*[a-zа-яё0-9_-]+@[a-zа-яё0-9-]+(\.[a-zа-яё0-9-]+)*\.[a-zа-яё]{2,6})?$",
        "CustomMask": None
    }

    return input_mask_mapping.get(display_format, None)

class EditOrCreateTextAttributeSchema(CommonAttributeFields):
    display_format: Literal[
        "PlainText",
        "MarkedText",
        "HtmlText",
        "LicensePlateNumberRuMask",
        "IndexRuMask",
        "PassportRuMask",
        "INNMask",
        "OGRNMask",
        "IndividualINNMask",
        "PhoneRuMask",
        "EmailMask",
        "CustomMask",
    ] = Field(
        description="Attribute display format."
            "RU: Формат отображения. When `display_format=CustomMask` provide `custom_mask`."
    )
    custom_mask: Optional[str] = Field(
        default=None,
        description="A special formatting mask; fill only if `display_format=CustomMask`."
            "RU: Особая маска. Do not escape back slashes.",
    )
    control_uniqueness: bool = Field(
        default=False,
        description="Set to `True` to control uniqueness of attribute values."
            "RU: Контролировать уникальность значения",
    )
    use_as_record_title: bool = Field(
        default=False,
        description="Set to `True` to display as a template record title."
            "RU: Использовать как заголовок записей",
    )
    use_to_search_records: bool = Field(
        default=False,
        description="Set to `True` to allow the users to search the records by this attribute's value."
            "RU: Использовать для поиска записей",
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

    @model_validator(mode="after")
    def validate_masks_and_calc(self) -> "EditOrCreateTextAttributeSchema":
        if self.display_format == "CustomMask":
            if not self.custom_mask or not str(self.custom_mask).strip():
                raise ValueError("custom_mask is required when display_format is 'CustomMask'")
        else:
            # Ensure custom_mask is not accidentally provided for non-CustomMask
            if self.custom_mask is not None and str(self.custom_mask).strip() != "":
                raise ValueError("custom_mask must be omitted unless display_format is 'CustomMask'")

        if self.expression_for_calculation is None:
            # Calculation must be off when expression is not provided
            object.__setattr__(self, "calculate_value", False)
        else:
            # Turn on calculation if expression is provided
            object.__setattr__(self, "calculate_value", True)

        return self

@tool("edit_or_create_text_attribute", return_direct=False, args_schema=EditOrCreateTextAttributeSchema)
def edit_or_create_text_attribute(
    operation: str,
    name: str,
    system_name: str,
    application_system_name: str,
    template_system_name: str,
    display_format: str,
    description: Optional[str] = None,
    custom_mask: Optional[str] = None,
    control_uniqueness: Optional[bool] = False,
    use_as_record_title: Optional[bool] = False,
    use_to_search_records: Optional[bool] = False,
    write_changes_to_the_log: Optional[bool] = False,
    calculate_value: Optional[bool] = False,
    expression_for_calculation: Optional[str] = None
) -> Dict[str, Any]:
    r"""
    Edit or Create a text attribute.
    
    Supports various display formats including custom masks for common Russian data types.

    - Strictly follow argument schema and its built-in descriptions.
    - Predefined masks used in built-in display_formats. Refer to these examples when compiling custom masks:
        - LicensePlateNumberRuMask: ([АВЕКМНОРСТУХавекмнорстух]{1}[0-9]{3}[АВЕКМНОРСТУХавекмнорстух]{2} [0-9]{3})
        - IndexRuMask: ([0-9]{6})
        - PassportRuMask: ([0-9]{4} [0-9]{6})
        - INNMask: ([0-9]{10})
        - OGRNMask: ([0-9]{13})
        - IndividualINNMask: ([0-9]{12})
        - PhoneRuMask: (\+7 \([0-9]{3}\) [0-9]{3}-[0-9]{2}-[0-9]{2})
        - EmailMask: ^(([a-zа-яё0-9_-]+\.)*[a-zа-яё0-9_-]+@[a-zа-яё0-9-]+(\.[a-zа-яё0-9-]+)*\.[a-zа-яё]{2,6})?$

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
        "type": "String",
        "format": display_format,
        "name": name,
        "description": description,
        "isUnique": control_uniqueness,
        "isIndexed": use_to_search_records,
        "isTracked": write_changes_to_the_log,
        "isTitle": use_as_record_title,
        "isCalculated": calculate_value if expression_for_calculation != None else False,
        "expression": expression_for_calculation,
        "validationMaskRegex": custom_mask if display_format == "CustomMask" else _set_input_mask(display_format)
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


@tool("get_text_attribute", return_direct=False, args_schema=CommonGetAttributeFields)
def get_text_attribute(
    application_system_name: str,
    template_system_name: str,
    system_name: str
    ) -> Dict[str, Any]:
    """
    Get a text attribute in a given template and application.
    
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
    results = edit_or_create_text_attribute.invoke({
        "operation": "create",
        "name": "US Phone Number",
        "system_name": "USPhoneNumber",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test",
        "display_format": "CustomMask",
        "custom_mask": r"^+1-?\d{3}-?\d{3}-?\d{4}$",
        "control_uniqueness": False,
        "use_as_record_title": False
    })
    print(results)