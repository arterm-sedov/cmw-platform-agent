from typing import Any, Dict, List, Optional, Literal
import re
from ..tool_utils import *

class PlainEnumValueModel(BaseModel):
    system_name: str = Field(
        description="Enum value system name. RU: Системное имя значения",
    )
    russian_name: Optional[str] = Field(
        description="Enum value Russian name. RU: Русское название значения",
    )
    english_name: Optional[str] = Field(
        default=None,
        description="Enum value English name. RU: Английское название значения",
    )
    deutsche_name: Optional[str] = Field(
        default=None,
        description="Enum value Deutsche name. RU: Немецкое название значения",
    )
    color: Optional[str] = Field(
        default=None,
        description="Enum value display color via hex code. RU: Цвет отображения значения"
    )

    @model_validator(mode='after')
    def check_at_least_one_name(cls, values):
        russian = values.russian_name
        english = values.english_name
        deutsche = values.deutsche_name

        if not any([russian, english, deutsche]):
            raise ValueError("At least one of 'russian_name', 'english_name', or 'deutsche_name' must be provided.")

        return values

    @field_validator('color')
    def validate_hex_color(cls, v):
        if v is None:
            return v
        if not re.match(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$', str(v)):
            raise ValueError('Color must be a valid hex color code, e.g. #RRGGBB or #RGB')
        return v

class EditOrCreateEnumAttributeSchema(CommonAttributeFields):
    display_format: Literal["Text", "Indicator", "Badge"] = Field(
        description="Attribute display format. RU: Формат отображения."
    )
    enum_values: List[PlainEnumValueModel] = Field(
        description="Attribute enum values. RU: Варианты значений атрибута"
    )

    @field_validator("display_format", "enum_values", mode="before")
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

def convert_plain_to_enum_value(plain: PlainEnumValueModel, attr_system_name: str) -> Dict[str, Any]:
    """Build API-ready variant payload from a plain enum value."""
    return {
        "alias": {
            "type": "Variant",
            "owner": attr_system_name,
            "alias": plain.system_name,
        },
        "name": {
            "ru": plain.russian_name,
            "en": plain.english_name,
            "de": plain.deutsche_name,
        },
        "color": plain.color,
    }

@tool("edit_or_create_enum_attribute", return_direct=False, args_schema=EditOrCreateEnumAttributeSchema)
def edit_or_create_enum_attribute(
    operation: str,
    name: str,
    system_name: str,
    application_system_name: str,
    template_system_name: str,
    display_format: str,
    enum_values: List[PlainEnumValueModel],
    description: Optional[str] = None,
    write_changes_to_the_log: Optional[bool] = False,
    calculate_value: Optional[bool] = False,
    expression_for_calculation: Optional[str] = None,
) -> Dict[str, Any]:
    r"""
    Edit or Create a enum attribute.
    
    IMPORTANT: When providing `enum_values`, you MUST follow this exact structure (see schema description for example):

    Example `enum_values`:
    [
      {
        "system_name": "status_active",
        "russian_name": "Активен",
        "english_name": "Active",
        "deutsche_name": "Aktiv",
        "color": "#4CAF50"
      }
    ]

    - `russian_name` is REQUIRED.
    - `system_name` is the system name of the enum value.
    - `color` must be valid hex (e.g. #RRGGBB or #RGB) or omitted.

    Returns:
        dict: {
            "success": bool - True if the attribute was created or edited successfully
            "status_code": int - HTTP response status code  
            "raw_response": dict|str|None - Raw response for auditing or payload body (sanitized)
            "error": str|None - Error message if operation failed
        }
    """

    convert_enum_values: List[Dict[str, Any]] = [
        convert_plain_to_enum_value(plain_enum_value, system_name)
        for plain_enum_value in enum_values
    ]

    request_body: Dict[str, Any] = {
        "globalAlias": {
            "owner": template_system_name,
            "type": "Undefined",
            "alias": system_name
        },
        "type": "Enum",
        "format": display_format,
        "name": name,
        "description": description,
        "isTracked": write_changes_to_the_log,
        "isCalculated": calculate_value if expression_for_calculation != None else False,
        "expression": expression_for_calculation,
        "variants": convert_enum_values
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

class GetEnumAttributeSchema(CommonGetAttributeFields):
    pass


@tool("get_enum_attribute", return_direct=False, args_schema=GetEnumAttributeSchema)
def get_enum_attribute(
    application_system_name: str,
    template_system_name: str,
    system_name: str
    ) -> Dict[str, Any]:
    """
    Get a enum attribute in a given template and application.
    
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

    return process_attribute_response(
        request_result=result,
        result_model=AttributeResult,
        response_mapping=ATTRIBUTE_RESPONE_MAPPING
    )

if __name__ == "__main__":
    results = edit_or_create_enum_attribute.invoke({
        "operation": "create",
        "name": "Status",
        "system_name": "Status",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test",
        "display_format": "Text",
        "enum_values": [
            {
                "system_name": "status_active",
                "russian_name": "Активен",
                "english_name": "Active",
                "deutsche_name": "Aktiv",
                "color": "#4CAF50"
            },
            {
                "system_name": "status_inactive",
                "russian_name": "Неактивен",
                "english_name": "Inactive",
                "deutsche_name": "Inaktiv",
                "color": "#9E9E9E"
            }
        ]
    })
    print(results)