from multiprocessing import Value
from typing import Any, Dict, List, Optional, Literal
from langchain.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator
import requests_
from models import AttributeResult
import re

ATTRIBUTE_ENDPOINT = "webapi/Attribute"

def _remove_nones(obj: Any) -> Any:
    """
    Recursively remove None values from dicts/lists to keep payload minimal and consistent with Platform expectations.
    """
    if isinstance(obj, dict):
        return {k: _remove_nones(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [ _remove_nones(v) for v in obj if v is not None]
    return obj

class VariantAliasModel(BaseModel):
    variant_type: Literal["Variant"] = Field(
        description="Variant type. RU: Тип значения",
        alias="type"
    )
    attribute_system_name: str = Field(
        description="Attribute system name. RU: Системное имя атрибута",
        alias="owner"
    )
    system_name: str = Field(
        description="Variant system name. Ru: Системное имя значения",
        alias="alias"
    )

    @model_validator(mode='before')
    def set_attribute_system_name_later(cls, values):
        return values

class VariantNameModel(BaseModel):
    english_name: Optional[str] = Field(
        default=None,
        description="Variant English name. RU: Английское название значения",
        alias="en"
    )
    russian_name: str = Field(
        description="Variant Russian name. RU: Русское название значения",
        alias="ru"
    )
    deutsche_name: Optional[str] = Field(
        default=None,
        description="Variant Deutsche name. RU: Немецкое название значения",
        alias="de"
    )

class VariantModel(BaseModel):
    sytem_name: VariantAliasModel = Field(
        alias="alias"
    )
    name: VariantNameModel
    color: Optional[str] = Field(
        default=None,
        description="Variant display color via hex code. RU: Цвет отображения значения"
    )

    @field_validator('color')
    def validate_hex_color(cls, v):
        if not re.match(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$', v):
            raise ValueError('Color must be a valid hex color code, e.g. #RRGGBB')

class EditOrCreateEnumAttributeSchema(BaseModel):
    operation: Literal["create", "edit"] = Field(
        description="Choose operation: Create or Edit the attribute. RU: Создать, Редактировать"
    )
    name: str = Field(
        description="Human-readable name of the attribute. RU: Название"
    )
    system_name: str = Field(
        description="System name of the attribute. RU: Системное имя"
    )
    application_system_name: str = Field(
        description="System name of the application with the template where the attribute is created or edited. RU: Системное имя приложения"
    )
    template_system_name: str = Field(
        description="System name of the template where the attribute is created or edited. RU: Системное имя шаблона"
    )
    display_format: Literal[
        "Text",
        "Indicator",
        "Badge"
    ] = Field(
        description="Attribute display format. RU: Формат отображения. When `display_format=CustomMask` provide `custom_mask`."
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable business-oriented description of the attribute (auto-generate if empty). RU: Описание",
    )
    write_changes_to_the_log: bool = Field(
        default=False,
        description="Set to `True` to log attribute value changes. RU: Записывать изменения в журнал",
    )
    calculate_value: bool = Field(
        default=False,
        description="Set to `True` to calculate the attribute value automatically. Relevant only when `expression_for_calculation` is provided. RU: Вычислять автоматически",
    )
    expression_for_calculation: Optional[str] = Field(
        default=None,
        description="Expression to calculate the attribute value automatically. User-provided. RU: Выражение для вычисления",
    )
    variants: List['VariantModel'] = Field(
        description="Attribute value variants. Ru: Варианты значений атрибута"
    )

    @model_validator(mode='after')
    def inject_attribute_system_name_into_aliases(self) -> 'EditOrCreateEnumAttributeSchema':
        for variant in self.variants:
            variant.sytem_name.attribute_system_name = self.system_name
        return self

    @field_validator("operation", mode="before")
    @classmethod
    def normalize_operation(cls, v: str) -> str:
        if v is None:
            return v
        value = str(v).strip().lower()
        mapping = {
            "создать": "create",
            "редактировать": "edit",
        }
        return mapping.get(value, value)

    @field_validator("name", "system_name", "application_system_name", "template_system_name", mode="before")
    @classmethod
    def non_empty_str(cls, v: Any) -> Any:
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

@tool("edit_or_create_enum_attribute", return_direct=False, args_schema=EditOrCreateEnumAttributeSchema)
def edit_or_create_enum_attribute(
    operation: str,
    name: str,
    system_name: str,
    application_system_name: str,
    template_system_name: str,
    display_format: str,
    variants: List[VariantModel],
    description: Optional[str] = None,
    write_changes_to_the_log: Optional[bool] = False,
    calculate_value: Optional[bool] = False,
    expression_for_calculation: Optional[str] = None,
) -> Dict[str, Any]:
    r"""
    Edit or Create a enum attribute.
    
    IMPORTANT: When providing `variants`, you MUST follow this exact structure (with aliases!):

    Example `variants`:
    [
        {
            "alias": {
                "type": "Variant",
                "owner": "<system_name of the attribute>",  # ← will be auto-filled, but you can omit or set to placeholder
                "alias": "variant_system_name_1"
            },
            "name": {
                "ru": "Название на русском",
                "en": "English name",
                "de": "Deutscher Name"
            },
            "color": "#FF5733"
        }
    ]

    - `owner` in `alias` will be automatically set to the attribute's `system_name` — you may omit it or set to any placeholder.
    - `color` must be a valid hex color, e.g. "#RRGGBB" or "#RGB".
    - At least one variant is required.
    - `ru` field in `name` is REQUIRED. `en` and `de` are optional.

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
        "type": "Enum",
        "format": display_format,
        "name": name,
        "description": description,
        "isTracked": write_changes_to_the_log,
        "isCalculated": calculate_value if expression_for_calculation != None else False,
        "expression": expression_for_calculation,
        "variants": [
            {
                "alias": variant.system_name.model_dump(),
                "name": variant.name.model_dump(),
                "color": variant.color
            }
            for variant in variants
        ]
    }

        # Remove None values
    request_body = _remove_nones(request_body) 

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

class GetEnumAttributeSchema(BaseModel):
    application_system_name: str = Field(
        description="System name of the application with the template where the attribute is located. RU: Системное имя приложения"
    )
    template_system_name: str = Field(
        description="System name of the template where the attribute is located. RU: Системное имя шаблона"
    )
    system_name: str = Field(
        description="Unique system name of the attribute to fetch. RU: Системное имя атрибута"
    )

    @field_validator("application_system_name", "template_system_name", "system_name", mode="before")
    @classmethod
    def non_empty(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("must be a non-empty string")
        return v


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

    keys_to_remove = ['isUnique', 'isTitle', 'isIndexed', 'isMultiValue', 'isMandatory', 'isOwnership', 'instanceGlobalAlias', 'imageColorType', 'imagePreserveAspectRatio']

    for key in keys_to_remove:
        if key in result_body['response']:
            result_body['response'].pop(key, None)

    result.update({"raw_response": result_body['response']})
    validated = AttributeResult(**result)
    return validated.model_dump()

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