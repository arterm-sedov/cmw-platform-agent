from typing import Any, Dict, List, Optional, Literal
from langchain.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator
import requests_

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

class EditOrCreateAccountAttributeSchema(BaseModel):
    operation: Literal["create", "edit"] = Field(
        description=(
            "Choose operation: Creates or Edits the attribute. Russian names allowed: "
            "['Создать', 'Редактировать']"
        )
    )
    name: str = Field(description="Human-readable name of the attribute. Рус: 'Название'")
    system_name: str = Field(
        description="Unique system name of the attribute. Рус: 'Системное имя'"
    )
    application_system_name: str = Field(
        description=(
            "System name of the application with the template where the attribute is created. "
            "Рус: 'Системное имя приложения'"
        )
    )
    template_system_name: str = Field(
        description=(
            "System name of the template where the attribute is created. Рус: 'Системное имя шаблона'"
        )
    )
    description: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable description of the attribute (auto-generate if omitted). Рус: 'Описание'"
        ),
    )
    use_as_record_title: bool = Field(
        default=False,
        description=(
            "Whether attribute values will be displayed as a template record title. Рус: 'Использовать как заголовок записей'"
        ),
    )
    write_changes_to_the_log: bool = Field(
        default=False,
        description=(
            "Whether attribute changes should be logged. Рус: 'Записывать изменения в журнал'"
        ),
    )
    calculate_value: bool = Field(
        default=False,
        description=(
            "Whether attribute value should be calculated automatically; relevant only when expression_for_calculation is provided. Рус: 'Вычислять автоматически'"
        ),
    )
    expression_for_calculation: Optional[str] = Field(
        default=None,
        description=(
            "Expression for automatically calculating attribute value; user-provided. Рус: 'Выражение для вычисления'"
        ),
    )
    related_template_system_name: str = Field(
        default="_Account",
        description=(
            "System name of the template to associate withe attribute. Рус: 'Связанный шаблон'"
        )
    )
    store_multiple_values: bool = Field(
        default=False,
        description=(
            "whether attribute should store multiple values or single values. Рус: 'Хранить несколько значений'"
        )
    )

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


class AttributeResult(BaseModel):
    success: bool
    status_code: int
    raw_response: dict | str | None = Field(default=None, description="Raw response for auditing or payload body")
    error: Optional[str] = Field(default=None)


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
    r"""
    Edit or Create a account attribute.

    - Strictly follow argument schema and its built-in descriptions.

    Returns (AttributeResult):
    - success (bool): True if the operation completed successfully
    - status_code (int): HTTP response status code
    - raw_response (object|string|null): Raw server response or payload used for the request
    - error (string|null): Error message if any
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

class GetAccountAttributeSchema(BaseModel):
    application_system_name: str = Field(
        description=(
            "System name of the application with the template where the attribute is created. "
            "Рус: 'Системное имя приложения'"
        )
    )
    template_system_name: str = Field(
        description=(
            "System name of the template where the attribute is created. Рус: 'Системное имя шаблона'"
        )
    )
    system_name: str = Field(
        description="Unique system name of the attribute. Рус: 'Системное имя'"
    )

    @field_validator("application_system_name", "template_system_name", "system_name", mode="before")
    @classmethod
    def non_empty(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("must be a non-empty string")
        return v


@tool("get_account_attribute", return_direct=False, args_schema=GetAccountAttributeSchema)
def get_text_attribute(
    application_system_name: str,
    template_system_name: str,
    system_name: str
    ) -> Dict[str, Any]:
    """
    Get a account attribute by its `system_name` within a given `template_system_name` and `application_system_name`.

    Returns (AttributeResult):
    - success (bool): True if attribute was fetched successfully
    - status_code (int): HTTP response status code
    - raw_response (object|null): Attribute payload; sanitized (some keys may be removed)
    - error (string|null): Error message if any
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