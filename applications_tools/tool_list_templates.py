from typing import Any, Dict, List, Optional, Literal
from langchain.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.v1.types import NoneBytes
import requests_

ATTRIBUTE_ENDPOINT = "webapi/"

class ListTemplates(BaseModel):
    template_type: Literal["record", "process", "account"] = Field(
        description=(
            "Choose type of template: Record, Process or Account. Russian names allowed: "
            "['Запись', 'Процесс', 'Аккаунт']"
        )
    )
    application_system_name: str = Field(
        description=(
            "System name of the application where the template is created. "
            "Рус: 'Системное имя приложения'"
        )
    )

    @field_validator("template_type", mode="before")
    @classmethod
    def normalize_operation(cls, v: str) -> str:
        if v is None:
            return v
        value = str(v).strip().lower()
        mapping = {
            "запись": "record",
            "процесс": "process",
            "аккаунт": "account"
        }
        return mapping.get(value, value)

    @field_validator("application_system_name", mode="before")
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

@tool("list_templates", return_direct=False, args_schema=ListTemplates)
def list_templates(
    template_type: str,
    application_system_name: str
    ) -> Dict[str, Any]:
    """
    List all template by its `template_type` within a given `application_system_name`.

    Returns (AttributeResult):
    - success (bool): True if templates list was fetched successfully
    - status_code (int): HTTP response status code
    - raw_response (object|null): Template payload; sanitized (some keys may be removed)
    - error (string|null): Error message if any
    """

    try:
        if template_type == "record":
            result = requests_._get_request(f"{ATTRIBUTE_ENDPOINT}/AccountTemplate/List/{application_system_name}")
        if template_type == "process":
            result = requests_._get_request(f"{ATTRIBUTE_ENDPOINT}/ProcessTemplate/List/{application_system_name}")
        if template_type == "account":
            result = requests_._get_request(f"{ATTRIBUTE_ENDPOINT}/RecordTemplate/List/{application_system_name}")
        else:
            result = {
                "success": False,
                "error": f"No such type of template: {template_type}. Available types: record, process, account",
                "status_code": 400
            }
    except Exception as e:
        result = {
            "success": False,
            "error": f"Tool execution failed: {str(e)}",
            "status_code": 500
        }

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

    result.update({"raw_response": result_body['response']})
    validated = AttributeResult(**result)
    return validated.model_dump()

if __name__ == "__main__":
    results = list_attributes.invoke({
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test"
    })
    print(results)