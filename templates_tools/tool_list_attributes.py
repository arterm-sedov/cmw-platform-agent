from typing import Any, Dict, List, Optional, Literal
from langchain.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.v1.types import NoneBytes
import requests_

ATTRIBUTE_ENDPOINT = "webapi/Attribute/List"

class ListAttributes(BaseModel):
    application_system_name: str = Field(
        description=(
            "System name of the application with the template where the attributes is created. "
            "Рус: 'Системное имя приложения'"
        )
    )
    template_system_name: str = Field(
        description=(
            "System name of the template where the attributes is created. Рус: 'Системное имя шаблона'"
        )
    )

    @field_validator("application_system_name", "template_system_name", mode="before")
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

@tool("list_attributes", return_direct=False, args_schema=ListAttributes)
def list_attributes(
    application_system_name: str,
    template_system_name: str,
    ) -> Dict[str, Any]:
    """
    List all attributes by `template_system_name` within a given `application_system_name`.

    Returns (AttributeResult):
    - success (bool): True if attributes list was fetched successfully
    - status_code (int): HTTP response status code
    - raw_response (object|null): Attribute payload; sanitized (some keys may be removed)
    - error (string|null): Error message if any
    """

    template_global_alias = f"Template@{application_system_name}.{template_system_name}"

    result = requests_._get_request(f"{ATTRIBUTE_ENDPOINT}/{template_global_alias}")

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

    result.update({"raw_response": result_body['response']})
    validated = AttributeResult(**result)
    return validated.model_dump()

if __name__ == "__main__":
    results = list_attributes.invoke({
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test"
    })
    print(results)