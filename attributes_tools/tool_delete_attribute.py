from typing import Any, Dict, List, Optional, Literal
from langchain.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.v1.types import NoneBytes
import requests_

ATTRIBUTE_ENDPOINT = "webapi/Attribute"

class DeleteAttribute(BaseModel):
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

    @field_validator("system_name", "application_system_name", "template_system_name", mode="before")
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

@tool("delete_attribute", return_direct=False, args_schema=DeleteAttribute)
def delete_attribute(
    application_system_name: str,
    template_system_name: str,
    system_name: str
    ) -> Dict[str, Any]:
    """
    Delete a text attribute by its `system_name` within a given `template_system_name` and `application_system_name`.

    Returns (AttributeResult):
    - success (bool): True if the operation completed successfully
    - status_code (int): HTTP response status code
    - raw_response (object|null): Attribute payload; sanitized (some keys may be removed)
    - error (string|null): Error message if any
    """

    attribute_global_alias = f"Attribute@{template_system_name}.{system_name}"

    result = requests_._delete_request(f"{ATTRIBUTE_ENDPOINT}/{application_system_name}/{attribute_global_alias}")

    validated = AttributeResult(**result)

    # Check if the request was successful and has the expected structure
    if not result.get('success', False):
        return validated.model_dump()

    return validated.model_dump()

if __name__ == "__main__":
    results = delete_attribute.invoke({
        "system_name": "Test2",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test"
    })
    print(results)