from typing import Any, Dict, List, Optional, Literal
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.v1.types import NoneBytes
import tools.requests_ as requests_
from tools.models import AttributeResult

ATTRIBUTE_ENDPOINT = "webapi/"

class ListTemplates(BaseModel):
    application_system_name: str = Field(
        description="System name of the application to fetch the templates from. "
                    "RU: Системное имя приложения"
    )
    template_type: Literal["record", "process", "account"] = Field(
        default="record",
        description="Choose template type: Record, Process, or Account. "
                    "RU: Шаблон записи, Шаблон процесса, Шаблон аккаунта"
    )

    @field_validator("template_type", mode="before")
    @classmethod
    def normalize_operation(cls, v: str) -> str:
        if v is None:
            return v
        
        mapping = {
            "запись": "record",
            "процесс": "process",
            "аккаунт": "account"
        }
        
        normalized_value = str(v).strip().lower()
        return mapping.get(normalized_value, normalized_value)

    @field_validator("application_system_name", mode="before")
    @classmethod
    def non_empty_str(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("must be a non-empty string")
        return v

@tool("list_templates", return_direct=False, args_schema=ListTemplates)
def list_templates(
    application_system_name: str,
    template_type: str = "record"
) -> Dict[str, Any]:
    """
    List all templates of a given type in an application.
    
    Default template type is "record".
    
    Returns:
        dict: {
            "success": bool - True if operation completed successfully
            "status_code": int - HTTP response status code  
            "raw_response": dict|str|None - Raw response payload for auditing or payload body (sanitized)
            "error": str|None - Error message if operation failed
        }
    """

    try:
        if template_type == "record":
            result = tools.requests_._get_request(f"{ATTRIBUTE_ENDPOINT}/RecordTemplate/List/{application_system_name}")
        elif template_type == "process":
            result = tools.requests_._get_request(f"{ATTRIBUTE_ENDPOINT}/ProcessTemplate/List/{application_system_name}")
        elif template_type == "account":
            result = tools.requests_._get_request(f"{ATTRIBUTE_ENDPOINT}/AccountTemplate/List/{application_system_name}")
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
    results = list_templates.invoke({
        "application_system_name": "AItestAndApi",
        "template_type": "record"
    })
    print(results)