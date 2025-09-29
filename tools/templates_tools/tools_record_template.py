from ..tool_utils import *


class EditOrCreateRecordTemplateSchema(BaseModel):
    operation: Literal["create", "edit"] = Field(
        description="Choose operation: Create or Edit the attribute. "
                    "RU: Создать, Редактировать"
    )
    application_system_name: str = Field(
        description="System name of the application where the template is created or edited. "
                    "RU: Системное имя приложения"
    )
    name: str = Field(
        description="Human-readable name of the template. "
                    "RU: Название"
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the template (auto-generate if omitted). "
                    "RU: Описание",
    )
    system_name: str = Field(
        description="System name of the template. "
                    "RU: Системное имя"
    )

    @field_validator("name", "system_name", mode="before")
    def non_empty_str(cls, v: Any) -> Any:
        """
        Validate that string fields are not empty.
        
        This field validator is automatically applied to the name, system_name, 
        application_system_name fields ensuring consistent validation.
        """
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("must be a non-empty string")
        return v

@tool("edit_or_create_record_template", return_direct=False, args_schema=EditOrCreateRecordTemplateSchema)
def edit_or_create_record_template(
    operation: str,
    name: str,
    system_name: str,
    application_system_name: str,
    description: Optional[str] = None
) -> Dict[str, Any]:
    r"""
    Edit or Create a record template.

    Returns:
        dict: {
            "success": bool - True if the template was created or edited successfully
            "status_code": int - HTTP response status code  
            "raw_response": dict|str|None - Raw response for auditing or payload body (sanitized)
            "error": str|None - Error message if operation failed
        }
    """

    request_body: Dict[str, Any] = {
        "name": name,
        "description": description,
        "globalAlias": {
            "type": "RecordTemplate",
            "alias": system_name
        }
    }

    endpoint = f"{RECORD_TEMPLATE_ENDPOINT}/{application_system_name}"

    return execute_edit_or_create_operation(
        request_body=request_body,
        operation=operation,
        endpoint=endpoint,
        result_model=TemplateResult
    )

if __name__ == "__main__":
    results = edit_or_create_record_template.invoke({
        "operation": "create",
        "name": "Sample Template",
        "system_name": "SampleTemplate",
        "application_system_name": "AItestAndApi",
        "description": "Created via CLI"
    })
    print(results)
