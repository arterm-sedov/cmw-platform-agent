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
    name: Optional[str] = Field(
        default=None,
        description="Human-readable name of the template. "
                    "RU: Название. Required for create, optional for edit."
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

    @field_validator("system_name", mode="before")
    @classmethod
    def non_empty_str(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("must be a non-empty string")
        return v

    @model_validator(mode="after")
    def _validate_create_required_fields(self) -> "EditOrCreateRecordTemplateSchema":
        if self.operation == "create":
            if not self.name or not self.name.strip():
                raise ValueError("name is REQUIRED when operation='create'")
        return self

@tool("edit_or_create_record_template", return_direct=False, args_schema=EditOrCreateRecordTemplateSchema)
def edit_or_create_record_template(
    operation: str,
    system_name: str,
    application_system_name: str,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    r"""
    Edit or Create a record template.

    Returns:
        dict: {
            "success": bool - True if the template was created or edited successfully
            "status_code": int - HTTP response status code
            "raw_response": dict|str|None - template model (sanitized)
            "error": str|None - Error message if operation failed
        }
    """

    request_body: Dict[str, Any] = {
        "globalAlias": {
            "type": "RecordTemplate",
            "alias": system_name
        }
    }
    if name is not None:
        request_body["name"] = name
    if description is not None:
        request_body["description"] = description

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
