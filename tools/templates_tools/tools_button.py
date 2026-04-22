from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator

from tools import requests_
from tools.models import AttributeResult, CommonButtonFields
from tools.tool_utils import _fetch_entity, build_global_alias, execute_edit_or_create_operation

BUTTON_ENDPOINT = "webapi/UserCommand"


class GetButtonSchema(CommonButtonFields):
    pass


class EditOrCreateButtonSchema(CommonButtonFields):
    operation: str = Field(
        description="Choose operation: Create or Edit the button. RU: Создать, Редактировать",
    )
    name: str | None = Field(
        default=None,
        description="Human-readable display name of the button. Required for create.",
    )
    description: str | None = Field(
        default=None,
        description="Description of what the button does.",
)
    kind: str = Field(
        default="Trigger scenario",
        description="Button action: Trigger scenario (triggers scenario on click), Create, Edit, Delete, Archive, Unarchive, Test. Default: Trigger scenario",
    )
    context: str = Field(
        default="Record",
        description="Execution context: Record, List, etc. Default: Record",
    )
    multiplicity: str = Field(
        default="OneByOne",
        description="Multiplicity: OneByOne, Many, etc. Default: OneByOne",
    )
    result_type: str = Field(
        default="DataChange",
        description="Result type: DataChange, Redirect, etc. Default: DataChange",
    )
    is_prepare: bool = Field(
        default=False,
        description="Set to True for prepare mode (validates but doesn't execute).",
    )
    skip_validation: bool = Field(
        default=False,
        description="Set to True to skip validation before execution.",
    )
    has_confirmation: bool = Field(
        default=False,
        description="Set to True to show confirmation dialog before executing.",
    )
    is_confirmation_active: bool = Field(
        default=False,
        description="Whether confirmation is active for this button.",
    )
    navigation_target: str = Field(
        default="Undefined",
        description="Navigation target after execution: Undefined, SameForm, NewForm, etc.",
    )
    related_entity: str | None = Field(
        default=None,
        description="Related entity system name for buttons that operate on related records.",
    )

    @field_validator("operation", mode="before")
    @classmethod
    def _normalize_operation(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip().lower()
            mapping = {
                "создать": "create",
                "редактировать": "edit",
                "create": "create",
                "edit": "edit",
            }
            return mapping.get(v, v)
        return v

    @model_validator(mode="after")
    def _validate_create_required_fields(self) -> "EditOrCreateButtonSchema":
        if self.operation == "create" and (not self.name or not self.name.strip()):
            raise ValueError("name is REQUIRED when operation='create'")
        return self


def _fetch_button(
    application_system_name: str, template_system_name: str, button_system_name: str
) -> dict[str, Any] | None:
    """Fetch current button JSON using generic _fetch_entity."""
    return _fetch_entity(
        "UserCommand",
        application_system_name,
        template_system_name,
        button_system_name,
        BUTTON_ENDPOINT,
    )


@tool("get_button", return_direct=False, args_schema=GetButtonSchema)
def get_button(
    application_system_name: str,
    template_system_name: str,
    button_system_name: str,
) -> dict[str, Any]:
    r"""
    Fetch a button by system name.

    Returns:
        dict: {
            "success": bool - True if the button was fetched successfully
            "status_code": int - HTTP response status code
            "error": str|None - Error message if operation failed
        }
    """
    result = _fetch_button(application_system_name, template_system_name, button_system_name)
    if result is None:
        return {
            "success": False,
            "status_code": 404,
            "error": "Button not found",
        }
    return {"success": True, "status_code": 200, "error": None, "data": result}


@tool(
    "edit_or_create_button", return_direct=False, args_schema=EditOrCreateButtonSchema
)
def edit_or_create_button(
    operation: str,
    application_system_name: str,
    template_system_name: str,
    button_system_name: str,
    name: str | None = None,
    description: str | None = None,
    kind: str = "Trigger scenario",
    context: str = "Record",
    multiplicity: str = "OneByOne",
    result_type: str = "DataChange",
    is_prepare: bool = False,
    skip_validation: bool = False,
    has_confirmation: bool = False,
    is_confirmation_active: bool = False,
    navigation_target: str = "Undefined",
    related_entity: str | None = None,
) -> dict[str, Any]:
    r"""
    Create or edit a button for a template.

    For edit operations, automatically fetches current schema and merges missing fields.
    Editable button properties: name, description, kind, context, multiplicity,
    result_type, is_prepare, skip_validation, has_confirmation, navigation_target, related_entity.

    WARNING:
    - Changing options other than name and description is NOT DESIRABLE for most buttons
    - Default platform buttons (create, edit, archive, delete) have predefined behavior
    - Only name and description changes are recommended unless you specifically need other changes
    - Other parameters may be ignored or cause unexpected behavior

    Returns:
        dict: {
            "success": bool - True if the button was created or edited successfully
            "status_code": int - HTTP response status code
            "error": str|None - Error message if operation failed
        }
    """
    endpoint = f"{BUTTON_ENDPOINT}/{application_system_name}"

    request_body: dict[str, Any] = {
        "globalAlias": build_global_alias("UserCommand", template_system_name, button_system_name),
        "container": {
            "type": "RecordTemplate",
            "alias": template_system_name,
        },
        "context": context,
        "multiplicity": multiplicity,
        "kind": kind,
        "resultType": result_type,
        "isPrepare": is_prepare,
        "skipValidation": skip_validation,
        "isConfirmationActive": has_confirmation or is_confirmation_active,
        "navigationTarget": navigation_target,
    }

    if name is not None:
        request_body["name"] = name
    if description is not None:
        request_body["description"] = description

    if operation == "edit":
        current = _fetch_button(
            application_system_name, template_system_name, button_system_name
        )
        if current:
            if name is not None:
                current["name"] = name
            if description is not None:
                current["description"] = description
            if kind != "Trigger scenario":
                current["kind"] = kind
            if has_confirmation:
                current["isConfirmationActive"] = has_confirmation
            if context != "Record":
                current["context"] = context
            if multiplicity != "OneByOne":
                current["multiplicity"] = multiplicity
            if result_type != "DataChange":
                current["resultType"] = result_type
            if is_prepare:
                current["isPrepare"] = is_prepare
            if skip_validation:
                current["skipValidation"] = skip_validation
            if navigation_target != "Undefined":
                current["navigationTarget"] = navigation_target
            if related_entity:
                current["relatedEntityGlobalAlias"] = {
                    "type": "RecordTemplate",
                    "alias": related_entity,
                }
            request_body = current

        if (
            "globalAlias" not in request_body
            and "owner" in request_body
            and "alias" in request_body
        ):
            request_body["globalAlias"] = build_global_alias("UserCommand", request_body["owner"], request_body["alias"])

        return requests_._put_request(request_body, endpoint)

    if related_entity:
        request_body["relatedEntityGlobalAlias"] = {
            "type": "RecordTemplate",
            "alias": related_entity,
        }

    return execute_edit_or_create_operation(
        request_body=request_body,
        operation=operation,
        endpoint=endpoint,
        result_model=AttributeResult,
    )


class ListButtonsSchema(BaseModel):
    application_system_name: str = Field(
        description="System name of the application. RU: Системное имя приложения",
    )
    template_system_name: str = Field(
        description="System name of the template. RU: Системное имя шаблона",
    )

    @field_validator("application_system_name", "template_system_name", mode="before")
    def non_empty_str(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("must be a non-empty string")
        return v


@tool("list_buttons", return_direct=False, args_schema=ListButtonsSchema)
def list_buttons(
    application_system_name: str,
    template_system_name: str,
) -> dict[str, Any]:
    """
    List all buttons for a given template.

    Returns:
        dict: {
            "success": bool - True if button list was fetched successfully
            "status_code": int - HTTP response status code
            "error": str|None - Error message if operation failed
            "data": list|None - List of buttons if successful
        }
    """
    template_global_alias = f"Template@{application_system_name}.{template_system_name}"
    endpoint = f"{BUTTON_ENDPOINT}/List/{template_global_alias}"

    result = requests_._get_request(endpoint)

    if result.get("success"):
        raw_response = result.get("raw_response")
        if isinstance(raw_response, dict) and "response" in raw_response:
            response_data = raw_response["response"]
            if isinstance(response_data, list):
                result["data"] = response_data

    return result


class ArchiveUnarchiveButtonSchema(BaseModel):
    application_system_name: str = Field(
        description="System name of the application. RU: Системное имя приложения",
    )
    template_system_name: str = Field(
        description="System name of the template. RU: Системное имя шаблона",
    )
    button_system_name: str = Field(
        description="System name of the button. RU: Системное имя кнопки",
    )
    operation: str = Field(
        description="Choose operation: Archive or Unarchive the button. RU: Архивировать, Разархивировать",
    )

    @field_validator("operation", mode="before")
    @classmethod
    def _normalize_operation(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip().lower()
            mapping = {
                "архивировать": "archive",
                "разархивировать": "unarchive",
                "archive": "archive",
                "unarchive": "unarchive",
            }
            return mapping.get(v, v)
        return v


@tool(
    "archive_unarchive_button",
    return_direct=False,
    args_schema=ArchiveUnarchiveButtonSchema,
)
def archive_unarchive_button(
    application_system_name: str,
    template_system_name: str,
    button_system_name: str,
    operation: str,
) -> dict[str, Any]:
    """
    Archive or unarchive a button.

    Returns:
        dict: {
            "success": bool - True if the operation was successful
            "status_code": int - HTTP response status code
            "error": str|None - Error message if operation failed
        }
    """
    button_global_alias = f"UserCommand@{template_system_name}.{button_system_name}"

    if operation == "archive":
        endpoint = f"{BUTTON_ENDPOINT}/{application_system_name}/{button_global_alias}/Disable"
    elif operation == "unarchive":
        endpoint = f"{BUTTON_ENDPOINT}/{application_system_name}/{button_global_alias}/Enable"
    else:
        return {
            "success": False,
            "status_code": 400,
            "error": f"Invalid operation: {operation}. Use 'archive' or 'unarchive'.",
        }

    return requests_._put_request({}, endpoint)


if __name__ == "__main__":
    print("Button tools loaded")
