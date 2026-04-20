from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator

from tools import requests_
from tools.models import AttributeResult, CommonToolbarFields
from tools.tool_utils import (
    _apply_partial_update,
    _build_toolbar_items,
    _fetch_entity,
    build_global_alias,
    execute_edit_or_create_operation,
)

TOOLBAR_ENDPOINT = "webapi/Toolbar"


class GetToolbarSchema(CommonToolbarFields):
    pass


class ToolbarItemInputSchema(BaseModel):
    button_system_name: str = Field(
        description="System name of the button/user command to add. RU: Системное имя кнопки",
    )
    display_name: str | None = Field(
        default=None,
        description="Display name for the toolbar item. Defaults to button name if not specified.",
    )
    item_order: int = Field(
        default=0,
        description="Display order of the item in the toolbar.",
    )
    icon: str | None = Field(
        default=None,
        description="Icon name: Plus, Minus, Clone, Envelope, Pencil, Search, Ok, User, List, Remove, Trash, Cog, Star, Asterisk, Home, File, Time, Print, Download, Upload, Refresh, Repeat, Info, Warning, Exclamation, Cart, Play, Comment, Folder, Bell, Flash, Picture, Book",
    )
    color: str | None = Field(
        default=None,
        description="Color: Red, Orange, Yellow, Green, Blue, Purple, Pink, Cyan",
    )


class EditOrCreateToolbarSchema(CommonToolbarFields):
    operation: str = Field(
        description="Choose operation: Create or Edit the toolbar. RU: Создать, Редактировать",
    )
    name: str | None = Field(
        default=None,
        description="Human-readable name of the toolbar. Required for create, optional for edit.",
    )
    is_default_for_forms: bool = Field(
        default=False,
        description="Set to True to make this the default toolbar for forms.",
    )
    is_default_for_lists: bool = Field(
        default=False,
        description="Set to True to make this the default toolbar for lists.",
    )
    is_default_for_task_lists: bool = Field(
        default=False,
        description="Set to True to make this the default toolbar for task lists.",
    )
    items: list[ToolbarItemInputSchema] | None = Field(
        default=None,
        description="List of toolbar items (buttons) to add or update. Each item references a button by system name.",
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
    def _validate_create_required_fields(self) -> "EditOrCreateToolbarSchema":
        if self.operation == "create" and (not self.name or not self.name.strip()):
            raise ValueError("name is REQUIRED when operation='create'")
        return self


class ToolbarItemSchema(BaseModel):
    item_type: str = Field(
        description="Type of toolbar item: Action, Group, Splitter. Default: Action",
    )
    system_name: str = Field(
        description="System name of the item.",
    )
    name: str | None = Field(
        default=None,
        description="Display name of the toolbar item.",
    )
    icon: str | None = Field(
        default=None,
        description="Icon name: Plus, Minus, Clone, Envelope, Pencil, Search, Ok, User, List, Remove, Trash, Cog, Star, Asterisk, Home, File, Time, Print, Download, Upload, Refresh, Repeat, Info, Warning, Exclamation, Cart, Play, Comment, Folder, Bell, Flash, Picture, Book",
    )
    color: str | None = Field(
        default=None,
        description="Color for the button: Red, Orange, Yellow, Green, Blue, Purple, Pink, Cyan",
    )
    order: int = Field(
        default=0,
        description="Display order of the item in the toolbar.",
    )


class EditToolbarItemsSchema(CommonToolbarFields):
    items: list[ToolbarItemSchema] = Field(
        description="List of toolbar items to add or update.",
    )


def _fetch_toolbar(
    application_system_name: str, template_system_name: str, toolbar_system_name: str
) -> dict[str, Any] | None:
    """Fetch current toolbar JSON using generic _fetch_entity."""
    return _fetch_entity(
        "Toolbar",
        application_system_name,
        template_system_name,
        toolbar_system_name,
        TOOLBAR_ENDPOINT,
    )


@tool("get_toolbar", return_direct=False, args_schema=GetToolbarSchema)
def get_toolbar(
    application_system_name: str,
    template_system_name: str,
    toolbar_system_name: str,
) -> dict[str, Any]:
    r"""
    Fetch a toolbar by application, template and toolbar system name.

    Returns:
        dict: {
            "success": bool - True if the toolbar was fetched successfully
            "status_code": int - HTTP response status code
            "error": str|None - Error message if operation failed
        }
    """
    result = _fetch_entity(
        "Toolbar",
        application_system_name,
        template_system_name,
        toolbar_system_name,
        TOOLBAR_ENDPOINT,
    )
    if result is None:
        return {
            "success": False,
            "status_code": 404,
            "error": "Toolbar not found",
        }
    return {"success": True, "status_code": 200, "error": None, "data": result}


@tool(
    "edit_or_create_toolbar", return_direct=False, args_schema=EditOrCreateToolbarSchema
)
def edit_or_create_toolbar(
    operation: str,
    application_system_name: str,
    template_system_name: str,
    toolbar_system_name: str,
    name: str | None = None,
    is_default_for_forms: bool = False,
    is_default_for_lists: bool = False,
    is_default_for_task_lists: bool = False,
    items: list[ToolbarItemInputSchema] | None = None,
) -> dict[str, Any]:
    r"""
    Create or edit a toolbar for a template.

    For edit operations, automatically fetches current schema and merges missing fields.
    Items (buttons) can be added/updated by providing their button system names.

    Returns:
        dict: {
            "success": bool - True if the toolbar was created or edited successfully
            "status_code": int - HTTP response status code
            "error": str|None - Error message if operation failed
        }
    """
    endpoint = f"{TOOLBAR_ENDPOINT}/{application_system_name}"

    request_body: dict[str, Any] = {
        "globalAlias": build_global_alias("Toolbar", template_system_name, toolbar_system_name),
        "container": {
            "type": "RecordTemplate",
            "alias": template_system_name,
        },
    }

    if name is not None:
        request_body["name"] = name

    if operation == "edit":
        current_toolbar = _fetch_toolbar(
            application_system_name, template_system_name, toolbar_system_name
        )
        if current_toolbar:
            if name is not None:
                current_toolbar["name"] = name
            current_toolbar["IsDefaultForForms"] = is_default_for_forms
            current_toolbar["IsDefaultForLists"] = is_default_for_lists
            current_toolbar["IsDefaultForTaskLists"] = is_default_for_task_lists

            if items is not None:
                current_toolbar["items"] = _build_toolbar_items(items, template_system_name)

            request_body = current_toolbar

        if (
            "globalAlias" not in request_body
            and "owner" in request_body
            and "alias" in request_body
        ):
            request_body["globalAlias"] = build_global_alias("Toolbar", request_body["owner"], request_body["alias"])

        merged_body = _apply_partial_update(endpoint, request_body)
        return requests_._put_request(merged_body, endpoint)

    request_body["IsDefaultForForms"] = is_default_for_forms
    request_body["IsDefaultForLists"] = is_default_for_lists
    request_body["IsDefaultForTaskLists"] = is_default_for_task_lists

    if items is not None:
        request_body["items"] = _build_toolbar_items(items, template_system_name)

    return execute_edit_or_create_operation(
        request_body=request_body,
        operation=operation,
        endpoint=endpoint,
        result_model=AttributeResult,
    )


class ListToolbarsSchema(BaseModel):
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


@tool("list_toolbars", return_direct=False, args_schema=ListToolbarsSchema)
def list_toolbars(
    application_system_name: str,
    template_system_name: str,
) -> dict[str, Any]:
    """
    List all toolbars for a given template.

    Returns:
        dict: {
            "success": bool - True if toolbar list was fetched successfully
            "status_code": int - HTTP response status code
            "error": str|None - Error message if operation failed
            "data": list|None - List of toolbars if successful
        }
    """
    template_alias = f"Template@{application_system_name}.{template_system_name}"
    endpoint = f"{TOOLBAR_ENDPOINT}/List/{template_alias}"
    result = requests_._get_request(endpoint)

    if result.get("success"):
        raw_response = result.get("raw_response")
        if isinstance(raw_response, dict) and "response" in raw_response:
            response_data = raw_response["response"]
            if isinstance(response_data, list):
                result["data"] = response_data

    return result


if __name__ == "__main__":
    print("Toolbar tools loaded")
