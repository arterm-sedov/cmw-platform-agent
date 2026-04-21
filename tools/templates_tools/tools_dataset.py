from typing import Any, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator

from tools import requests_
from tools.models import CommonDatasetFields, DatasetResult
from tools.tool_utils import (
    _apply_partial_update,
    _fetch_entity,
    build_global_alias,
    execute_edit_or_create_operation,
    execute_get_operation,
)

DATASET_ENDPOINT = "webapi/Dataset"


class GetDatasetSchema(CommonDatasetFields):
    pass


class EditOrCreateDatasetSchema(CommonDatasetFields):
    operation: str = Field(
        description="Choose operation: Create or Edit the dataset. RU: Создать, Редактировать",
    )
    name: Optional[str] = Field(
        default=None,
        description="Human-readable name of the dataset. Required for create, optional for edit.",
    )
    view_type: Optional[str] = Field(
        default=None,
        description="Dataset view type: Undefined, General, SplitVertical, SplitHorizontal. Optional for edit.",
    )
    is_default: Optional[bool] = Field(
        default=None,
        description="Set as default dataset for the template (true/false).",
    )
    toolbar_system_name: Optional[str] = Field(
        default=None,
        description="System name of toolbar to link to this dataset (e.g., 'newlistToolbar'). The toolbar must exist.",
    )
    show_disabled: Optional[bool] = Field(
        default=None,
        description="Show disabled records in dataset.",
    )
    columns: Optional[dict[str, Optional[dict[str, Any]]]] = Field(
        default=None,
        description=(
            "Column edits. Format: {columnAlias: {property: value}} or {columnAlias: null} to delete. "
            "Supported properties: "
            "name (display label), isHidden (true to hide column from UI), "
            "layout.flex, layout.width. "
            "To add new column: {newColumnAlias: {name: 'Label', propertyPath: [{type:'Attribute', owner:'Template', alias:'AttributeAlias'}]}}. "
            "To delete column: {columnAlias: null} or {columnAlias: {_delete: true}}. "
            "Example: {'Title': {'name': 'New Title'}, 'OldColumn': null, 'NewColumn': {'name': 'Added', 'propertyPath': [...]}}"
        ),
    )
    sorting: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description=(
            "Sorting rules. Format: [{propertyPath: [{type:'Attribute', owner:'Template', alias:'AttrAlias'}], direction: 'Asc'|'Desc', nullValuesOnTop: bool}]. "
            "Example: [{'propertyPath': [{'type': 'Attribute', 'owner': 'LegalEntity', 'alias': 'Title'}], 'direction': 'Asc', 'nullValuesOnTop': false}]"
        ),
    )
    grouping: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description=(
            "Grouping rules. Format: [{propertyPath: [...], name: 'GroupName', direction: 'Asc'|'Desc', level: 1, fields: [{propertyPath: [...], aggregationMethod: 'Count'|'Sum'|..., type: 'String'|'Number'|..., format: 'Undefined'}]}]. "
            "Example: [{'propertyPath': [{'type': 'Attribute', 'owner': 'LegalEntity', 'alias': 'MembersTest'}], 'name': 'Group', 'direction': 'Asc', 'level': 1, 'fields': [{'propertyPath': [{'type': 'Attribute', 'owner': 'System#', 'alias': 'isDisabled'}], 'aggregationMethod': 'Count', 'type': 'Boolean'}]}]"
        ),
    )
    totals: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description=(
            "Totals/summary rules. Format: [{propertyPath: [...], aggregationMethod: 'Count'|'Sum'|'Average'|..., type: 'String'|'Number'|...}]. "
            "Example: [{'propertyPath': [{'type': 'Attribute', 'owner': 'System#', 'alias': 'isDisabled'}], 'aggregationMethod': 'Count', 'type': 'Boolean'}]"
        ),
    )

    @field_validator("operation", mode="before")
    @classmethod
    def _normalize_operation(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip().lower()
            mapping = {"создать": "create", "редактировать": "edit", "create": "create", "edit": "edit"}
            return mapping.get(v, v)
        return v

    @model_validator(mode="after")
    def _validate_create_required_fields(self) -> "EditOrCreateDatasetSchema":
        if self.operation == "create":
            if not self.name or not self.name.strip():
                raise ValueError("name is REQUIRED when operation='create'")
        return self


def _apply_column_edits(
    dataset_data: dict[str, Any],
    column_edits: dict[str, Optional[dict[str, Any]]],
    template_system_name: str = "",
) -> dict[str, Any]:
    """
    Apply column edits to dataset.

    Args:
        dataset_data: Full dataset JSON structure
        column_edits: Dict of {columnAlias: {property: value}}
                    E.g., {"Title": {"name": "New Title"}, "isDisabled": {"name": "Status"}}
                    To add new column: {"NewCol": {"name": "Label", "propertyPath": [...]}}
                    To delete column: {"ColumnAlias": null} or {"ColumnAlias": {"_delete": true}}
        template_system_name: Template system name for auto-building propertyPath
    """
    if not column_edits:
        return dataset_data

    columns = dataset_data.get("columns", [])
    if not columns:
        columns = []
        dataset_data["columns"] = columns

    columns_to_remove = []

    for col_alias, edits in column_edits.items():
        if edits is None or (isinstance(edits, dict) and edits.get("_delete")):
            columns_to_remove.append(col_alias)
            continue

        found = False
        for col in columns:
            property_path = col.get("propertyPath", [{}])[0]
            if property_path.get("alias") == col_alias:
                found = True
                if isinstance(edits, dict):
                    for prop, value in edits.items():
                        if prop == "propertyPath":
                            col[prop] = value
                        elif prop == "_delete":
                            pass
                        else:
                            col[prop] = value
                break

        if not found and isinstance(edits, dict):
            if "propertyPath" in edits or "name" in edits:
                pp = edits.get("propertyPath", [])
                if not pp and col_alias:
                    pp = [{"type": "Attribute", "owner": template_system_name, "alias": col_alias}]
                new_col = {
                    "propertyPath": pp,
                    "name": edits.get("name", col_alias),
                    "layout": {"flex": 1.0, "width": 0.0},
                    "isHidden": edits.get("isHidden", False),
                }
                for prop, value in edits.items():
                    if prop not in ("propertyPath", "name", "layout", "isHidden", "_delete"):
                        new_col[prop] = value
                columns.append(new_col)

    if columns_to_remove:
        dataset_data["columns"] = [
            col for col in columns
            if col.get("propertyPath", [{}])[0].get("alias") not in columns_to_remove
        ]

    return dataset_data


def _fetch_dataset(
    application_system_name: str,
    template_system_name: str,
    dataset_system_name: str,
) -> dict[str, Any] | None:
    """Fetch current dataset JSON using generic _fetch_entity."""
    return _fetch_entity(
        "Dataset",
        application_system_name,
        template_system_name,
        dataset_system_name,
        DATASET_ENDPOINT,
    )


@tool("edit_or_create_dataset", return_direct=False, args_schema=EditOrCreateDatasetSchema)
def edit_or_create_dataset(
    operation: str,
    application_system_name: str,
    template_system_name: str,
    dataset_system_name: str,
    name: Optional[str] = None,
    view_type: Optional[str] = None,
    is_default: Optional[bool] = None,
    toolbar_system_name: Optional[str] = None,
    show_disabled: Optional[bool] = None,
    columns: Optional[dict[str, Optional[dict[str, Any]]]] = None,
    sorting: Optional[list[dict[str, Any]]] = None,
    grouping: Optional[list[dict[str, Any]]] = None,
    totals: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    r"""
    Create or edit a dataset for a template.

    Supports full editing:
    - Dataset metadata: name, viewType, isDefault, showDisabled
    - Link toolbar: toolbar_system_name (must exist first)
    - Sorting: sorting field with propertyPath and direction
    - Grouping: grouping field with propertyPath, fields, level
    - Totals: totals field with aggregationMethod
    - Add/remove/rename columns

    For edit operations, automatically fetches current schema and merges missing fields.

    Returns:
        dict: {
            "success": bool - True if the dataset was created or edited successfully
            "status_code": int - HTTP response status code
            "error": str|None - Error message if operation failed
        }
    """
    endpoint = f"{DATASET_ENDPOINT}/{application_system_name}"

    request_body: dict[str, Any] = {
        "globalAlias": build_global_alias("Dataset", template_system_name, dataset_system_name),
        "container": {
            "type": "RecordTemplate",
            "alias": template_system_name,
        },
    }

    if name is not None:
        request_body["name"] = name
    if view_type is not None:
        request_body["viewType"] = view_type

    if operation == "edit":
        # Always fetch full dataset to preserve all related data
        current_dataset = _fetch_dataset(
            application_system_name, template_system_name, dataset_system_name
        )
        if not current_dataset:
            return {
                "success": False,
                "status_code": 404,
                "error": "Could not fetch current dataset",
            }

        # Update name if provided
        if name is not None:
            current_dataset["name"] = name
        if view_type is not None:
            current_dataset["viewType"] = view_type
        if is_default is not None:
            current_dataset["isDefault"] = is_default
        if show_disabled is not None:
            current_dataset["showDisabled"] = show_disabled
        if toolbar_system_name is not None:
            current_dataset["toolbar"] = {
                "type": "Toolbar",
                "owner": template_system_name,
                "alias": toolbar_system_name,
            }
        if sorting is not None:
            current_dataset["sorting"] = sorting
        if grouping is not None:
            current_dataset["grouping"] = grouping
        if totals is not None:
            current_dataset["totals"] = totals

        # Apply column edits if provided
        if columns:
            current_dataset = _apply_column_edits(current_dataset, columns, template_system_name)

        request_body = current_dataset

        # Ensure globalAlias is present
        if "globalAlias" not in request_body:
            request_body["globalAlias"] = build_global_alias("Dataset", template_system_name, dataset_system_name)

        merged_body = _apply_partial_update(endpoint, request_body)
        return requests_._put_request(merged_body, endpoint)

    if columns:
        request_body["columns"] = []
        request_body = _apply_column_edits(request_body, columns, template_system_name)
    if is_default is not None:
        request_body["isDefault"] = is_default
    if show_disabled is not None:
        request_body["showDisabled"] = show_disabled
    if sorting is not None:
        request_body["sorting"] = sorting
    if grouping is not None:
        request_body["grouping"] = grouping
    if totals is not None:
        request_body["totals"] = totals

    return execute_edit_or_create_operation(
        request_body=request_body,
        operation=operation,
        endpoint=endpoint,
        result_model=DatasetResult,
    )


class ListDatasetsSchema(BaseModel):
    application_system_name: str = Field(
        description=(
            "System name of the application with the template where the datasets are "
            "to be found. RU: Системное имя приложения"
        )
    )
    template_system_name: str = Field(
        description=(
            "System name of the template where the datasets are to be found. "
            "RU: Системное имя шаблона"
        )
    )

    @field_validator("application_system_name", "template_system_name", mode="before")
    @classmethod
    def non_empty_str(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            msg = "must be a non-empty string"
            raise ValueError(msg)
        return v


@tool(
    "get_dataset",
    return_direct=False,
    args_schema=GetDatasetSchema,
)
def get_dataset(
    application_system_name: str,
    template_system_name: str,
    dataset_system_name: str,
) -> dict[str, Any]:
    r"""
    Fetch a dataset model for a given template and application.

        Returns:
        dict: {
            "success": bool - True if the dataset was fetched successfully
            "status_code": int - HTTP response status code
            "error": str|None - Error message if operation failed
            "data": dict - Dataset model (contains name, columns, viewType, etc.)
        }
    """
    result = _fetch_entity(
        "Dataset",
        application_system_name,
        template_system_name,
        dataset_system_name,
        DATASET_ENDPOINT,
    )
    if result is None:
        return {
            "success": False,
            "status_code": 404,
            "error": "Dataset not found",
        }
    return {"success": True, "status_code": 200, "error": None, "data": result}


@tool(
    "list_datasets",
    return_direct=False,
    args_schema=ListDatasetsSchema,
)
def list_datasets(
    application_system_name: str,
    template_system_name: str,
) -> dict[str, Any]:
    r"""
    List all datasets for a given template.

        Returns:
        dict: {
            "success": bool - True if succeeded
            "status_code": int - HTTP response status code
            "error": str|None - Error message if operation failed
            "data": list - List of datasets
        }
    """
    template_global_alias = f"Template@{application_system_name}.{template_system_name}"
    endpoint = f"{DATASET_ENDPOINT}/List/{template_global_alias}"
    # Use GET for list endpoint
    result = requests_._get_request(endpoint)

    if not result.get("success"):
        return result

    raw = result.get("raw_response", {})
    return {
        "success": True,
        "status_code": result.get("status_code", 200),
        "data": raw.get("response", []),
    }