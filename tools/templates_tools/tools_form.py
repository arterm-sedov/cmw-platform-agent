from copy import deepcopy
import json
from typing import Any, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator

from tools import requests_
from tools.models import AttributeResult, CommonFormFields, FormResult
from tools.templates_tools.form_api import get_form_raw, replace_form_delete_post
from tools.templates_tools.form_builder import (
    build_field_component,
    build_form_body,
    build_group_panel,
    build_layout_row,
    build_root_from_widgets,
    ml_text,
)
from tools.templates_tools.form_structure import (
    count_field_components,
    list_field_components,
)
from tools.tool_utils import (
    _apply_partial_update,
    _fetch_entity,
    build_global_alias,
    execute_edit_or_create_operation,
    execute_get_operation,
    execute_list_operation,
)

FORM_ENDPOINT = "webapi/Form"


class GetFormSchema(CommonFormFields):
    pass


class EditOrCreateFormSchema(CommonFormFields):
    operation: str = Field(
        description="Choose operation: Create or Edit the form. RU: Создать, Редактировать",
    )
    name: str | None = Field(
        default=None,
        description="Human-readable name of the form. Required for create, optional for edit.",
    )
    form_size: str | None = Field(
        default=None,
        description="Form size: Undefined, Small, Medium, Large, ExtraLarge. Optional for edit.",
    )
    widgets: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Desired form fields keyed by attribute/widget system name. Missing "
            "FieldComponent nodes are created/upserted."
        ),
    )
    create_from_attributes: bool = Field(
        default=False,
        description="Create a visible form from all non-system template attributes.",
    )
    copy_from: dict[str, Any] | None = Field(
        default=None,
        description="Copy source form config. Keys mirror copy_form_from_template.",
    )
    replace_root: dict[str, Any] | None = Field(
        default=None,
        description="Full replacement root tree for structural form edits.",
    )
    delete_existing: bool = Field(
        default=True,
        description="Use DELETE + POST for structural edits. Kept for tool schema clarity.",
    )
    verify: bool = Field(
        default=True,
        description="Verify saved form contains FieldComponent nodes after structural edits.",
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

    @field_validator("widgets", "copy_from", "replace_root", mode="before")
    @classmethod
    def _parse_json_object_fields(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError) as err:
                raise ValueError("value must be a valid JSON object") from err
        return v

    @model_validator(mode="after")
    def _validate_create_required_fields(self) -> "EditOrCreateFormSchema":
        if self.operation == "create" and (not self.name or not self.name.strip()):
            raise ValueError("name is REQUIRED when operation='create'")
        return self


def _apply_widget_edits(
    form_data: dict[str, Any], widget_edits: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """
    Recursively traverse form structure and apply widget edits.

    Args:
        form_data: Full form JSON structure
        widget_edits: Dict of {widgetAlias: {property: value}}
    """
    if not widget_edits:
        return form_data

    def traverse(obj: Any) -> Any:
        if isinstance(obj, dict):
            ga = obj.get("globalAlias", {})
            widget_alias = ga.get("alias")

            if widget_alias and widget_alias in widget_edits:
                edits = widget_edits[widget_alias]
                for prop, value in edits.items():
                    if prop == "label":
                        if "label" in obj:
                            obj["label"]["text"] = {
                                "en": value,
                                "de": value,
                                "ru": value,
                            }
                            obj["label"]["hidden"] = False
                        elif "text" in obj:
                            obj["text"] = {"en": value, "de": value, "ru": value}
                        else:
                            obj["label"] = {
                                "text": {"en": value, "de": value, "ru": value},
                                "hidden": False,
                            }
                        if "content" in obj:
                            obj["content"] = {"en": value, "de": value, "ru": value}
                    elif prop == "helpText":
                        obj["helpText"] = {"text": value}
                    elif prop == "placeholder":
                        obj["placeholder"] = {"text": value}
                    elif prop == "content":
                        obj["content"] = {"en": value, "de": value, "ru": value}
                    elif prop == "text":
                        obj["text"] = {"en": value, "de": value, "ru": value}
                    else:
                        obj[prop] = value

            for key, value in obj.items():
                obj[key] = traverse(value)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                obj[i] = traverse(item)

        return obj

    return traverse(form_data)


def _attribute_alias_from_field(field: dict[str, Any]) -> str:
    property_path = field.get("propertyPath")
    if isinstance(property_path, list):
        for item in property_path:
            if isinstance(item, dict) and item.get("alias"):
                return str(item["alias"])
    if isinstance(property_path, dict) and property_path.get("alias"):
        return str(property_path["alias"])
    global_alias = field.get("globalAlias")
    if isinstance(global_alias, dict) and global_alias.get("alias"):
        return str(global_alias["alias"])
    return ""


def _apply_common_widget_edits(field: dict[str, Any], edits: dict[str, Any]) -> None:
    for prop, value in edits.items():
        if prop in {"prototype", "_delete"}:
            continue
        if prop == "label":
            field["label"] = {"text": ml_text(str(value)), "hidden": False}
        elif prop in {"text", "content"}:
            field[prop] = ml_text(str(value))
        elif prop in {"helpText", "placeholder"}:
            field[prop] = {"text": str(value)}
        elif prop != "name":
            field[prop] = value


def _find_append_rows(root: dict[str, Any]) -> list[Any]:
    rows = root.setdefault("rows", [])
    if not rows:
        panel = build_group_panel("Main information", [build_layout_row([])])
        rows.append(panel)
        return panel["layout"]["rows"]

    for node in rows:
        if isinstance(node, dict):
            layout = node.get("layout")
            if isinstance(layout, dict):
                nested_rows = layout.setdefault("rows", [])
                if nested_rows:
                    return nested_rows
                nested_rows.append(build_layout_row([]))
                return nested_rows
    return rows


def _append_field_to_root(root: dict[str, Any], field: dict[str, Any]) -> None:
    rows = _find_append_rows(root)
    if rows and isinstance(rows[-1], dict):
        components = rows[-1].setdefault("components", [])
        if isinstance(components, list):
            components.append(field)
            return
    rows.append(build_layout_row([field]))


def _upsert_widgets(
    *,
    current_form: dict[str, Any] | None,
    widgets: dict[str, dict[str, Any]],
    template_system_name: str,
    form_system_name: str,
    form_name: str,
) -> dict[str, Any]:
    if current_form:
        form_body = deepcopy(current_form)
        form_body.setdefault(
            "globalAlias",
            build_global_alias("Form", template_system_name, form_system_name),
        )
        form_body.setdefault(
            "container",
            {"type": "RecordTemplate", "alias": template_system_name},
        )
        form_body.setdefault("name", form_name)
        form_body.setdefault("type", "PublicForm")
    else:
        return build_form_body(
            template_system_name=template_system_name,
            form_system_name=form_system_name,
            name=form_name,
            root=build_root_from_widgets(
                template_system_name=template_system_name,
                form_system_name=form_system_name,
                widgets=widgets,
            ),
        )

    root = form_body.get("root")
    if not isinstance(root, dict) or count_field_components(form_body) == 0:
        form_body["root"] = build_root_from_widgets(
            template_system_name=template_system_name,
            form_system_name=form_system_name,
            widgets=widgets,
        )
        return form_body

    fields_by_alias = {
        alias: field
        for field in list_field_components(form_body)
        if (alias := _attribute_alias_from_field(field))
    }
    for alias, edits in widgets.items():
        if not isinstance(edits, dict):
            continue
        label = str(
            edits.get("label") or edits.get("name") or edits.get("text") or alias
        )
        field = fields_by_alias.get(alias)
        if field:
            _apply_common_widget_edits(field, edits)
        else:
            new_field = build_field_component(
                template_system_name=template_system_name,
                form_system_name=form_system_name,
                attribute_system_name=alias,
                label=label,
                prototype=edits.get("prototype"),
            )
            _apply_common_widget_edits(new_field, edits)
            _append_field_to_root(root, new_field)
    return form_body


def _verify_structural_result(
    result: dict[str, Any],
    *,
    verify: bool,
) -> dict[str, Any]:
    if (
        verify
        and result.get("success")
        and count_field_components(result.get("data")) == 0
    ):
        result["success"] = False
        result["status_code"] = 422
        result["error"] = "Form was saved but contains no FieldComponent nodes."
    return result


def _fetch_form(
    application_system_name: str, template_system_name: str, form_system_name: str
) -> dict[str, Any] | None:
    """Fetch current form JSON using generic _fetch_entity."""
    return _fetch_entity(
        "Form",
        application_system_name,
        template_system_name,
        form_system_name,
        FORM_ENDPOINT,
    )


@tool("edit_or_create_form", return_direct=False, args_schema=EditOrCreateFormSchema)
def edit_or_create_form(
    operation: str,
    application_system_name: str,
    template_system_name: str,
    form_system_name: str,
    name: str | None = None,
    form_size: str | None = None,
    widgets: dict[str, dict[str, Any]] | None = None,
    create_from_attributes: bool = False,
    copy_from: dict[str, Any] | None = None,
    replace_root: dict[str, Any] | None = None,
    delete_existing: bool = True,
    verify: bool = True,
) -> dict[str, Any]:
    r"""
    Create or edit a form for a template.

    Supports:
    - Form metadata: name, formSize
    - Widget properties: label, helpText, placeholder, etc.

    For edit operations, automatically fetches current schema and merges missing fields.
    Widget editing: Provide widgets dict with widget aliases and their new property values.
    Example widgets={"TipRabot": {"label": "Work Type", "helpText": "Select work type"}}

    Returns:
        dict: {
            "success": bool - True if the form was created or edited successfully
            "status_code": int - HTTP response status code
            "error": str|None - Error message if operation failed
        }
    """
    if create_from_attributes:
        from tools.templates_tools.tool_create_form_from_attributes import (
            create_form_from_attributes,
        )

        return create_form_from_attributes.invoke(
            {
                "application_system_name": application_system_name,
                "template_system_name": template_system_name,
                "form_system_name": form_system_name,
                "form_name": name or form_system_name,
                "verify": verify,
            }
        )

    if copy_from:
        from tools.templates_tools.tool_copy_form import copy_form_from_template

        payload = {
            "source_application_system_name": copy_from.get(
                "source_application_system_name",
                copy_from.get("application_system_name", application_system_name),
            ),
            "source_template_system_name": copy_from.get("source_template_system_name"),
            "target_application_system_name": application_system_name,
            "target_template_system_name": template_system_name,
            "source_form_system_name": copy_from.get(
                "source_form_system_name", form_system_name
            ),
            "target_form_system_name": form_system_name,
            "replace_tokens": copy_from.get("replace_tokens"),
            "verify": verify,
        }
        if not payload["source_template_system_name"]:
            return {
                "success": False,
                "status_code": 422,
                "data": None,
                "error": "copy_from.source_template_system_name is required.",
            }
        return copy_form_from_template.invoke(payload)

    if replace_root is not None or widgets:
        current_form = get_form_raw(
            application_system_name,
            template_system_name,
            form_system_name,
        )
        if replace_root is not None:
            form_body = current_form or build_form_body(
                template_system_name=template_system_name,
                form_system_name=form_system_name,
                name=name or form_system_name,
                root=replace_root,
            )
            form_body = deepcopy(form_body)
            form_body["root"] = replace_root
            form_body.setdefault(
                "globalAlias",
                build_global_alias("Form", template_system_name, form_system_name),
            )
            form_body.setdefault(
                "container",
                {"type": "RecordTemplate", "alias": template_system_name},
            )
            form_body.setdefault("type", "PublicForm")
            if name is not None:
                form_body["name"] = name
        else:
            form_body = _upsert_widgets(
                current_form=current_form,
                widgets=widgets or {},
                template_system_name=template_system_name,
                form_system_name=form_system_name,
                form_name=name or form_system_name,
            )
        if form_size is not None:
            form_body["formSize"] = form_size
        if not delete_existing:
            return {
                "success": False,
                "status_code": 422,
                "data": form_body,
                "error": "Structural form edits require delete_existing=True.",
            }
        result = replace_form_delete_post(
            app=application_system_name,
            template=template_system_name,
            form=form_system_name,
            new_form=form_body,
        )
        return _verify_structural_result(result, verify=verify)

    endpoint = f"{FORM_ENDPOINT}/{application_system_name}"

    request_body: dict[str, Any] = {
        "globalAlias": build_global_alias(
            "Form", template_system_name, form_system_name
        ),
    }

    if name is not None:
        request_body["name"] = name
    if form_size is not None:
        request_body["formSize"] = form_size

    if operation == "edit":
        if widgets:
            current_form = _fetch_form(
                application_system_name, template_system_name, form_system_name
            )
            if not current_form:
                return {
                    "success": False,
                    "status_code": 404,
                    "error": "Could not fetch current form",
                }

            current_form = _apply_widget_edits(current_form, widgets)

            if name is not None:
                current_form["name"] = name
            if form_size is not None:
                current_form["formSize"] = form_size

            request_body = current_form

        # Reconstruct globalAlias if missing (process_data strips it)
        if (
            "globalAlias" not in request_body
            and "owner" in request_body
            and "alias" in request_body
        ):
            request_body["globalAlias"] = {
                "type": "Form",
                "owner": request_body["owner"],
                "alias": request_body["alias"],
            }

        merged_body = _apply_partial_update(endpoint, request_body)
        return requests_._put_request(merged_body, endpoint)

    return execute_edit_or_create_operation(
        request_body=request_body,
        operation=operation,
        endpoint=endpoint,
        result_model=FormResult,
    )


class ListFormsSchema(BaseModel):
    application_system_name: str = Field(
        description=(
            "System name of the application with the template where the forms are "
            "to be found. RU: Системное имя приложения"
        )
    )
    template_system_name: str = Field(
        description=(
            "System name of the template where the forms are to be found. "
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
    "get_form",
    return_direct=False,
    args_schema=GetFormSchema,
)
def get_form(
    application_system_name: str,
    template_system_name: str,
    form_system_name: str,
) -> dict[str, Any]:
    r"""
    Fetch a form model for a given template and application.

        Returns:
        dict: {
            "success": bool - True if the template was created or edited successfully
            "status_code": int - HTTP response status code
            "error": str|None - Error message if operation failed
            "dict|None — Form model (normalized)
        }
    """
    form_global_alias = f"Form@{template_system_name}.{form_system_name}"
    endpoint = f"{FORM_ENDPOINT}/{application_system_name}/{form_global_alias}"
    result = execute_get_operation(AttributeResult, endpoint)

    # Lean normalization: translate API terms to platform terminology
    if result.get("success"):
        result = _normalize_form_terms(result)

    return result


def _normalize_form_terms(data: dict) -> dict:
    """
    Lean normalization of API terms to platform terminology.
    Translates 'alias' → 'systemName' and 'Property' → 'Attribute'
    while preserving 'globalAlias' structures.
    """
    if data is None or not isinstance(data, dict):
        return data

    normalized = {}
    for key, value in data.items():
        # Rename 'alias' to 'systemName' (but not 'globalAlias')
        if key == "alias":
            normalized["systemName"] = value
        elif "Alias" in key and "globalalias" not in key.lower():
            normalized[key.replace("Alias", "SystemName")] = value
        # Rename 'Property' to 'Attribute' in camelCase
        elif "Property" in key:
            normalized[key.replace("Property", "Attribute")] = value
        # Recursively process nested structures
        elif isinstance(value, dict):
            normalized[key] = _normalize_form_terms(value)
        elif isinstance(value, list):
            normalized[key] = [
                _normalize_form_terms(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            normalized[key] = value

    return normalized


@tool(
    "list_forms",
    return_direct=False,
    args_schema=ListFormsSchema,
)
def list_forms(
    application_system_name: str,
    template_system_name: str,
) -> dict[str, Any]:
    """
    List all forms for a given template and application.

    Returns:
        dict: {
            "success": bool - True if form list was fetched successfully
            "status_code": int - HTTP response status code
            "error": str|None - Error message if operation failed
            "data": list|None - List of forms if successful
        }
    """

    template_global_alias = f"Template@{application_system_name}.{template_system_name}"
    endpoint = f"{FORM_ENDPOINT}/List/{template_global_alias}"

    result = requests_._get_request(endpoint)

    # Apply form-specific normalization if the request was successful
    if result.get("success"):
        raw_response = result.get("raw_response")
        if isinstance(raw_response, dict) and "response" in raw_response:
            response_data = raw_response["response"]
            if isinstance(response_data, list):
                # Apply normalization to each form in the list
                normalized_forms = []
                for form in response_data:
                    if isinstance(form, dict):
                        # Extract systemName from globalAlias for forms
                        if "globalAlias" in form and isinstance(
                            form["globalAlias"], dict
                        ):
                            global_alias = form["globalAlias"]
                            if "alias" in global_alias:
                                form["systemName"] = global_alias["alias"]

                        # Apply full normalization
                        normalized_form = _normalize_form_terms(form)
                        normalized_forms.append(normalized_form)
                    else:
                        normalized_forms.append(form)

                # Update the result with normalized data
                result["raw_response"]["response"] = normalized_forms

    return execute_list_operation(response_data=result, result_model=AttributeResult)


if __name__ == "__main__":
    get_form_results = get_form.invoke(
        {
            "application_system_name": "Велестест",
            "template_system_name": "Prichina_otkaza",
            "form_system_name": "defaultForm",
        }
    )
    print(get_form_results)

    # Test list_forms
    forms_list_results = list_forms.invoke(
        {
            "application_system_name": "Велестест",
            "template_system_name": "Prichina_otkaza",
        }
    )
    print("Forms:", forms_list_results)
