"""Builders for conservative CMW form structures."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def ml_text(value: str) -> dict[str, str]:
    """Return multilingual text used by CMW form labels."""
    return {"en": value, "de": value, "ru": value}


def _label_model(value: str) -> dict[str, Any]:
    return {"text": ml_text(value), "hidden": False}


def _field_alias(form_system_name: str, attribute_system_name: str) -> str:
    return f"{form_system_name}_{attribute_system_name}"


def build_field_component(
    *,
    template_system_name: str,
    form_system_name: str,
    attribute_system_name: str,
    label: str,
    prototype: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a FieldComponent, optionally reusing an existing field as prototype."""
    field = deepcopy(prototype) if prototype else {}
    field["type"] = "FieldComponent"
    field.setdefault(
        "globalAlias",
        {
            "type": "FormComponent",
            "owner": form_system_name,
            "alias": _field_alias(form_system_name, attribute_system_name),
        },
    )
    field["propertyPath"] = [
        {
            "type": "Attribute",
            "owner": template_system_name,
            "alias": attribute_system_name,
        }
    ]
    field["label"] = _label_model(label)
    field.setdefault("accessType", "Editable")
    field.setdefault("width", 0)
    return field


def build_layout_row(components: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one row that contains form components."""
    return {"type": "LayoutModel", "components": components}


def build_group_panel(title: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a titled group panel with nested vertical layout."""
    return {
        "type": "PanelModel",
        "displayName": title,
        "label": _label_model(title),
        "layout": {
            "type": "VerticalLayout",
            "rows": rows,
        },
    }


def build_form_root(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the top-level form root."""
    return {
        "type": "VerticalLayout",
        "rows": groups,
    }


def build_form_body(
    *,
    template_system_name: str,
    form_system_name: str,
    name: str,
    root: dict[str, Any],
    is_default: bool = True,
) -> dict[str, Any]:
    """Build a full form body suitable for webapi/Form POST."""
    return {
        "globalAlias": {
            "type": "Form",
            "owner": template_system_name,
            "alias": form_system_name,
        },
        "container": {
            "type": "RecordTemplate",
            "alias": template_system_name,
        },
        "name": name,
        "type": "PublicForm",
        "formSize": "Common",
        "isDefault": is_default,
        "root": root,
    }


def build_root_from_widgets(
    *,
    template_system_name: str,
    form_system_name: str,
    widgets: dict[str, dict[str, Any]],
    group_title: str = "Main information",
) -> dict[str, Any]:
    """Build a simple visible root from desired widget fields."""
    fields = [
        build_field_component(
            template_system_name=template_system_name,
            form_system_name=form_system_name,
            attribute_system_name=alias,
            label=str(edits.get("label") or edits.get("name") or edits.get("text") or alias),
            prototype=edits.get("prototype") if isinstance(edits, dict) else None,
        )
        for alias, edits in widgets.items()
        if isinstance(edits, dict)
    ]
    return build_form_root([build_group_panel(group_title, [build_layout_row(fields)])])
