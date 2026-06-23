"""Tool for creating a visible CMW form from template attributes."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tools.templates_tools.form_api import replace_form_delete_post
from tools.templates_tools.form_attribute_groups import infer_form_groups
from tools.templates_tools.form_builder import (
    build_field_component,
    build_form_body,
    build_form_root,
    build_group_panel,
    build_layout_row,
)
from tools.templates_tools.form_structure import count_field_components
from tools.templates_tools.tool_list_attributes import list_attributes


class CreateFormFromAttributesSchema(BaseModel):
    application_system_name: str = Field(description="Application system name.")
    template_system_name: str = Field(description="Template system name.")
    form_system_name: str = Field(default="defaultForm")
    form_name: str | None = Field(default=None)
    groups: list[dict[str, Any]] | None = Field(default=None)
    verify: bool = Field(default=True)


def build_form_from_attribute_groups(
    *,
    template_system_name: str,
    form_system_name: str,
    form_name: str,
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a form body from normalized attribute groups."""
    panels = []
    for group in groups:
        fields = [
            build_field_component(
                template_system_name=template_system_name,
                form_system_name=form_system_name,
                attribute_system_name=attribute["system_name"],
                label=attribute["name"],
            )
            for attribute in group.get("attributes", [])
        ]
        if fields:
            panels.append(build_group_panel(group.get("title", "Fields"), [build_layout_row(fields)]))
    root = build_form_root(panels)
    return build_form_body(
        template_system_name=template_system_name,
        form_system_name=form_system_name,
        name=form_name,
        root=root,
    )


@tool(
    "create_form_from_attributes",
    return_direct=False,
    args_schema=CreateFormFromAttributesSchema,
)
def create_form_from_attributes(
    application_system_name: str,
    template_system_name: str,
    form_system_name: str = "defaultForm",
    form_name: str | None = None,
    groups: list[dict[str, Any]] | None = None,
    verify: bool = True,
) -> dict[str, Any]:
    """Create a visible form containing all non-system template attributes."""
    attrs_result = list_attributes.invoke(
        {
            "application_system_name": application_system_name,
            "template_system_name": template_system_name,
        }
    )
    if not attrs_result.get("success"):
        return {
            "success": False,
            "status_code": int(attrs_result.get("status_code") or 0),
            "data": None,
            "error": attrs_result.get("error") or "Could not list attributes.",
        }

    raw_attributes = attrs_result.get("data") or []
    inferred_groups = groups or infer_form_groups(raw_attributes)
    included = [
        attribute
        for group in inferred_groups
        for attribute in group.get("attributes", [])
    ]
    body = build_form_from_attribute_groups(
        template_system_name=template_system_name,
        form_system_name=form_system_name,
        form_name=form_name or form_system_name,
        groups=inferred_groups,
    )
    result = replace_form_delete_post(
        app=application_system_name,
        template=template_system_name,
        form=form_system_name,
        new_form=body,
    )
    result["included_attributes"] = [attribute["system_name"] for attribute in included]
    result["excluded_count"] = max(len(raw_attributes) - len(included), 0)

    if verify and result.get("success"):
        field_count = count_field_components(result.get("data"))
        if field_count != len(included):
            result["success"] = False
            result["status_code"] = 422
            result["error"] = (
                "Saved form FieldComponent count does not match included attributes: "
                f"{field_count} != {len(included)}"
            )
    return result
