"""Tool for copying a working CMW form between templates."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from tools.templates_tools.form_api import get_form_raw, replace_form_delete_post
from tools.templates_tools.form_structure import (
    assert_no_stale_tokens,
    count_field_components,
)


class CopyFormFromTemplateSchema(BaseModel):
    source_application_system_name: str = Field(description="Source application system name.")
    source_template_system_name: str = Field(description="Source template system name.")
    target_application_system_name: str = Field(description="Target application system name.")
    target_template_system_name: str = Field(description="Target template system name.")
    source_form_system_name: str = Field(default="defaultForm")
    target_form_system_name: str = Field(default="defaultForm")
    replace_tokens: dict[str, str] | None = Field(default=None)
    verify: bool = Field(default=True)

    @field_validator("replace_tokens", mode="before")
    @classmethod
    def _parse_replace_tokens(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError) as err:
                msg = "replace_tokens must be a valid JSON object"
                raise ValueError(msg) from err
        return value


def _replace_tokens(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for source, target in replacements.items():
            value = value.replace(source, target)
        return value
    if isinstance(value, list):
        return [_replace_tokens(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_tokens(item, replacements) for key, item in value.items()}
    return value


def build_copied_form(
    *,
    source_form: dict[str, Any],
    source_application_system_name: str,
    source_template_system_name: str,
    target_template_system_name: str,
    target_form_system_name: str,
    replace_tokens: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a target form by copying and token-rewriting a source form."""
    replacements = {
        source_application_system_name: "",
        source_template_system_name: target_template_system_name,
    }
    if replace_tokens:
        replacements.update(replace_tokens)
    form = _replace_tokens(deepcopy(source_form), replacements)
    form["globalAlias"] = {
        "type": "Form",
        "owner": target_template_system_name,
        "alias": target_form_system_name,
    }
    form["container"] = {
        "type": "RecordTemplate",
        "alias": target_template_system_name,
    }
    form["isDefault"] = bool(form.get("isDefault", target_form_system_name == "defaultForm"))
    form.setdefault("type", "PublicForm")
    return form


@tool("copy_form_from_template", return_direct=False, args_schema=CopyFormFromTemplateSchema)
def copy_form_from_template(
    source_application_system_name: str,
    source_template_system_name: str,
    target_application_system_name: str,
    target_template_system_name: str,
    source_form_system_name: str = "defaultForm",
    target_form_system_name: str = "defaultForm",
    replace_tokens: dict[str, str] | None = None,
    verify: bool = True,
) -> dict[str, Any]:
    """Copy a visible form between templates and rewrite system-name tokens."""
    source = get_form_raw(
        source_application_system_name,
        source_template_system_name,
        source_form_system_name,
    )
    if not source:
        return {
            "success": False,
            "status_code": 404,
            "data": None,
            "error": "Could not fetch source form.",
        }
    if verify and count_field_components(source) == 0:
        return {
            "success": False,
            "status_code": 422,
            "data": source,
            "error": "Source form contains no FieldComponent nodes.",
        }

    target = build_copied_form(
        source_form=source,
        source_application_system_name=source_application_system_name,
        source_template_system_name=source_template_system_name,
        target_template_system_name=target_template_system_name,
        target_form_system_name=target_form_system_name,
        replace_tokens=replace_tokens,
    )
    if verify:
        stale_tokens = [source_template_system_name]
        if replace_tokens:
            stale_tokens.extend(replace_tokens)
        try:
            assert_no_stale_tokens(target, stale_tokens)
        except ValueError as err:
            return {
                "success": False,
                "status_code": 422,
                "data": target,
                "error": str(err),
            }

    result = replace_form_delete_post(
        app=target_application_system_name,
        template=target_template_system_name,
        form=target_form_system_name,
        new_form=target,
    )
    if verify and result.get("success") and count_field_components(result.get("data")) == 0:
        result["success"] = False
        result["status_code"] = 422
        result["error"] = "Form was saved but contains no FieldComponent nodes."
    return result
