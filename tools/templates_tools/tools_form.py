from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from tools import requests_
from tools.models import AttributeResult, CommonFormFields
from tools.tool_utils import execute_get_operation, execute_list_operation

FORM_ENDPOINT = "webapi/Form"


class GetFormSchema(CommonFormFields):
    pass


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

    result = requests_._get_request(endpoint)  # noqa: SLF001

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
                        if ("globalAlias" in form and
                            isinstance(form["globalAlias"], dict)):
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

    return execute_list_operation(
        response_data=result,
        result_model=AttributeResult
    )


if __name__ == "__main__":
    get_form_results = get_form.invoke({
        "application_system_name": "Велестест",
        "template_system_name": "Prichina_otkaza",
        "form_system_name": "defaultForm"
    })
    print(get_form_results)

    # Test list_forms
    forms_list_results = list_forms.invoke({
        "application_system_name": "Велестест",
        "template_system_name": "Prichina_otkaza"
    })
    print("Forms:", forms_list_results)


