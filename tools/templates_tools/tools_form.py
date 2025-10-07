from ..tool_utils import *
from ..models import CommonFormFields

FORM_ENDPOINT = "webapi/Form"


class GetFormSchema(CommonFormFields):
    pass


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
    Fetch a form model.

        Returns:
        dict: {
            "success": bool - True if the template was created or edited successfully
            "status_code": int - HTTP response status code
            "error": str|None - Error message if operation failed
            form model fields: dict|list|None — Form structure without wrapper key (normalized)
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
    if not isinstance(data, dict):
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


if __name__ == "__main__":
    results = get_form.invoke({
        "application_system_name": "Велестест",
        "template_system_name": "Prichina_otkaza",
        "form_system_name": "defaultForm"
    })
    print(results)


