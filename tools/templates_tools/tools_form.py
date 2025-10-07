from ..tool_utils import *

FORM_ENDPOINT = "webapi/Form"


class GetFormSchema(BaseModel):
    application_system_name: str = Field(
        description="System name of the application. RU: Системное имя приложения",
    )
    template_system_name: str = Field(
        description="System name of the template. RU: Системное имя шаблона",
    )
    form_system_name: str = Field(
        description="System name of the form. RU: Системное имя формы",
    )

    @field_validator(
        "application_system_name",
        "template_system_name",
        "form_system_name",
        mode="before",
    )
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
        elif "Alias" in key and not "globalalias" in key.lower():
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


