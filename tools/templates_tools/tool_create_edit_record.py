from typing import Any  # noqa: I001
from datetime import datetime
from decimal import Decimal, InvalidOperation

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator

from tools import requests_

ATTRIBUTE_LIST_ENDPOINT = "webapi/Attribute/List"
RECORD_ENDPOINT = "webapi/Record"


class UpsertRecordSchema(BaseModel):
    operation: str = Field(
        description=(
            "Operation: 'create' or 'edit'. RU: Создать или Редактировать"
        )
    )
    application_system_name: str = Field(
        description="Application system name. RU: Системное имя приложения"
    )
    template_system_name: str = Field(
        description="Template system name. RU: Системное имя шаблона"
    )
    values: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Mapping of attribute system name to value. Unspecified attributes "
            "will be sent as empty string ''."
        ),
    )
    record_id: str | None = Field(
        default=None,
        description=(
            "Existing record identifier for edit operation. RU: Идентификатор "
            "записи при редактировании"
        ),
    )

    @field_validator(
        "operation", "application_system_name", "template_system_name", mode="before"
    )
    @classmethod
    def non_empty_str(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            msg = "must be a non-empty string"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_edit_requires_id(self) -> "UpsertRecordSchema":
        op = str(self.operation or "").strip().lower()
        if op in {"редактировать", "edit"} and not (
            self.record_id and str(self.record_id).strip()
        ):
            msg = "record_id is required when operation is 'edit'"
            raise ValueError(msg)
        # normalize RU → EN for operation
        mapping = {"создать": "create", "редактировать": "edit"}
        object.__setattr__(self, "operation", mapping.get(op, op))
        return self


def _extract_attribute_alias(item: dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    # Prefer raw structure from API
    gl = item.get("globalAlias")
    if isinstance(gl, dict):
        alias = gl.get("alias")
        if isinstance(alias, str):
            return alias
    # Fallbacks in case of different shapes
    alias = item.get("alias")
    if isinstance(alias, str):
        return alias
    return ""


def _extract_attribute_type(item: dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    t = item.get("type")
    if isinstance(t, str):
        return t
    gl = item.get("globalAlias")
    if isinstance(gl, dict):
        alias_type = gl.get("type")
        if isinstance(alias_type, str):
            return alias_type
    return ""


def _extract_is_multivalue(item: dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    mv = item.get("isMultiValue")
    return bool(mv)


def _coerce_scalar_value(attr_type: str, value: Any) -> Any:
    t = (attr_type or "").lower()
    # Leave empty string as-is
    if value == "":
        return value
    # String-like types → str
    stringish = {
        "string",
        "document",
        "image",
        "drawing",
        "record",
        "role",
        "account",
        "enum",
    }
    if t in stringish:
        return str(value) if value is not None else ""
    # Boolean
    if t == "boolean":
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in {"true", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "0", "no", "n", "off"}:
            return False
        return ""  # fall back to empty string if not recognizable
    # DateTime → ISO string
    if t == "datetime":
        if isinstance(value, datetime):
            return value.isoformat()
        # pass through string; the server will validate masks
        return str(value) if value is not None else ""
    # Decimal → numeric (prefer stringified original if parse fails)
    if t == "decimal":
        try:
            if isinstance(value, (int, float, Decimal)):
                return value
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return ""
    # Duration and other unknowns → string passthrough
    return str(value) if value is not None else ""


def _coerce_value(attr_type: str, *, is_multi: bool, value: Any) -> Any:
    if bool(is_multi):
        value_list = [value] if not isinstance(value, list) else value
        return [_coerce_scalar_value(attr_type, v) for v in value_list]
    return _coerce_scalar_value(attr_type, value)


@tool(
    "create_edit_record",
    return_direct=False,
    args_schema=UpsertRecordSchema,
)
def create_edit_record(
    operation: str,
    application_system_name: str,
    template_system_name: str,
    values: dict[str, Any],
    record_id: str | None = None,
) -> dict[str, Any]:
    r"""
    Create or edit a record for a template.

    - Sets the provided values.
    - Default values are controlled by the Comindware Platform.
    - Does not set system attributes. 
    - Sets "_color" attribute as it's user-editable even though it's a system attribute.
    - Fetches the template's attribute list to get type information for proper coercion.
    - Applies type-aware coercion to provided values based on attribute metadata.

    Returns:
        dict: {
            "success": bool - True if record was created or edited successfully
            "status_code": int - HTTP response status code
            "record_id": str|None - record id
            "error": str|None - Error message if operation failed
        }
    """

    template_global_alias = f"Template@{application_system_name}.{template_system_name}"

    # 1) Fetch attribute list to know all attribute system names
    attrs_resp = requests_._get_request(  # noqa: SLF001
        f"{ATTRIBUTE_LIST_ENDPOINT}/{template_global_alias}"
    )

    all_attr_names: list[str] = []
    attr_meta_by_alias: dict[str, dict[str, Any]] = {}
    if attrs_resp.get("success") and isinstance(attrs_resp.get("raw_response"), dict):
        response_payload = attrs_resp["raw_response"].get("response")
        if isinstance(response_payload, list):
            for item in response_payload:
                alias = _extract_attribute_alias(item)
                if alias:
                    all_attr_names.append(alias)
                    attr_meta_by_alias[alias] = {
                        "type": _extract_attribute_type(item),
                        "isMultiValue": _extract_is_multivalue(item),
                        "isSystem": item.get("isSystem", False),
                    }

    # 2) Build request body with only provided values, properly typed
    # Exclude system fields (except "color" which is user-editable)
    body: dict[str, Any] = {}
    if isinstance(values, dict):
        for key, val in values.items():
            # Skip system fields (except "_color" which is user-editable)
            meta = attr_meta_by_alias.get(key) or {}
            if meta.get("isSystem", False) and key != "_color":
                continue
                
            coerced = _coerce_value(
                meta.get("type", ""), is_multi=bool(meta.get("isMultiValue")), value=val
            )
            # Only add non-empty values
            if coerced is not None and coerced != "":
                body[key] = coerced


    # 3) Choose endpoint/method and execute without stripping empty strings
    op = str(operation or "").strip().lower()
    if op == "create":
        endpoint = f"{RECORD_ENDPOINT}/{template_global_alias}"
        result = requests_._post_request(body, endpoint)  # noqa: SLF001
        return _normalize_upsert_result(result, record_id)

    if op == "edit":
        endpoint = f"{RECORD_ENDPOINT}/{record_id}"
        result = requests_._put_request(body, endpoint)  # noqa: SLF001
        return _normalize_upsert_result(result, record_id)

    return {
        "success": False,
        "status_code": 400,
        "record_id": None,
        "error": f"Unsupported operation: {operation}. Use 'create' or 'edit'",
    }


def _normalize_upsert_result(result: dict[str, Any], provided_record_id: str | None = None) -> dict[str, Any]:
    """Normalize upsert result to standard format with record_id extraction."""
    if not result.get("success", False):
        return {
            "success": False,
            "status_code": result.get("status_code", 500),
            "record_id": None,
            "error": result.get("error"),
        }

    # Extract record_id from response
    record_id = None
    raw_response = result.get("raw_response")
    
    if isinstance(raw_response, str):
        # Direct string response (record ID)
        record_id = raw_response
    elif isinstance(raw_response, dict):
        # WebApiResponse[System.String] - record ID is in the "response" field
        record_id = raw_response.get("response")
    
    # For edit operations, if we can't extract from response, 
    # use the provided record_id that was used in the request
    if record_id is None and provided_record_id:
        record_id = provided_record_id

    return {
        "success": True,
        "status_code": result.get("status_code", 200),
        "record_id": record_id,
        "error": None,
    }


if __name__ == "__main__":
    # Example local smoke run (requires env/config)
    from datetime import datetime
    results = create_edit_record.invoke(
        {
            "operation": "create",
            "application_system_name": "AItestAndApi",
            "template_system_name": "Test",
            "values": {
                "Test1": f"ПРОВЕРКА1234567890. EDITED {datetime.now().isoformat()}",
            },
        }
    )
    print(results)
