from contextlib import suppress
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from tools import requests_
from tools.models import AttributeResult
from tools.tool_utils import execute_list_operation, remove_values

RECORDS_ENDPOINT_PREFIX = "webapi/Records"


class ListTemplateRecordsSchema(BaseModel):
    application_system_name: str = Field(
        description=(
            "System name of the application with the template. "
            "RU: Системное имя приложения"
        )
    )
    template_system_name: str = Field(
        description=(
            "System name of the template to fetch records from. "
            "RU: Системное имя шаблона"
        )
    )
    attributes: list[str] | None = Field(
        default=None,
        description=(
            "List of attribute system names to return. If omitted, returns all "
            "template attributes."
        ),
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional mapping attribute->value to filter dataset. "
            "Keep filters narrow; prefer pagination over large scans. "
        ),
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=100,
        description=(
            "Number of records to fetch (page size). Default 100. Max 100 per request. "
        ),
    )
    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Position to start fetching from (use for paging). Example: first "
            "page offset=0, second page offset=10."
        ),
    )
    sort_by: str | None = Field(
        default="creationDate",
        description="Attribute system name to sort by.",
    )
    sort_desc: bool = Field(
        default=False,
        description=(
            "Sort in descending order if True, ascending if False. "
        ),
    )

    @field_validator("application_system_name", "template_system_name", mode="before")
    @classmethod
    def non_empty_str(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            msg = "must be a non-empty string"
            raise ValueError(msg)
        return v


@tool(
    "list_template_records",
    return_direct=False,
    args_schema=ListTemplateRecordsSchema,
)
def list_template_records(
    application_system_name: str,
    template_system_name: str,
    *,
    attributes: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    limit: int = 100,
    offset: int = 0,
    sort_by: str | None = "creationDate",
    sort_desc: bool = False,
) -> dict[str, Any]:
    r"""
    List records for a given template with filtering, attribute projection,
    sorting, and pagination.

    - Before calling this tool fetch the template attribute model using the `list_attributes` tool
    and use it to determine the available attributes properties.
    - Filtering: exact match by keys in `filters`; list values are supported (contains/all-of).
    - Attributes: when a non-empty `attributes` list is provided, only those attributes are returned.
    - Pagination/sorting: hard limit 100 per call (page via `offset`).
    - After each iteration talk to the user about the results instead of fetching too many records at once.

    Returns:
        dict: {
            "success": bool - True if attribute list was fetched successfully
            "status_code": int - HTTP response status code  
            "data": list|None - List of attributes if successful
            "error": str|None - Error message if operation failed
        }
    """

    # Build Template@solution.template path parameter
    template_global_alias_str = f"Template@{application_system_name}.{template_system_name}"
    endpoint = f"{RECORDS_ENDPOINT_PREFIX}/{template_global_alias_str}"

    result = requests_._get_request(endpoint)

    # If the API call failed or structure is unexpected, normalize via common adapter
    if not result.get("success", False):
        return execute_list_operation(
            response_data=result,
            result_model=AttributeResult,
        )

    raw = result.get("raw_response")
    if not isinstance(raw, dict) or "response" not in raw:
        return execute_list_operation(
            response_data=result,
            result_model=AttributeResult,
        )

    # Normalize response to a list of records
    response_payload = raw.get("response")
    if isinstance(response_payload, dict):
        data = list(response_payload.values())
    elif isinstance(response_payload, list):
        data = response_payload
    else:
        data = []

    # Client-side filtering (exact match semantics with light list support)
    if isinstance(filters, dict) and filters:
        def _matches(rec: dict[str, Any]) -> bool:
            for key, expected in filters.items():
                actual = (rec or {}).get(key, None)
                # List-aware comparison
                if isinstance(actual, list):
                    if isinstance(expected, list):
                        # Require all expected items to be present
                        if not set(expected).issubset(set(actual)):
                            return False
                    else:
                        if expected not in actual:
                            return False
                else:
                    if actual != expected:
                        return False
            return True

        data = [r for r in data if isinstance(r, dict) and _matches(r)]

    # Client-side attribute projection (None or empty list => return all fields)
    if attributes not in (None, []) and isinstance(attributes, list):
        attr_set = set(attributes)
        projected: list[dict[str, Any]] = []
        for rec in data:
            if not isinstance(rec, dict):
                continue
            projected.append({k: v for k, v in rec.items() if k in attr_set})
        data = projected

    # Optional client-side sorting
    if sort_by:
        with suppress(Exception):
            data = sorted(
                data,
                key=lambda r: (r or {}).get(sort_by),
                reverse=bool(sort_desc),
            )

    # Client-side pagination — enforce hard ceiling of 100 per request
    start = max(0, int(offset))
    page_size = max(1, min(int(limit), 100))
    end = max(start, start + page_size)
    paged = data[start:end]

    adapted = dict(result)
    adapted["raw_response"] = {"response": paged}

    return execute_list_operation(
        response_data=adapted,
        result_model=AttributeResult,
    )


if __name__ == "__main__":
    results = list_template_records.invoke({
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test",
        "attributes": [],
        "filters": {},
        "limit": 10,
        "offset": 0,
        "sort_by": None,
        "sort_desc": False,
    })
    print(results)



