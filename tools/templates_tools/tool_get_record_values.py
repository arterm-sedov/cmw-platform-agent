"""LangChain tool: read attribute values for one record (GetPropertyValues).

The HTTP call is implemented in :func:`tools.platform_record_document.fetch_record_field_values`
so the same client is reused (e.g. document id resolution) and stays in one place.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from tools.platform_record_document import fetch_record_field_values


class GetRecordValuesSchema(BaseModel):
    record_id: str = Field(description="Record id to read.")
    attribute_system_names: list[str] = Field(
        min_length=1,
        description=(
            "At least one attribute system name. GetPropertyValues with an empty list does not "
            "return all fields on the platform (typically only id); list_attributes first if needed."
        ),
    )

    @field_validator("record_id", mode="before")
    @classmethod
    def strip_rid(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            msg = "record_id must be a non-empty string"
            raise ValueError(msg)
        return v.strip()


@tool(
    "get_record_values",
    return_direct=False,
    args_schema=GetRecordValuesSchema,
)
def get_record_values(
    record_id: str, attribute_system_names: list[str]
) -> dict[str, Any]:
    """
    Get current values for one or more attributes on a record (by system name). Use before fetch
    or attach, or whenever you need the live property values for a record.
    """
    return fetch_record_field_values(record_id, attribute_system_names)


__all__ = [
    "GetRecordValuesSchema",
    "get_record_values",
]
