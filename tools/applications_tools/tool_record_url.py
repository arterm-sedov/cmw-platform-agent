from typing import Any, Dict

from pydantic import BaseModel, Field, field_validator

from ..tool_utils import tool, requests_  # type: ignore


class GetRecordUrlSchema(BaseModel):
    """
    Schema for generating a direct URL to a specific record by its ID.
    """

    record_id: str = Field(description="Record ID to generate URL for")

    @field_validator("record_id", mode="before")
    @classmethod
    def non_empty_str(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("Value must be a non-empty string")
        return v


@tool("get_record_url", return_direct=False, args_schema=GetRecordUrlSchema)
def get_record_url(record_id: str) -> Dict[str, Any]:
    """
    Get the URL for a record by its ID.

    Returns:
        dict: {
            "success": bool,
            "status_code": int,
            "record_url": str|None,
            "error": str|None,
        }
    """

    try:
        cfg = requests_._load_server_config()
        base_url = cfg.base_url.rstrip("/")

        if not base_url:
            return {
                "success": False,
                "status_code": 500,
                "record_url": None,
                "error": "Base URL not found in server configuration",
            }

        record_url = f"{base_url}/#Resolver/{record_id}"

        return {
            "success": True,
            "status_code": 200,
            "record_url": record_url,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "status_code": 500,
            "record_url": None,
            "error": f"Error generating record URL: {str(e)}",
        }


if __name__ == "__main__":
    # Local manual test example
    results = get_record_url.invoke({
        "record_id": "1"
    })
    print(results)

