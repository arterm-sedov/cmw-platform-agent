from ..tool_utils import *

from datetime import datetime

APPLICATION_ENDPOINT = "webapi/Solution"


class CreateApplicationSchema(BaseModel):
    """
    Schema for creating a new business application.
    """
    system_name: str = Field(description="Business application system name.")
    display_name: str = Field(description="Business application display name.")
    description: Optional[str] = Field(default=None, description="Business application description.")
    @field_validator("system_name", "display_name", mode="before")
    @classmethod
    def non_empty_str(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("Value must be a non-empty string")
        return v

@tool("create_app", return_direct=False, args_schema=CreateApplicationSchema)
def create_app(
    system_name: str,
    display_name: str,
    description: str = None,
) -> Dict[str, Any]:
    """
    Create a new application by system name.
    
    Creating the app takes time. So if the result is "timeout",
    list the apps to verify the app was created.

    Returns:
        dict: {
            "success": bool - True if application was created successfully,
            "status_code": int - HTTP response status code,
            "error": str|None - Error message if operation failed
        }
    """

    endpoint = f"{APPLICATION_ENDPOINT}"

    request_body = {
        "alias": system_name,
        "name": display_name,
        "description": description,
    }

    return execute_edit_or_create_operation(
        request_body=request_body,
        operation="create",
        endpoint=endpoint,
        result_model=AttributeResult,
    )


if __name__ == "__main__":
    results = create_app.invoke({
        "system_name": f"demo_app_alias_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "display_name": "Demo App",
        "description": "Created by CMW Copilot",
    })
    print(results)


