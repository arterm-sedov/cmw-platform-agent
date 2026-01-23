from ..tool_utils import *

ATTRIBUTE_ENDPOINT = "webapi/Attribute/List"

class ListAttributes(BaseModel):
    application_system_name: str = Field(
        description="System name of the application with the template where the attributes are to be found."
            "RU: Системное имя приложения"
    )
    template_system_name: str = Field(
        description="System name of the template where the attributes are to be found."
            "RU: Системное имя шаблона"
    )

    @field_validator("application_system_name", "template_system_name", mode="before")
    @classmethod
    def non_empty_str(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("must be a non-empty string")
        return v

@tool("list_attributes", return_direct=False, args_schema=ListAttributes)
def list_attributes(
    application_system_name: str,
    template_system_name: str,
    ) -> Dict[str, Any]:
    """
    List all attributes in the given template and application.

    Returns:
        dict: {
            "success": bool - True if attribute list was fetched successfully
            "status_code": int - HTTP response status code  
            "data": list|None - List of attributes if successful
            "error": str|None - Error message if operation failed
        }
    """

    template_global_alias = f"Template@{application_system_name}.{template_system_name}"

    result = requests_._get_request(f"{ATTRIBUTE_ENDPOINT}/{template_global_alias}")

    return execute_list_operation(
            response_data = result,
            result_model=AttributeResult
    )

if __name__ == "__main__":
    results = list_attributes.invoke({
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test"
    })
    print(results)