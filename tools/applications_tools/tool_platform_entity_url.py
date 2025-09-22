from ..tool_utils import *
import json
import ast

class GetPlatformEntityUrlSchema(BaseModel):
    """
    Schema for getting platform entity URL by entity system_name and type.
    """
    type: Literal['Application', 'Record Template', 'Role Template', 'Process Template', 'Organizational Structure Template'] = Field(
        description="Type of the entity to generate URL for"
    )
    system_name: str = Field(
        description="System name of the entity to generate URL for"
    )

    @field_validator("type", "system_name", mode="before")
    @classmethod
    def non_empty_str(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("Value must be a non-empty string")
        return v

@tool("get_platform_entity_url", return_direct=False, args_schema=GetPlatformEntityUrlSchema)
def get_platform_entity_url(
    system_name: str,
    type: str
    ) -> Dict[str, Any]:
    """
    Get the URL for an entity by its type and sytem name
        
    Returns:
        dict: {
            "success": bool - True if the entity URL was generated successfully
            "status_code": int - HTTP response status code  
            "entity_url": str|None - The generated entity URL. Example: https://{base_url}/#Resolver/{entity_id}
            "error": str|None - Error message if operation failed
        }
    """

    try:
        # Load server config to get base_url
        cfg = requests_._load_server_config()
        base_url = cfg.base_url.rstrip("/")
        
        if not base_url:
            final_result = {
                "success": False,
                "status_code": 500,
                "entity_url": None,
                "error": "Base URL not found in server configuration"
            }
            return final_result

        entity_type = GET_URL_TYPE_MAPPING[type]

        request_body = {
        "Type": entity_type
        }

        endpoint = "api/public/system/Solution/TemplateService/List"

        result = requests_._post_request(request_body, endpoint)
        result_body = result["raw_response"]
        result_body = ast.literal_eval(result_body)

        if entity_type == "Undefined":
            list_applications = list_applications.func()
            result_list_applications_body = list_applications['raw_response']
            application_name = next((item["name"] for item in result_list_applications_body if item["alias"] == system_name))

            result_body = result['raw_response']
            result_body = json.loads(result_body)
            entity_id = next((item["solution"] for item in result_body if item["solutionName"] == application_name))
        else:
            for item in result_body:
                alias = item.get("alias", "")

                if alias.strip() == system_name.strip():
                    entity_id = item['id']
                    break
   
        entity_url = f"{base_url}/#Resolver/{entity_id}"
        
        final_result = {
            "success": True,
            "status_code": 200,
            "entity_url": entity_url,
            "error": None
        }
        
        return final_result
        
    except Exception as e:
        final_result = {
            "success": False,
            "status_code": 500,
            "entity_url": None,
            "error": f"Error generating entity URL: {str(e)}"
        }
        return final_result

if __name__ == "__main__":
    results = get_platform_entity_url.invoke({
        "type": "Record Template",
        "system_name": "Test"
    })
    print(results)