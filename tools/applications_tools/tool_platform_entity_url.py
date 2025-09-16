from ..tool_utils import *

class GetPlatformEntityUrlSchema(BaseModel):
    """
    Schema for getting platform entity URL by entity ID.
    """
    entity_id: str = Field(
        description="The string ID of the entity to generate URL for",
        min_length=1
    )

    @field_validator("entity_id", mode="before")
    @classmethod
    def non_empty_str(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("Entity ID must be a non-empty string")
        return v

@tool("get_platform_entity_url", return_direct=False, args_schema=GetPlatformEntityUrlSchema)
def get_platform_entity_url(entity_id: str) -> Dict[str, Any]:
    """
    Get the URL for an entity by its string ID
        
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
        base_url = cfg.get("base_url", "").rstrip("/")
        
        if not base_url:
            final_result = {
                "success": False,
                "status_code": 500,
                "data": None,
                "error": "Base URL not found in server configuration"
            }
            validated = AttributeResult(**final_result)
            return validated.model_dump()
        
        # Construct the entity URL
        entity_url = f"{base_url}/#Resolver/{entity_id}"
        
        final_result = {
            "success": True,
            "status_code": 200,
            "entity_url": entity_url,
            "error": None
        }
        
        validated = AttributeResult(**final_result)
        return validated.model_dump()
        
    except Exception as e:
        final_result = {
            "success": False,
            "status_code": 500,
            "data": None,
            "error": f"Error generating entity URL: {str(e)}"
        }
        validated = AttributeResult(**final_result)
        return validated.model_dump()

if __name__ == "__main__":
    results = get_platform_entity_url.invoke({"entity_id": "test-entity-123"})
    print(results)
