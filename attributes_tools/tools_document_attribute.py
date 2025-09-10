from tool_utils import *

ALLOWED_EXTENSIONS_LIST = ['TXT', 'PNG', 'JPG', 'CSV', 'XLSX', 'DOCX', 'PPTX', 'VSDX', 'MSG', 'ZIP', 'BMP', 'EMF', 'DWG', 'BPMN', 'LOG', 'RAR', 'TAR', 'TAR.GZ(TGZ)', 'GZ', 'BZ2', 'TAR.BZ2', 'ENV', 'UNL', 'EML', 'SQL', 'ISO', 'CONF', 'ICO']

ALLOWED_EXTENSIONS = Literal[tuple(ALLOWED_EXTENSIONS_LIST)]
ALLOWED_EXTENSIONS_SET = set(ALLOWED_EXTENSIONS_LIST)

class EditOrCreateDocumentAttributeSchema(CommonAttributeFields):
    display_format: Literal[
        "Attachment",
        "SignedDocument",
        "InlineDocument"
    ] = Field(
        description="Attribute display format. "
                    "RU: 'Формат отображения'."
    )
    use_to_search_records: bool = Field(
        default=False,
        description="Set to `True` to allow the users to search the records by this attribute's value. "
                    "RU: Использовать для поиска записей",
    )
    file_extensions_filter: Optional[List[Literal[ALLOWED_EXTENSIONS]]] = Field(
        default=None,
        description="Filter for file extensions that the attribute can store. "
                    "RU: Фильтр расширений файлов"
    )

    @field_validator("file_extensions_filter", mode="before")
    @classmethod
    def normalize_file_extensions_filter(cls, v: Any) -> Optional[List[str]]:
        if v is None:
            return v
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        if not isinstance(v, list):
            raise ValueError("file_extensions_filter must be a list or comma-separated strings")    
        normalized = [str(ext).strip().lower() for ext in v]

        invalid = set(normalized) - ALLOWED_EXTENSIONS_SET
        if invalid:
            raise ValueError(f"Invalid file extensions: {sorted(invalid)}. " f"Allowed: {sorted(ALLOWED_EXTENSIONS_SET)}")
        return normalized

@tool("edit_or_create_document_attribute", return_direct=False, args_schema=EditOrCreateDocumentAttributeSchema)
def edit_or_create_document_attribute(
    operation: str,
    name: str,
    system_name: str,
    application_system_name: str,
    template_system_name: str,
    display_format: str,
    description: Optional[str] = None,
    use_to_search_records: Optional[bool] = False,
    write_changes_to_the_log: Optional[bool] = False,
    store_multiple_values: Optional[bool] = False,
    file_extensions_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Edit or Create a document attribute.
    
    Returns:
        dict: {
            "success": bool - True if the attribute was created or edited successfully
            "status_code": int - HTTP response status code  
            "raw_response": dict|str|None - Raw response for auditing or payload body (sanitized)
            "error": str|None - Error message if operation failed
        }
    """

    request_body: Dict[str, Any] = {
        "globalAlias": {
            "owner": template_system_name,
            "type": "Undefined",
            "alias": system_name
        },
        "type": "String",
        "format": display_format,
        "name": name,
        "description": description,
        "isIndexed": use_to_search_records,
        "isTracked": write_changes_to_the_log,
        "isMultiValue": store_multiple_values,
        "fileFormat": file_extensions_filter
    }

        # Remove None values
    request_body = remove_nones(request_body) 

    try:
        if operation == "create":
            result = requests_._post_request(request_body, f"{ATTRIBUTE_ENDPOINT}/{application_system_name}")
        if operation == "edit" or operation == "create":
            result = requests_._put_request(request_body, f"{ATTRIBUTE_ENDPOINT}/{application_system_name}")
            print("edit is complited")
        else:
            result = {
                "success": False,
                "error": f"No such operation for attribute: {operation}. Available operations: create, edit",
                "status_code": 400
            }
    except Exception as e:
        result = {
            "success": False,
            "error": f"Tool execution failed: {str(e)}",
            "status_code": 500
        }

    # Ensure result is always a dict with proper structure
    if not isinstance(result, dict):
        result = {
            "success": False,
            "error": f"Unexpected result type: {type(result)}",
            "status_code": 500
        }
    
    # Add additional error information if the API call failed
    if not result.get("success", False) and result.get("error"):
        error_info = result.get("error", "")
        result["error"] = f"API operation failed: {error_info}"

    validated = AttributeResult(**result)
    return validated.model_dump()


@tool("get_document_attribute", return_direct=False, args_schema=CommonGetAttributeFields)
def get_document_attribute(
    application_system_name: str,
    template_system_name: str,
    system_name: str
    ) -> Dict[str, Any]:
    """
    Get a document attribute in a given template and application.
    
    Returns:
        dict: {
            "success": bool - True if the attribute was fetched successfully
            "status_code": int - HTTP response status code  
            "raw_response": dict|str|None - Raw response payload for auditing or payload body (sanitized)
            "error": str|None - Error message if operation failed
        }
    """

    attribute_global_alias = f"Attribute@{template_system_name}.{system_name}"

    result = requests_._get_request(f"{ATTRIBUTE_ENDPOINT}/{application_system_name}/{attribute_global_alias}")

    # Check if the request was successful and has the expected structure
    if not result.get('success', False):
        return result
    
    result_body = result.get('raw_response')
    if result_body is None:
        result.update({"error": "No response data received from server"})
        return result
    
    # Check if result_body has the expected 'response' key
    if not isinstance(result_body, dict) or 'response' not in result_body:
        result.update({"error": "Unexpected response structure from server"})
        return result

    keys_to_remove = ['isTitle', 'isUnique', 'isCalculated', 'isMandatory', 'isOwnership', 'instanceGlobalAlias', 'imageColorType', 'imagePreserveAspectRatio']

    for key in keys_to_remove:
        if key in result_body['response']:
            result_body['response'].pop(key, None)

    result.update({"raw_response": result_body['response']})
    validated = AttributeResult(**result)
    return validated.model_dump()

if __name__ == "__main__":
    results = edit_or_create_document_attribute.invoke({
        "operation": "create",
        "name": "Contract Document",
        "system_name": "ContractDocument",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test",
        "display_format": "Attachment",
        "description": "Contract document attachment",
        "use_to_search_records": False
    })
    print(results)