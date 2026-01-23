from ..tool_utils import *

ALLOWED_EXTENSIONS_LIST = ['PNG', 'JPG', 'BMP', 'EMF']
ALLOWED_COLOR_RENDERING_MODES_LIST = ['Original', 'Bitonal', 'GreyScale']

ALLOWED_EXTENSIONS_SET = set(ALLOWED_EXTENSIONS_LIST)
ALLOWED_COLOR_RENDERING_MODES_SET = set(ALLOWED_COLOR_RENDERING_MODES_LIST)
class EditOrCreateImageAttributeSchema(CommonAttributeFields):
    rendering_color_mode: str = Field(
        description="Image color mode. "
                    "RU: Цветовой режим. "
                    f"Allowed: {ALLOWED_COLOR_RENDERING_MODES_LIST}"
    )
    use_to_search_records: bool = Field(
        default=False,
        description="Set to `True` to allow the users to search the records by this attribute's value. "
                    "RU: Использовать для поиска записей",
    )
    file_extensions_filter: Optional[List[str]] = Field(
        default=None,
        description="Filter of file extensions that can store an attribute. "
                    "RU: Фильтр расширений файлов. "
                    f"Allowed: {ALLOWED_EXTENSIONS_LIST}"
    )
    image_width: Optional[int] = Field(
        default=None,
        description="Image width. "
                    "RU: Ширина"
    )
    image_height: Optional[int] = Field(
        default=None,
        description="Image height. "
                    "RU: Высота"
    )
    save_image_aspect_ratio: bool = Field(
        default=False,
        description="Set to `True` to maintain the image's original aspect ratio. "
                    "RU: Сохранить соотношения сторон"
    )

    @field_validator("file_extensions_filter", mode="before")
    def normalize_file_extensions_filter(cls, v: Any) -> Optional[List[str]]:
        if v is None:
            return v
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        if not isinstance(v, list):
            raise ValueError("file_extensions_filter must be a list or comma-separated string")    
        normalized = [str(ext).strip().lower() for ext in v]

        invalid = set(normalized) - ALLOWED_EXTENSIONS_SET
        if invalid:
            raise ValueError(f"Invalid file extensions: {sorted(invalid)}. " f"Allowed: {sorted(ALLOWED_EXTENSIONS_SET)}")
        return normalized

    @field_validator("rendering_color_mode", mode="before")
    def validate_rendering_color_mode(cls, v: Any) -> str:
        if v is None:
            raise ValueError("rendering_color_mode is required")
        normalized = str(v).strip()
        if normalized not in ALLOWED_COLOR_RENDERING_MODES_SET:
            raise ValueError(f"Invalid color mode: {normalized}. Allowed: {sorted(ALLOWED_COLOR_RENDERING_MODES_SET)}")
        return normalized

@tool("edit_or_create_image_attribute", return_direct=False, args_schema=EditOrCreateImageAttributeSchema)
def edit_or_create_image_attribute(
    operation: str,
    name: str,
    system_name: str,
    application_system_name: str,
    template_system_name: str,
    rendering_color_mode: str,
    description: Optional[str] = None,
    use_to_search_records: Optional[bool] = False,
    write_changes_to_the_log: Optional[bool] = False,
    store_multiple_values: Optional[bool] = False,
    file_extensions_filter: Optional[List[str]] = None,
    save_image_aspect_ration: Optional[bool] = False,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None
) -> Dict[str, Any]:
    """
    Edit or Create an image attribute (Изображение).

    Image attribute stores image files with configurable color modes and dimensions.

    The image will be converted and stored in the specified color mode and dimensions.

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
        "type": "Image",
        "name": name,
        "description": description,
        "isIndexed": use_to_search_records,
        "isTracked": write_changes_to_the_log,
        "isMultiValue": store_multiple_values,
        "fileFormat": file_extensions_filter,
        "imageWidth": image_width,
        "imageHeight": image_height,
        "imageColorType": rendering_color_mode,
        "imagePreserveAspectRatio": save_image_aspect_ration
    }

    endpoint = f"{ATTRIBUTE_ENDPOINT}/{application_system_name}"

    return execute_edit_or_create_operation(
        request_body=request_body,
        operation=operation,
        endpoint=endpoint,
        result_model=AttributeResult
    )

if __name__ == "__main__":
    results = edit_or_create_image_attribute.invoke({
        "operation": "create",
        "name": "Product Image",
        "system_name": "ProductImage",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test",
        "rendering_color_mode": "Original",
        "description": "Product image attachment",
        "use_to_search_records": False
    })
    print(results)