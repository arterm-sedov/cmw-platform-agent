from ..tool_utils import *

ALLOWED_EXTENSIONS_LIST = [
    "TXT",
    "PNG",
    "JPG",
    "CSV",
    "XLSX",
    "DOCX",
    "PPTX",
    "VSDX",
    "MSG",
    "ZIP",
    "BMP",
    "EMF",
    "DWG",
    "BPMN",
    "LOG",
    "RAR",
    "TAR",
    "TAR.GZ(TGZ)",
    "GZ",
    "BZ2",
    "TAR.BZ2",
    "ENV",
    "UNL",
    "EML",
    "SQL",
    "ISO",
    "CONF",
    "ICO",
]

ALLOWED_EXTENSIONS = Literal[tuple(ALLOWED_EXTENSIONS_LIST)]
ALLOWED_EXTENSIONS_SET = set(ALLOWED_EXTENSIONS_LIST)


class EditOrCreateDocumentAttributeSchema(CommonAttributeFields):
    display_format: Optional[
        Literal["Attachment", "SignedDocument", "InlineDocument"]
    ] = Field(
        default=None,
        description="Attribute display format. "
        "RU: 'Формат отображения'. "
        "Required for create, optional for edit.",
    )
    use_to_search_records: bool = Field(
        default=False,
        description="Set to `True` to allow the users to search the records by this attribute's value. "
        "RU: Использовать для поиска записей",
    )
    file_extensions_filter: Optional[List[str]] = Field(
        default=None,
        description="Filter for file extensions that the attribute can store. "
        "RU: Фильтр расширений файлов",
    )

    @field_validator("display_format", mode="before")
    def non_empty_str(cls, v: Any) -> Any:
        """
        Validate that string fields are not empty.

        This field validator is automatically applied to the name, system_name,
        application_system_name, and template_system_name fields in all schemas
        that inherit from CommonAttributeFields, ensuring consistent validation.
        """
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("must be a non-empty string")
        return v

    @field_validator("file_extensions_filter", mode="before")
    def normalize_file_extensions_filter(cls, v: Any) -> Optional[List[str]]:
        if v is None:
            return v
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        if not isinstance(v, list):
            raise ValueError(
                "file_extensions_filter must be a list or comma-separated strings"
            )
        normalized = [str(ext).strip().lower() for ext in v]

        invalid = set(normalized) - ALLOWED_EXTENSIONS_SET
        if invalid:
            raise ValueError(
                f"Invalid file extensions: {sorted(invalid)}. "
                f"Allowed: {sorted(ALLOWED_EXTENSIONS_SET)}"
            )
        return normalized

    @model_validator(mode="after")
    def _validate_create_required_fields(self) -> "EditOrCreateDocumentAttributeSchema":
        """
        Validate that required fields are provided for create operations.

        When operation is 'create', the following fields are REQUIRED:
            - display_format: How the document is displayed (Attachment, SignedDocument, InlineDocument)

        When operation is 'edit', all fields are OPTIONAL - the tool will
        fetch current values from the API for any missing fields.
        """
        if self.operation == "create":
            if self.display_format is None:
                raise ValueError(
                    "display_format is REQUIRED when operation='create'. "
                    "Choose from: Attachment, SignedDocument, InlineDocument. "
                    "For edit operations, this field is optional."
                )
        return self


@tool(
    "edit_or_create_document_attribute",
    return_direct=False,
    args_schema=EditOrCreateDocumentAttributeSchema,
)
def edit_or_create_document_attribute(
    operation: str,
    name: str,
    system_name: str,
    application_system_name: str,
    template_system_name: str,
    display_format: Optional[str] = None,
    description: Optional[str] = None,
    use_to_search_records: Optional[bool] = False,
    write_changes_to_the_log: Optional[bool] = False,
    store_multiple_values: Optional[bool] = False,
    file_extensions_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Edit or Create a document attribute (Документа).

    Document attribute stores file attachments with configurable file format filters.

    Can also store signed documents with digital signatures.

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
            "alias": system_name,
        },
        "type": "Document",
        "name": name,
        "description": description,
        "isIndexed": use_to_search_records,
        "isTracked": write_changes_to_the_log,
        "isMultiValue": store_multiple_values,
    }
    if display_format is not None:
        request_body["format"] = display_format
    if file_extensions_filter is not None:
        request_body["fileFormat"] = file_extensions_filter

    endpoint = f"{ATTRIBUTE_ENDPOINT}/{application_system_name}"

    return execute_edit_or_create_operation(
        request_body=request_body,
        operation=operation,
        endpoint=endpoint,
        result_model=AttributeResult,
    )


if __name__ == "__main__":
    results = edit_or_create_document_attribute.invoke(
        {
            "operation": "create",
            "name": "Contract Document",
            "system_name": "ContractDocument",
            "application_system_name": "AItestAndApi",
            "template_system_name": "Test",
            "display_format": "Attachment",
            "description": "Contract document attachment",
            "use_to_search_records": False,
        }
    )
    print(results)
