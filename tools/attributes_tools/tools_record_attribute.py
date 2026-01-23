from ..tool_utils import *

class EditOrCreateRecordAttributeSchema(CommonAttributeFields):
    related_template_system_name: str = Field(
        description="System name of the template to link with the attribute. "
                    "RU: Связанный шаблон"
    )
    related_attribute_system_name: Optional[str] = Field(
        default=None,
        description="System name of a record attribute in the related template to back-link with the current attribute. "
                    "RU: Взаимная связь с атрибутом"
    )

    @field_validator("related_template_system_name", mode="before")
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

@tool("edit_or_create_record_attribute", return_direct=False, args_schema=EditOrCreateRecordAttributeSchema)
def edit_or_create_record_attribute(
    operation: str,
    name: str,
    system_name: str,
    application_system_name: str,
    template_system_name: str,
    related_template_system_name: str,
    description: Optional[str] = None,
    write_changes_to_the_log: Optional[bool] = False,
    calculate_value: Optional[bool] = False,
    expression_for_calculation: Optional[str] = None,
    store_multiple_values: Optional[bool] = False,
    related_attribute_system_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Edit or Create a record attribute (Запись, коллекция).

    Record attribute is is linked to records in a related template.

    Record attribute stores one or several IDs of the lined records in the related template.

    Record attribute can be mutually linked with the attribute in the related template. Mutually linked attributes are automatically cross-linked whenever the values of one of the attributes change.

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
        "type": "Instance",
        "name": name,
        "description": description,
        "isTracked": write_changes_to_the_log,
        "isMultiValue": store_multiple_values,
        "isCalculated": calculate_value if expression_for_calculation != None else False,
        "expression": expression_for_calculation,
        "instanceGlobalAlias": {
            "type": "Attribute" if related_attribute_system_name != None else "RecordTemplate",
            "owner": related_template_system_name if related_attribute_system_name !=None else None,
            "alias": related_attribute_system_name if related_attribute_system_name != None else related_template_system_name
        }
    }

    endpoint = f"{ATTRIBUTE_ENDPOINT}/{application_system_name}"

    return execute_edit_or_create_operation(
        request_body=request_body,
        operation=operation,
        endpoint=endpoint,
        result_model=AttributeResult
    )

if __name__ == "__main__":
    results = edit_or_create_record_attribute.invoke({
        "operation": "create",
        "name": "Related Task",
        "system_name": "RelatedTask",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test",
        "related_template_system_name": "Task",
        "description": "Related task instance",
        "store_multiple_values": False
    })
    print(results)