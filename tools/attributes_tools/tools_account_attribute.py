from ..tool_utils import *

class EditOrCreateAccountAttributeSchema(CommonAttributeFields):
    use_as_record_title: bool = Field(
        default=False,
        description="Set to `True` to display as a template record title."
            "RU: Использовать как заголовок записей",
    )
    related_template_system_name: str = Field(
        default="_Account",
        description="System name of the template to associate with the attribute."
            "RU: Связанный шаблон"
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

@tool("edit_or_create_account_attribute", return_direct=False, args_schema=EditOrCreateAccountAttributeSchema)
def edit_or_create_account_attribute(
    operation: str,
    name: str,
    system_name: str,
    application_system_name: str,
    template_system_name: str,
    description: Optional[str] = None,
    use_as_record_title: Optional[bool] = False,
    write_changes_to_the_log: Optional[bool] = False,
    calculate_value: Optional[bool] = False,
    expression_for_calculation: Optional[str] = None,
    related_template_system_name: Optional[str] = "_Account",
    store_multiple_values: Optional[bool] = False
) -> Dict[str, Any]:
    """
    Edit or Create an account attribute (Аккаунт, учетная запись).

    Account attribute is linked to accounts in the system.

    Account attribute stores one or several linked account IDs.

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
        "type": "Account",
        "name": name,
        "description": description,
        "isTracked": write_changes_to_the_log,
        "isMultiValue": store_multiple_values,
        "isTitle": use_as_record_title,
        "isCalculated": calculate_value if expression_for_calculation != None else False,
        "expression": expression_for_calculation,
        "instanceGlobalAlias": {
            "type": "RecordTemplate",
            "alias": related_template_system_name
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
    results = edit_or_create_account_attribute.invoke({
        "operation": "create",
        "name": "Assigned User",
        "system_name": "AssignedUser",
        "application_system_name": "AItestAndApi",
        "template_system_name": "Test",
        "description": "User assigned to this record",
        "related_template_system_name": "_Account",
        "use_as_record_title": False
    })
    print(results)