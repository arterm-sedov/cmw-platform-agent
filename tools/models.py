from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator

class TemplateResult(BaseModel):
    """
    Result model for a fetched template.
    Contains cleaned data after succesful fetch and processing.
    """
    success: bool
    status_code: int
    error: Optional[str] = Field(default=None)
class AttributeResult(BaseModel):
    """
    Result model for a fetched attribute.
    Contains cleaned data after successful fetch and processing.
    """
    success: bool
    status_code: int
    error: Optional[str] = Field(default=None)

    class Config:
        extra = 'allow'

class CommonAttributeFields(BaseModel):
    """
    Common field definitions for attribute schemas across all tool modules.
    
    This mixin class provides standardized field definitions to reduce code duplication
    and ensure consistency across all attribute-related schemas.
    """
    operation: Literal["create", "edit"] = Field(
        description="Choose operation: Create or Edit the attribute. "
                    "RU: Создать, Редактировать"
    )
    name: str = Field(
        description="Human-readable name of the attribute. "
                    "RU: Название"
    )
    system_name: str = Field(
        description="System name of the attribute. "
                    "RU: Системное имя"
    )
    application_system_name: str = Field(
        description="System name of the application with the template where the attribute is created or edited. "
                    "RU: Системное имя приложения"
    )
    template_system_name: str = Field(
        description="System name of the template where the attribute is created or edited. "
                    "RU: Системное имя шаблона"
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the attribute (auto-generate if omitted). "
                    "RU: Описание",
    )
    write_changes_to_the_log: bool = Field(
        default=False,
        description="Set to `True` to log attribute value changes. "
                    "RU: Записывать изменения в журнал",
    )
    calculate_value: bool = Field(
        default=False,
        description="Set to `True` to calculate the attribute value automatically. "
                    "Relevant only when `expression_for_calculation` is provided. "
                    "RU: Вычислять автоматически"
    )
    expression_for_calculation: Optional[str] = Field(
        default=None,
        description="Expression to automatically calculate the attribute value; user-provided. "
                    "RU: Выражение для вычисления",
    )
    store_multiple_values: bool = Field(
        default=False,
        description="Set to `True` to store multiple values. "
                    "RU: Хранить несколько значений"
    )

    @field_validator("operation", mode="before")
    def normalize_operation_create_edit(cls, v: str) -> str:
        """
        Normalize Create/Edit operation field values by converting them to lowercase and applying a mapping.
        
        This field validator is automatically applied to the operation field in all schemas
        that inherit from CommonAttributeFields, ensuring consistent operation normalization.
        """
        if v is None:
            return v
        
        # Default mapping used by most attribute tools
        mapping = {
            "создать": "create",
            "редактировать": "edit",
        }
        
        normalized_value = str(v).strip().lower()
        return mapping.get(normalized_value, normalized_value)

    @field_validator("name", "system_name", "application_system_name", "template_system_name", mode="before")
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


class CommonGetAttributeFields(BaseModel):
    """
    Common field definitions for get attribute schemas across all tool modules.
    
    This mixin class provides standardized field definitions for schemas that need
    to fetch/retrieve attributes, reducing code duplication and ensuring consistency.
    """
    application_system_name: str = Field(
        description="System name of the application with the template where the attribute is located. "
                    "RU: Системное имя приложения"
    )
    template_system_name: str = Field(
        description="System name of the template where the attribute is located. "
                    "RU: Системное имя шаблона"
    )
    system_name: str = Field(
        description="Unique system name of the attribute to fetch. "
                    "RU: Системное имя атрибута"
    )

    @field_validator("application_system_name", "template_system_name", "system_name", mode="before")
    def non_empty_str(cls, v: Any) -> Any:
        """
        Validate that string fields are not empty.
        
        This field validator is automatically applied to the application_system_name, 
        template_system_name, and system_name fields in all get attribute schemas
        that inherit from CommonGetAttributeFields, ensuring consistent validation.
        """
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("must be a non-empty string")
        return v


def normalize_operation_archive_unarchive(value: str) -> str:
    """
    Normalize operation values for archive/unarchive operations.
    
    This utility function handles the specific mapping for archive/unarchive operations,
    converting Russian terms to English equivalents.
    
    Args:
        value: The operation value to normalize
        
    Returns:
        Normalized operation value
    """
    if value is None:
        return value
    
    mapping = {
        "архивировать": "archive",
        "разархивировать": "unarchive"
    }
    
    normalized_value = str(value).strip().lower()
    return mapping.get(normalized_value, normalized_value)