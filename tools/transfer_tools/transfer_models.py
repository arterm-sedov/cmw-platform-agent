# transfer_models.py - Pydantic schemas for transfer tools
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ExportApplicationSchema(BaseModel):
    """Schema for exporting an application to CTF format."""

    application_system_name: str = Field(
        description="System name (alias) of the application to export. "
                    "RU: Системное имя приложения"
    )
    save_to_file: bool = Field(
        default=True,
        description="If True, saves the CTF file to disk and returns the file path. "
                    "RU: Сохранить в файл",
    )

    @field_validator("application_system_name", mode="before")
    @classmethod
    def non_empty_str(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("Value must be a non-empty string")
        return v


class ImportApplicationSchema(BaseModel):
    """Schema for importing an application from CTF format."""

    application_system_name: str = Field(
        description="System name (alias) for the imported application. "
                    "RU: Системное имя приложения"
    )
    ctf_data: str | None = Field(
        default=None,
        description="Base64-encoded CTF data. Required if ctf_file_path not provided. "
                    "RU: CTF данные в Base64",
    )
    ctf_file_path: str | None = Field(
        default=None,
        description="Path to a local CTF file. If provided, CTF data will be read from it. "
                    "RU: Путь к CTF файлу",
    )
    update_existing: bool = Field(
        default=False,
        description="If True, update/replace existing application with same name. "
                    "If False, create as new application with ApplyNew policy.",
    )

    @field_validator("application_system_name", mode="before")
    @classmethod
    def non_empty_str(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("Value must be a non-empty string")
        return v
