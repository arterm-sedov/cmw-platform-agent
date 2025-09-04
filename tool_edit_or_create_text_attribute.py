from typing import Any, Dict, List, Optional
import json
from typing import Optional
import requests
import yaml
import base64
from langchain.tools import tool
import requests_

# Load server config from YAML
def _load_server_config() -> Dict[str, str]:
    with open("server_config.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
        base_url = (cfg.get("base_url") or "").rstrip("/")
        login = cfg.get("login") or ""
        password = cfg.get("password") or ""
        if not base_url:
            raise RuntimeError("'base_url' is required in server_config.yml")
        if not login:
            raise RuntimeError("'login' is required in server_config.yml")
        if not password:
            raise RuntimeError("'password' is required in server_config.yml")
        return {"base_url": base_url, "login": login, "password": password}

# ---- Configuration ----
# Load connection settings at call time to always use the latest config

def _set_validation_mask(display_format: str) -> str:
    # Setting validation mask via display format
    validation_mask_mapping: Dict[str, str] = {
        "LicensePlateNumberRuMask":"([АВЕКМНОРСТУХавекмнорстух]{1}[0-9]{3}[АВЕКМНОРСТУХавекмнорстух]{2} [0-9]{3})",
        "IndexRuMask": "([0-9]{6})",
        "PassportRuMask": "([0-9]{4} [0-9]{6})",
        "INNMask": "([0-9]{10})",
        "OGRNMask": "([0-9]{13})",
        "IndividualINNMask": "([0-9]{12})",
        "PhoneRuMask": "(\+7 \([0-9]{3}\) [0-9]{3}-[0-9]{2}-[0-9]{2})",
        "EmailMask": "^(([a-zа-яё0-9_-]+\.)*[a-zа-яё0-9_-]+@[a-zа-яё0-9-]+(\.[a-zа-яё0-9-]+)*\.[a-zа-яё]{2,6})?$",
        "CustomMask": None
    }

    if validation_mask_mapping[display_format]:
        return validation_mask_mapping[display_format]
    else:
        return None
    

def _basic_headers() -> Dict[str, str]:
    # Basic authentication from YAML config
    cfg = _load_server_config()
    login = cfg.get("login")
    password = cfg.get("password")
    credentials = base64.b64encode(f"{login}:{password}".encode("ascii")).decode("ascii")
    return {
        "Authorization": f"Basic {credentials}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

def _remove_nones(obj: Any) -> Any:
    """
    Recursively remove None values from dicts/lists to keep payload minimal and consistent with Platform expectations.
    """
    if isinstance(obj, dict):
        return {k: _remove_nones(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [ _remove_nones(v) for v in obj if v is not None]
    return obj

@tool("edit_or_create_text_attribute", return_direct=False)
def create_text_attribute(
    name: str,
    system_name: str,
    application_system_name: str,
    template_system_name: str,
    display_format: str,
    description: Optional[str] = None,
    custom_mask: Optional[str] = None,
    control_uniqueness: Optional[bool] = False,
    use_as_record_title: Optional[bool] = False
) -> Dict[str, Any]:
    r"""
    {
        "Decstiption": "Edit or Create a text attribute",
        "Returns": {
            "success": {
                "type": "boolean",
                "description": "True if attribute was created successfully"
            },
            "status_code": {
                "type": "integer",
                "description": "HTTP response status code"
            },
            "raw_response": {
                "type": "string",
                "description": "Raw response text for auditing"
            },
            "error": {
                "type": "string",
                "default": null,
                "description": "Error message if any"
            }
        },
        "Parameters": {
            "name": {
                "Russian name": "Название",
                "English name": "Name",
                "type": "string",
                "description": "Human-readable name of the attribute"
            },
            "system_name": {
                "Russian name": "Системное имя",
                "English name": "System name",
                "type": "string",
                "description": "Unique system name of the attribute"
            },
            "description": {
                "Russian name": "Описание",
                "English name": "Description",
                "type": "string",
                "default": null,
                "description": "Human-readable description of the attribute. Generate if not provided"
            },
            "application_system_name": {
                "Russian name": "Системное имя приложения",
                "English name": "Application system name",
                "type": "string",
                "description": "System name of the application with the template where the attribute is created"
            },
            "template_system_name": {
                "Russian name": "Системное имя шаблона",
                "English name": "Template system name",
                "type": "string",
                "description": "System name of the template where the attribute is created"
            },
            "display_format": {
                "Russian name": "Формат отображения",
                "English name": "Display format",
                "type": "string",
                "description": "Attribute display format",
                "allowed_values": {
                    "PlainText": {
                        "Russian name": "Простой текст"
                    },
                    "MarkedText": {
                        "Russian name": "Размеченный текст"
                    },
                    "HtmlText": {
                        "Russian name": "HTML-текст"
                    },
                    "LicensePlateNumberRuMask": {
                        "Russian name": "Регистрация номера ТС (РФ)",
                        "Input mask": "([АВЕКМНОРСТУХавекмнорстух]{1}[0-9]{3}[АВЕКМНОРСТУХавекмнорстух]{2} [0-9]{3})"
                    },
                    "IndexRuMask": {
                        "Russian name": "Индекс (РФ)",
                        "Input mask": "([0-9]{6})"
                    },
                    "PassportRuMask": {
                        "Russian name": "Паспорт (РФ)",
                        "Input mask": "([0-9]{4} [0-9]{6})"
                    },
                    "INNMask": {
                        "Russian name": "ИНН юрлица",
                        "Input mask": "([0-9]{10})"
                    },
                    "OGRNMask": {
                        "Russian name": "ОГРН",
                        "Input mask": "([0-9]{13})"
                    },
                    "IndividualINNMask": {
                        "Russian name": "ИНН физлица",
                        "Input mask": "([0-9]{12})"
                    },
                    "PhoneRuMask": {
                        "Russian name": "Телефон (РФ)",
                        "Input mask": "(\\+7 \\([0-9]{3}\\) [0-9]{3}-[0-9]{2}-[0-9]{2})"
                    },
                    "EmailMask": {
                        "Russian name": "Адрес эл. почты",
                        "Input mask": "^(([a-zа-яё0-9_-]+\\.)*[a-zа-яё0-9_-]+@[a-zа-яё0-9-]+(\\.[a-zа-яё0-9-]+)*\\.[a-zа-яё]{2,6})?$"
                    },
                    "CustomMask": {
                        "Russian name": "Особая маска",
                        "Input mask": "user-defined input mask"
                    }
                }
            },
            "custom_mask": {
                "Russian name": "Особая маска",
                "English name": "Custom mask",
                "type": "string",
                "default": null,
                "description": "A special formatting mask; fill only if display_format = CustomMask"
            },
            "control_uniqueness": {
                "Russian name": "Контролировать уникальность значения",
                "English name": "Control uniqueness",
                "type": "boolean",
                "default": false,
                "description": "Whether attribute values must be unique"
            },
            "use_as_record_title": {
                "Russian name": "Использовать как заголовок записей",
                "English name": "Use as record title",
                "type": "boolean",
                "default": false,
                "description": "Whether attribute values will be displayed as a template record title"
            }
        }
    }
    """
    # Base URL from YAML (mandatory, validated in loader)
    cfg = _load_server_config()
    base_url = cfg.get("base_url")
    url = f"{base_url}/webapi/Attribute/{application_system_name}"
    headers = _basic_headers()

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
        "isUnique": control_uniqueness,
        "isTitle": use_as_record_title,
        "validationMaskRegex": custom_mask if display_format == "CustomMask" else _set_validation_mask(display_format)
    }

    # Remove None values
    request_body = _remove_nones(request_body) 

    result = requests_._create_attribute_request(request_body, application_system_name)

    requests_._edit_attribute_request(request_body, application_system_name)

    return result


if __name__ == "__main__":
    results = create_text_attribute.invoke({
        "name": "Test15",
        "system_name": "Test15",
        "application_system_name": "Malatik",
        "template_system_name": "Test",
        "display_format": "INNMask",
        "description": None,
        "custom_mask": None,
        "control_uniqueness": False,
        "use_as_record_title": True,
        "attribute_type": "String"
    })
    print(results)