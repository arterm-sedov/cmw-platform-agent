from typing import Any, Dict, List, Optional
from typing import Optional
from langchain.tools import tool
import requests_

ATTRIBUTE_ENDPOINT = "webapi/Attribute"

def _set_input_mask(display_format: str) -> str:
    # Setting validation mask via display format
    input_mask_mapping: Dict[str, str] = {
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

    if input_mask_mapping[display_format]:
        return input_mask_mapping[display_format]
    else:
        return None

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
def edit_or_create_text_attribute(
    operation: str,
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
            "operation": {
                "Rusian names": ["Создать", "Редактировать"],
                "variants": ["create", "edit"],
                "description": "Choose operation: Creates or Edits the attribute."
            }
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
                "English name": "Validate for unique values",
                "type": "boolean",
                "default": false,
                "description": "Control whether attribute values must be unique"
            },
            "use_as_record_title": {
                "Russian name": "Использовать как заголовок записей",
                "English name": "Use as record title",
                "type": "boolean",
                "default": false,
                "description": "Control whether attribute values will be displayed as a template record title"
            }
        }
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
        "isUnique": control_uniqueness,
        "isTitle": use_as_record_title,
        "validationMaskRegex": custom_mask if display_format == "CustomMask" else _set_input_mask(display_format)
    }

        # Remove None values
    request_body = _remove_nones(request_body) 

    if operation == "create":

        result = requests_._post_request(request_body, f"{ATTRIBUTE_ENDPOINT}/{application_system_name}")

        requests_._put_request(request_body, f"{ATTRIBUTE_ENDPOINT}/{application_system_name}")


    elif operation == "edit":

        result = requests_._put_request(request_body, f"{ATTRIBUTE_ENDPOINT}/{application_system_name}")
    
    else:
        result = "Нет такой операции над атрибутом."

    return result

@tool("get_text_attribute", return_direct=False)
def get_text_attribute(
    application_system_name: str,
    template_system_name: str,
    system_name: str
    ) -> Dict[str, Any]:
    """
    {
        "Decstiption": "Get a text attribute",
        "Returns": {
            "success": {
                "type": "boolean",
                "description": "True if attribute was geted successfully"
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
            "system_name": {
                "Russian name": "Системное имя",
                "English name": "System name",
                "type": "string",
                "description": "Unique system name of the attribute"
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
            }
        }
    }
    """

    attribute_global_alias = f"Attribute@{template_system_name}.{system_name}"

    result = requests_._get_request(f"{ATTRIBUTE_ENDPOINT}/{application_system_name}/{attribute_global_alias}")

    result_body = result['raw_response']

    keys_to_remove = ['isMultiValue', 'isMandatory', 'isOwnership', 'instanceGlobalAlias', 'imageColorType', 'imagePreserveAspectRatio']

    for key in keys_to_remove:
        if key in result_body['response']:
            result_body['response'].pop(key, None)

    result.update({"raw_response": result_body['response']})
    return result

if __name__ == "__main__":
    results = get_text_attribute.invoke({
        "system_name": "Test16",
        "application_system_name": "Malatik",
        "template_system_name": "Test",
    })
    print(results)