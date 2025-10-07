from typing import Any, Dict, List, Optional, Literal, Type, Set
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator
from . import requests_ as requests_
from .models import (
    TemplateResult,
    AttributeResult, 
    CommonAttributeFields, 
    CommonGetAttributeFields,
    normalize_operation_archive_unarchive
)

import inspect

# Common constants
APPLICATION_ENDPOINT = "webapi/Solution"
ATTRIBUTE_ENDPOINT = "webapi/Attribute"
RECORD_TEMPLATE_ENDPOINT = "webapi/RecordTemplate"
KEYS_TO_REMOVE_MAPPING = {
    "String": ['isMultiValue', 'isMandatory', 'isOwnership', 'instanceGlobalAlias', 'imageColorType', 'imagePreserveAspectRatio'],
    "Role": ['instanceGlobalAlias', 'isTitle', 'isUnique', 'isIndexed', 'isMandatory', 'isOwnership', 'imageColorType', 'imagePreserveAspectRatio'],
    "Instance": ['isTitle', 'isUnique', 'isIndexed', 'isMandatory', 'isOwnership', 'imageColorType', 'imagePreserveAspectRatio'],
    "Image": ['isUnique', 'format', 'isCalculated', 'isTitle', 'isMandatory', 'isOwnership', 'instanceGlobalAlias'],
    "Enum": ['isMultiValue', 'isMandatory', 'isOwnership', 'instanceGlobalAlias', 'imageColorType', 'imagePreserveAspectRatio'],
    "Duration": ['isUnique', 'isIndexed', 'isMultiValue', 'isMandatory', 'isOwnership', 'instanceGlobalAlias', 'imageColorType', 'imagePreserveAspectRatio'],
    "Drawing": ['isIndexed', 'isMultiValue', 'imageColorType', 'imagePreserveAspectRatio', 'isUnique', 'format', 'isCalculated', 'isTitle', 'isMandatory', 'isOwnership', 'instanceGlobalAlias'],
    "Document": ['isTitle', 'isUnique', 'isCalculated', 'isMandatory', 'isOwnership', 'instanceGlobalAlias', 'imageColorType', 'imagePreserveAspectRatio'],
    "Decimal": ['isMultiValue', 'isMandatory', 'isOwnership', 'instanceGlobalAlias', 'imageColorType', 'imagePreserveAspectRatio'],
    "DateTime": ['isUnique', 'isIndexed', 'isMultiValue', 'isMandatory', 'isOwnership', 'instanceGlobalAlias', 'imageColorType', 'imagePreserveAspectRatio'],
    "Boolean": ['isMultiValue', 'isMandatory', 'isOwnership', 'instanceGlobalAlias', 'imageColorType', 'imagePreserveAspectRatio'],
    "Account": ['isUnique', 'isIndexed', 'isMandatory', 'isOwnership', 'imageColorType', 'imagePreserveAspectRatio'],
    "Process": ['isUnique', 'isIndexed', 'isTitle', 'isCalculated', 'isMultiValue', 'isMandatory', 'isOwnership', 'instanceGlobalAlias', 'imageColorType', 'imagePreserveAspectRatio'],
    "Conversation": ['isUnique', 'isIndexed', 'isTitle', 'isCalculated', 'isMultiValue', 'isMandatory', 'isOwnership', 'instanceGlobalAlias', 'imageColorType', 'imagePreserveAspectRatio'],
    "Color": ['isUnique', 'isIndexed', 'isTitle', 'isCalculated', 'isMultiValue', 'isMandatory', 'isOwnership', 'instanceGlobalAlias', 'imageColorType', 'imagePreserveAspectRatio'],
    "Record": ['relatedTemplate', 'isReferenceData', 'isTransferable', 'keyProperty', 'conversationDisplayConfig'],
    "Application": []

}

ATTRIBUTE_MODEL_DESCRIPTIONS = {
    "Account": """Stores one or several linked account IDs.""",
    "Boolean": "Stores `true` or `false`.",
    "DateTime": """Stores date and time along with a `Display format`:
        - DateISO: 1986-09-04
        - YearMonth: сентябрь 1986
        - TimeTracker: остался 1 д 12 ч 55 мин
        - DateTimeISO: 1986-09-04 03:30:00
        - ShortDateLongTime: 04.09.1986 г. 03:30:00
        - ShortDateShortTime: 04.09.1986 03:30
        - CondensedDateTime: 4 сент. 1986 г. 03:30
        - MonthDay: 4 сентября
        - CondensedDate: 4 сент. 1986 г.
        - ShortDate: 4.09.1986
        - LongDate: 4 сентября 1986 г.
        - LongDateLongTime: 4 сентября 1986 г. 03:30:00
        - LongDateShortTime: 4 сентября 1986 г. 03:30""",
    "Decimal": """Stores numeric numbers with configurable decimal places.
        Decimal places examples:
        - 0: 123
        - 1: 123.4
        - 2: 123.45""",
    "Document": "Stores file attachments with configurable file format filters.",
    "Drawing": "Stores floor plans based on CAD files.",
    "Duration": """Supports various display formats:
        - DurationHMS: 2 ч 55 м 59 с
        - DurationHM: 2 ч 55 м
        - DurationHMSTime: 2:55:59 (макс 23:59:59)
        - DurationD8HM: 3 д 12 ч 55 м (8-часовой день)
        - DurationD24HM: 1 д 12 ч 55 м (24-часовой день)
        - DurationFullShort: 1 д 12 ч 55 м 59 с
        - DurationHMTime: 2:55 (макс. 23:59)""",
    "Enum": "Stores a list of values with multiple language support and color coding.",
    "Image": "Stores image files with configurable color modes and dimensions.",
    "Instance": "Stores one or several IDs of the linked records in the related template. "
        "Record attribute can be mutually linked with the attribute in the related template. "
        "Mutually linked attributes are automatically cross-linked whenever the values of one of the attributes change.",
    "Role": "Stores one or several linked role IDs.",
    "String": r"""Stores a sting value.
        Supports various display formats including predefined and custom masks for common Russian data types:
        - LicensePlateNumberRuMask: ([АВЕКМНОРСТУХавекмнорстух]{1}[0-9]{3}[АВЕКМНОРСТУХавекмнорстух]{2} [0-9]{3})
        - IndexRuMask: ([0-9]{6})
        - PassportRuMask: ([0-9]{4} [0-9]{6})
        - INNMask: ([0-9]{10})
        - OGRNMask: ([0-9]{13})
        - IndividualINNMask: ([0-9]{12})
        - PhoneRuMask: (\+7 \([0-9]{3}\) [0-9]{3}-[0-9]{2}-[0-9]{2})
        - EmailMask: ^(([a-zа-яё0-9_-]+\.)*[a-zа-яё0-9_-]+@[a-zа-яё0-9-]+(\.[a-zа-яё0-9-]+)*\.[a-zа-яё]{2,6})?$"""
}

ATTRIBUTE_RESPONSE_MAPPING = {
    "owner": "Parent template system name",
    "alias": "Attribute system name",
    "type": "Attribute type",
    "format": "Display format",
    "name": "Name",
    "description": "Description",
    "isSystem": "Is system",
    "isDisabled": "Archived",
    "isUnique": "Control value uniqueness",
    "isIndexed": "Use to search records",
    "isTracked": "Write changes to the log",
    "isDigitGrouping": "Group digits numbers",
    "isMultiValue": "Store multiple values",
    "isTitle": "Use as record title",
    "isCalculated": "Calculate value",
    "isReadonly": "Is readonly",
    "expression": "Expression for value calculation",
    "fileFormat": "File extensions filter",
    "uriSchemeFormats": "Allowed URI schemes  ",
    "instanceAlias": "Related template system name",
    "instanceAttributeAlias": "Related attribute system name",
    "imageColorType": "Rendering color mode",
    "imageWidth": "Image width",
    "imageHeight": "Image height",
    "imagePreserveAspectRatio": "Save image aspect ratio",
    "imageXResolution": "X-axis image resolution",
    "imageYResolution": "Y-axis image resolution",
    "decimalPlaces": "Number decimal places",
    "validationMaskRegex": "Custom mask",
    "variants": "Enum values",
    "linkedRecordTemplate": "Related template ID"
}

TEMPLATE_RESPONSE_MAPPING = {
    "alias": "Template system name",
    "type": "Template type",
    "name": "Name",
    "description": "Description"
}

APPLICATION_RESPONSE_MAPPING = {
    "alias": "Application system name",
    "name": "Name",
    "description": "Description",
    "isDefault": "Use by default"
}

ENTITY_TYPE_MAPPING = {
    "attribute": [ATTRIBUTE_MODEL_DESCRIPTIONS, ATTRIBUTE_RESPONSE_MAPPING],
    "template": [None, TEMPLATE_RESPONSE_MAPPING],
    "application": [None, APPLICATION_RESPONSE_MAPPING]
}

GET_URL_TYPE_MAPPING = {
    "Record Template": "Record",
    "Application": "Undefined",
    "Role Template": "Role",
    "Process Template": "Process",
    "Organizational Structure Template": "OrgStructure"
}

def remove_values(
    obj: Any,
    exclude_values: Set[Any] = None
) -> Any:
    """
    Recursively remove specified values from dicts/lists.
    
    Args:
        obj: The object to clean (dict, list, or any other type)
        exclude_values: Set of values to remove (default: {None, ""})
        
    Returns:
        Cleaned object
    """
    if exclude_values is None:
        exclude_values = {None, ""}

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            cleaned_v = remove_values(v, exclude_values)
            # Safe comparison: only check membership if cleaned_v is hashable
            # For unhashable types (dict, list), we assume they should be kept
            try:
                if cleaned_v not in exclude_values:
                    result[k] = cleaned_v
            except TypeError:
                # cleaned_v is unhashable (dict, list, etc.), so keep it
                result[k] = cleaned_v
        return result

    if isinstance(obj, list):
        result = []
        for v in obj:
            cleaned_v = remove_values(v, exclude_values)
            try:
                if cleaned_v not in exclude_values:
                    result.append(cleaned_v)
            except TypeError:
                # cleaned_v is unhashable (dict, list, etc.), so keep it
                result.append(cleaned_v)
        return result

    return obj

def _set_input_mask(display_format: str) -> str:
    # Setting validation mask via display format
    input_mask_mapping: Dict[str, str] = {
        "PlainText": None,
        "MarkedText": None,
        "HtmlText": None,
        "LicensePlateNumberRuMask":"([АВЕКМНОРСТУХавекмнорстух]{1}[0-9]{3}[АВЕКМНОРСТУХавекмнорстух]{2} [0-9]{3})",
        "IndexRuMask": "([0-9]{6})",
        "PassportRuMask": "([0-9]{4} [0-9]{6})",
        "INNMask": "([0-9]{10})",
        "OGRNMask": "([0-9]{13})",
        "IndividualINNMask": "([0-9]{12})",
        "PhoneRuMask": "(\\+7 \\([0-9]{3}\\) [0-9]{3}-[0-9]{2}-[0-9]{2})",
        "EmailMask": "^(([a-zа-яё0-9_-]+\\.)*[a-zа-яё0-9_-]+@[a-zа-яё0-9-]+(\\.[a-zа-яё0-9-]+)*\\.[a-zа-яё]{2,6})?$",
        "CustomMask": None
    }

    return input_mask_mapping.get(display_format, None)

def execute_get_operation(
    result_model: Type[BaseModel],
    endpoint: str
) -> Dict[str, Any]:
    """
    :param result_model: Pydantic-модель для валидации финального результата (должна иметь поля: success, status_code, data, error)
    :return: Валидированный результат в виде dict (model_dump)
    """

    stack = inspect.stack()
    caller_frame = stack[1]
    caller_name = caller_frame.function

    result = requests_._get_request(endpoint)

    if not result.get('success', False):
        adapted = {
            "success": result.get("success", False),
            "status_code": result.get("status_code"),
            "data": None,
            "error": result.get("error")
        }
        return result_model(**adapted).model_dump()

    # Извлекаем тело ответа
    raw_response = result.get('raw_response')
    if raw_response is None:
        adapted = {
            "success": False,
            "status_code": result.get("status_code"),
            "data": None,
            "error": "No response data received from server"
        }
        return result_model(**adapted).model_dump()

    # Проверяем структуру
    if not isinstance(raw_response, dict) or 'response' not in raw_response:
        adapted = {
            "success": False,
            "status_code": result.get("status_code"),
            "data": None,
            "error": "Unexpected response structure from server"
        }
        return result_model(**adapted).model_dump()

    # Копируем данные, чтобы не мутировать оригинал
    data = raw_response['response'].copy() if isinstance(raw_response['response'], dict) else raw_response['response']

    if isinstance(data, dict):
        # Определяем тип атрибута
        data = process_data(data, f"{caller_name}")

    # Формируем финальный результат
    final_result = {
        "success": True,
        "status_code": result["status_code"],
        "error": None
    }

    final_result = {**data, **final_result}

    # Валидируем и возвращаем
    validated = result_model(**final_result)
    return validated.model_dump()

def execute_edit_or_create_operation(
    request_body: Dict[str, Any],
    operation: str,
    endpoint: str,
    result_model: Type[BaseModel]
) -> Dict[str, Any]:
    """
    Выполняет операцию (create/edit) через API.
    Возвращает словарь с результатом.
    """
    # Убираем None-значения
    request_body = remove_values(request_body)

    try:
        if operation == "create":
            result = requests_._post_request(
                request_body, 
                endpoint
            )
        elif operation == "edit":
            result = requests_._put_request(
                request_body, 
                endpoint
            )
        else:
            result = {
                "success": False,
                "error": f"No such operation: {operation}. Available operations: create, edit",
                "status_code": 400
            }

    except Exception as e:
        result = {
            "success": False,
            "error": f"Tool execution failed: {str(e)}",
            "status_code": 500
        }

    # Гарантируем, что result — это dict
    if not isinstance(result, dict):
        result = {
            "success": False,
            "error": f"Unexpected result type: {type(result)}",
            "status_code": 500
        }

    # Добавляем префикс к ошибке, если операция неуспешна
    if not result.get("success", False) and result.get("error"):
        error_info = result.get("error", "")
        result["error"] = f"API operation failed: {error_info}"

    validated = result_model(**result)
    return validated.model_dump()

def process_data(
    data: Dict[str, Any],
    caller_name: str
) -> Dict[str, Any]:

    if isinstance(data, dict):
        # Определяем тип атрибута
        if data.get("globalAlias"):
            entity_type = data.get("globalAlias")
            entity_type = entity_type.get("type")
            if entity_type == "Attribute":
                type = data.get("type")
            else:
                type = entity_type
        else:
            type = "Application"
            entity_type = "Application"
        keys_to_remove = KEYS_TO_REMOVE_MAPPING.get(type, []) # по умолчанию - пустой список
        # Удаляем ненужные ключи (если это словарь)
        if keys_to_remove is not None:
            for key in keys_to_remove:
                data.pop(key, None)

        # Обрабатываем globalAlias: вытаскиваем owner и alias, удаляем globalAlias и type
        if "globalAlias" in data:
            global_alias = data.pop("globalAlias", {})
            if isinstance(global_alias, dict):
                # Добавляем owner и alias в корень, если они есть
                new_items = {}
                if "owner" in global_alias:
                    new_items["owner"] = global_alias["owner"]
                if "alias" in global_alias:
                    new_items["alias"] = global_alias["alias"]
                # type игнорируется и не добавляется

                # Пересоздаём словарь: сначала новые ключи, потом остальные
                data = {**new_items, **data}

        # Обрабатываем instanceGlobalAlias: вытаскиваем owner и alias, удаляем instanceGlobalAlias и type
        if "instanceGlobalAlias" in data:
            instance_global_alias = data.pop("instanceGlobalAlias", {})
            if isinstance(instance_global_alias, dict):
                # Добавляем owner и alias в корень, если они есть
                if "owner" in instance_global_alias:
                    data["instanceAlias"] = instance_global_alias["owner"]
                    data["instanceAttributeAlias"] = instance_global_alias["alias"]
                else:
                    if "alias" in instance_global_alias:
                        data["instanceAlias"] = instance_global_alias["alias"]
                # type игнорируется и не добавляется

        # Специальная обработка variants - преобразуем структуру каждого элемента
        if "variants" in data:
            processed_variants = []
            for variant in data["variants"]:
                if not isinstance(variant, dict):
                    continue # пропускаем, если не словарь

                # Извлекаем alias.alias
                alias_value = ""
                if "alias" in variant and isinstance(variant["alias"], dict):
                    alias_value = variant["alias"].get("alias", "")

                # Извлекаем переводы из name
                name_data = variant.get("name", {}) if isinstance(variant.get("name"), dict) else {}
                en_name = name_data.get("en", "")
                ru_name = name_data.get("ru", "")
                de_name = name_data.get("de", "")

                # Цвет
                color = variant.get("color", "")

                # Формируем новый элемент
                processed_variants.append({
                    "System name": alias_value,
                    "English name": en_name,
                    "Russian name": ru_name,
                    "German name": de_name,
                    "Color": color
                })

                # Заменяем старый variants на обрботанный
                data["variants"] = processed_variants

        data = rename_data(data, f"{caller_name}", entity_type, type)
    return data

def rename_data(
    data: Dict[str, Any],
    caller_name: str,
    entity_type: str,
    type:str
) -> Dict[str, Any]:

    if isinstance(data, dict):
        # Определяем тип атрибута
        renamed_data = {}

        # Добавляем описание типа атрибута как первый элемент
        model_description = None
        model_response = None
        for key, value in ENTITY_TYPE_MAPPING.items():
            if caller_name.__contains__(key):
                model_description = value[0]
                model_response = value[1]
                break
        if model_description is not None:
            if entity_type and entity_type in model_description:
                renamed_data[f"{entity_type} type description"] = model_description[type]
        if model_response is not None:
            for key, value in data.items():
                # Если ключ есть в маппинге - используем новое имя, иначе оставляем как есть
                new_key = model_response.get(key, key)
                renamed_data[new_key] = value
            data = renamed_data

    return data

def execute_list_operation(
    response_data: Dict[str, Any],
    result_model: Type[BaseModel]
) -> Dict[str, Any]:
    """
    :param request_result: Результат вызова _get_request(...)
    :return: Валидированный результат в виде dict (model_dump)
    """

    stack = inspect.stack()
    caller_frame = stack[1]
    caller_name = caller_frame.function

    if not response_data.get('success', False):
        adapted = {
            "success": response_data.get("success", False),
            "status_code": response_data.get("status_code"),
            "data": None,
            "error": response_data.get("error")
        }
        return result_model(**adapted).model_dump()

    # Извлекаем тело ответа
    raw_response = response_data.get('raw_response')
    if raw_response is None:
        adapted = {
            "success": False,
            "status_code": response_data.get("status_code"),
            "data": None,
            "error": "No response data received from server"
        }
        return result_model(**adapted).model_dump()

    # Проверяем структуру
    if not isinstance(raw_response, dict) or 'response' not in raw_response:
        adapted = {
            "success": False,
            "status_code": response_data.get("status_code"),
            "data": None,
            "error": "Unexpected response structure from server"
        }
        return result_model(**adapted).model_dump()

    # Копируем данные, чтобы не мутировать оригинал
    data = raw_response['response'].copy() if isinstance(raw_response['response'], list) else raw_response['response']

    for i, item in enumerate(data):
        if isinstance(item, dict):
            data[i] = process_data(item, f"{caller_name}")

    # Формируем финальный результат
    final_result = {
        "success": True,
        "status_code": response_data["status_code"],
        "error": None
    }

    final_result["data"] = data

    # Валидируем и возвращаем
    validated = result_model(**final_result)
    return validated.model_dump()