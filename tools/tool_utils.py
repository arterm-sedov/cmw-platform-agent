from typing import Any, Dict, List, Optional, Literal, Type
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator
from . import requests_ as requests_
from .models import (
    AttributeResult, 
    CommonAttributeFields, 
    CommonGetAttributeFields,
    normalize_operation_archive_unarchive
)

# Common constants
ATTRIBUTE_ENDPOINT = "webapi/Attribute"
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
    "Account": ['isUnique', 'isIndexed', 'isMandatory', 'isOwnership', 'imageColorType', 'imagePreserveAspectRatio']
}
ATTRIBUTE_RESPONE_MAPPING = {
    "owner": "Template System Name",
    "alias": "System Name",
    "type": "Type",
    "format": "Display format",
    "name": "Name",
    "description": "Description",
    "isSystem": "Is System",
    "isDisabled": "In Archive",
    "isUnique": "Control Uniqueness",
    "isIndexed": "Use to search records",
    "isTracked": "Write changes to the log",
    "isDigitGrouping": "Group digits numbers",
    "isMultiValue": "Store multiple values",
    "isTitle": "Use as record title",
    "isCalculated": "Calculate value",
    "isReadonly": "Is Readonly",
    "expression": "Expression for calculation",
    "fileFormat": "File extensions filter",
    "uriChemeFormats": "Uri chemes filter",
    "instanceAlias": "Related template system name",
    "instanceAttributeAlias": "Related attribute system name",
    "imageColorType": "Rendering color mode",
    "imageWidth": "Image Width",
    "imageHeight": "Image Height",
    "imagePreserveAspectRatio": "Save image aspect ration",
    "imageXResolution": "X-axis image resolution",
    "imageYResolution": "Y-axis image resolution",
    "decimalPlaces": "Number decimal places",
    "validationMaskRegex": "Custom mask",
    "variants": "Enum values",
    "linkedRecordTemplate": "Related template Id"
}

def remove_nones(obj: Any) -> Any:
    """
    Recursively remove None values from dicts/lists to keep payload minimal and consistent with Platform expectations.
    
    This utility function is used across all tool modules to clean up data before sending to the API,
    ensuring that only meaningful data is transmitted and reducing payload size.
    
    Args:
        obj: The object to clean (dict, list, or any other type)
        
    Returns:
        Cleaned object with None values removed
    """
    if isinstance(obj, dict):
        return {k: remove_nones(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [remove_nones(v) for v in obj if v is not None]
    return obj

def process_attribute_response(
    request_result: Dict[str, Any],
    result_model: Type[BaseModel],
    response_mapping: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Универсальная функция для постобработки ответа от _get_request.
    Извлекает, очищает и валидирует данные атрибута.

    :param request_result: Результат вызова _get_request(...)
    :param keys_to_remove: Список ключей, которые нужно удалить из response
    :param result_model: Pydantic-модель для валидации финального результата (должна иметь поля: success, status_code, data, error)
    :return: Валидированный результат в виде dict (model_dump)
    """
    if not request_result.get('success', False):
        adapted = {
            "success": request_result.get("success", False),
            "status_code": request_result.get("status_code"),
            "data": None,
            "error": request_result.get("error")
        }
        return result_model(**adapted).model_dump()

    # Извлекаем тело ответа
    raw_response = request_result.get('raw_response')
    if raw_response is None:
        adapted = {
            "success": False,
            "status_code": request_result.get("status_code"),
            "data": None,
            "error": "No response data received from server"
        }
        return result_model(**adapted).model_dump()

    # Проверяем структуру
    if not isinstance(raw_response, dict) or 'response' not in raw_response:
        adapted = {
            "success": False,
            "status_code": request_result.get("status_code"),
            "data": None,
            "error": "Unexpected response structure from server"
        }
        return result_model(**adapted).model_dump()

    # Копируем данные, чтобы не мутировать оригинал
    attribute_data = raw_response['response'].copy() if isinstance(raw_response['response'], dict) else raw_response['response']

    # Работаем только если attribute_data - словарь
    if isinstance(attribute_data, dict):
        # Определяем тип атрибута
        attr_type = attribute_data.get("type")
        keys_to_remove = KEYS_TO_REMOVE_MAPPING.get(attr_type, []) # по умолчанию - пустой список
        # Удаляем ненужные ключи (если это словарь)
        for key in keys_to_remove:
            attribute_data.pop(key, None)

    # Обрабатываем globalAlias: вытаскиваем owner и alias, удаляем globalAlias и type
    if "globalAlias" in attribute_data:
        global_alias = attribute_data.pop("globalAlias", {})
        if isinstance(global_alias, dict):
            # Добавляем owner и alias в корень, если они есть
            new_items = {}
            if "owner" in global_alias:
                new_items["owner"] = global_alias["owner"]
            if "alias" in global_alias:
                new_items["alias"] = global_alias["alias"]
            # type игнорируется и не добавляется

            # Пересоздаём словарь: сначала новые ключи, потом остальные
            attribute_data = {**new_items, **attribute_data}

    # Обрабатываем instanceGlobalAlias: вытаскиваем owner и alias, удаляем instanceGlobalAlias и type
    if "instanceGlobalAlias" in attribute_data:
        instance_global_alias = attribute_data.pop("instanceGlobalAlias", {})
        if isinstance(instance_global_alias, dict):
            # Добавляем owner и alias в корень, если они есть
            if "owner" in global_alias:
                attribute_data["instanceAlias"] = instance_global_alias["owner"]
                attribute_data["instanceAttributeAlias"] = instance_global_alias["alias"]
            else:
                attribute_data["instanceAlias"] = instance_global_alias["alias"]
            # type игнорируется и не добавляется

    # Специальная обработка variants - преобразуем структуру каждого элемента
    if isinstance(attribute_data, dict) and "variants" in attribute_data:
        processed_variants = []
        for variant in attribute_data["variants"]:
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

            # Формуруем новый элемент
            processed_variants.append({
                "System name": alias_value,
                "English name": en_name,
                "Russian name": ru_name,
                "Deutsche name": de_name,
                "Color": color
            })

            # Заменяем старый variants на обрботанный
            attribute_data["variants"] = processed_variants

    # Переименовываем ключи согласно response_mapping
    if isinstance(attribute_data, dict):
        renamed_data = {}
        for key, value in attribute_data.items():
            # Если ключ есть в маппинге - используем новое имя, иначе оставляем как есть
            new_key = response_mapping.get(key, key)
            renamed_data[new_key] = value
        attribute_data = renamed_data

    # Формируем финальный результат
    final_result = {
        "success": True,
        "status_code": request_result["status_code"],
        "data": attribute_data,
        "error": None
    }

    # Валидируем и возвращаем
    validated = result_model(**final_result)
    return validated.model_dump()