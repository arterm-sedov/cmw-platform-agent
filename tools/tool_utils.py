from typing import Any, Dict, List, Optional, Literal, Type
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator
from . import requests_ as requests_
from tools.models import (
    AttributeResult, 
    CommonAttributeFields, 
    CommonGetAttributeFields,
    normalize_operation_archive_unarchive
)

# Common constants
ATTRIBUTE_ENDPOINT = "webapi/Attribute"

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
    keys_to_remove: List[str],
    result_model: Type[BaseModel]
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

    # Удаляем ненужные ключи (если это словарь)
    if isinstance(attribute_data, dict):
        for key in keys_to_remove:
            attribute_data.pop(key, None)

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