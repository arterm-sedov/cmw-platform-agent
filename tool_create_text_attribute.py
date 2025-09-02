from typing import Any, Dict, List, Optional
import json
from typing import Optional
import requests
import yaml
import base64
from langchain.tools import tool

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

@tool("create_text_attribute", return_direct=False)
def create_text_attribute(
    name: str,
    system_name: str,
    application_system_name: str,
    template_system_name: str,
    display_format: str,
    description: Optional[str] = None,
    custom_mask: Optional[str] = None,
    control_uniqueness: Optional[bool] = False,
    use_as_record_title: Optional[bool] = False,
    attribute_type: Optional[str] = "String"
) -> Dict[str, Any]:
    r"""
    Create a text attribute.

    - Returns: dict with keys:
        - succes: bool
        - status_code: int      -> HTTP status code
        - raw_response: str     -> raw response text for auditing
        - error: str | None     -> error message if any
    
    Parameters:
        - name (string): Human-readable name (Название) of the attribute.
        - system_name (string): Unique system name (Системное имя) of the attribute.
        - description (string | None): Human-readable description (Описание) of the attribute.
        - application_system_name (string): System name (Системное имя приложения) of the application with the template where the attribute is created.
        - template_system_name (string): System name of the template (Системное имя шаблона) where the attribute is created.
        - display_format (string): Attribute display format (Формат отображения). Value mapping to Russian format names:
            {
                "PlainText": "Простой текст",
                "MarkedText": "Размеченный текст",
                "HtmlText": "HTML-текст",
                "LicensePlateNumberRuMask": "Регистрация номера ТС (РФ)",
                "IndexRuMask": "Индекс (РФ)",
                "PassportRuMask": "Паспорт (РФ)",
                "INNMask": "ИНН юрлица",
                "OGRNMask": "ОГРН",
                "IndividualINNMask": "ИНН физлица",
                "PhoneRuMask": "Телефон (РФ)",
                "EmailMask": "Адрес эл. почты",
                "CustomMask": "Особая маска"
            }
        - custom_mask (str | None): A special formatting mask (Особая маска). Fill only if display_format=CustomMask.
        - control_uniqueness (bool | False): Flag (Контролировать уникальность значения) to control whether the values of this attribute must be unique.
        - use_as_record_title (bool | False): Flag (Использовать как заголовок записей) to control whether the values of this attribute will be displayed as a template record title.
        

    Notes:
        - This tool only calls Attribute (POST) and does not perform subsequent operations.
        
    Validation Masks mapping (Displaying format -> Validation Mask):
        - LicensePlateNumberRuMask -> ([АВЕКМНОРСТУХавекмнорстух]{1}[0-9]{3}[АВЕКМНОРСТУХавекмнорстух]{2} [0-9]{3})
        - IndexRuMask -> r'([0-9]{6})'
        - PassportRuMask -> r'([0-9]{4} [0-9]{6})'
        - INNMask -> r'([0-9]{10})'
        - OGRNMask -> r'([0-9]{13})'
        - IndividualINNMask -> r'([0-9]{12})'
        - PhoneRuMask -> r'(\+7 \([0-9]{3}\) [0-9]{3}-[0-9]{2}-[0-9]{2})'
        - EmailMask -> r'^(([a-zа-яё0-9_-]+\.)*[a-zа-яё0-9_-]+@[a-zа-яё0-9-]+(\.[a-zа-яё0-9-]+)*\.[a-zа-яё]{2,6})?$'
        - CustomMask -> simple validation mask
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
        "validationMaskRegex": custom_mask if display_format == "CustomMask" else None
    }

    # Remove None values
    request_body = _remove_nones(request_body) 

    response = requests.post(
        url,
        headers=headers,
        data=json.dumps(request_body),
        timeout=30
    )

    # Avoid printing sensitive headers

    result: Dict[str, Any] = {
        "success": False,
        "base_url": url,
        "body": request_body,
        "status_code": response.status_code,
        "raw_response": response.text,
        "error": None
    }

    # Success: Platform returns 200 with response body being the created property id (often as quored string)
    if response.status_code == 200:
        result.update({"success": True})
        return result

    # Known error pattern: 500 with JSON body describing an issue (e.g., alias already exists)
    try:
        err_json = response.json()
        result["error"] = json.dumps(err_json, ensure_ascii=False)
    except Exception:
        result["error"] = response.text or f"HTTP {response.status_code}"

    return result


if __name__ == "__main__":
    results = create_text_attribute.invoke({
        "name": "Test3",
        "system_name": "Test5",
        "application_system_name": "Malatik",
        "template_system_name": "Test",
        "display_format": "PlainText",
        "description": None,
        "custom_mask": None,
        "control_uniqueness": False,
        "use_as_record_title": True,
        "attribute_type": "String"
    })
    print(results)