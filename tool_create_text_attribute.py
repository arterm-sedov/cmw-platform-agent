from typing import Any, Dict, List, Optional
import json
from typing import Optional
from urllib import request
from numpy import isin
import requests
from sqlalchemy.engine.base import ExceptionContextImpl
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
    solutionAlias: str,
    owner: str,
    alias: str,
    format: str,
    name: str,
    type: Optional[str] = "Undefined",
    attribute_type: str = "String",
    description: Optional[str] = None,
    isUnique: Optional[bool] = False,
    isTitle: Optional[bool] = False,
    validationMaskRegex: Optional[str] = None
) -> Dict[str, Any]:
    r"""
    Create an text attribute in Comindware Platform via API method /webapi/Attribute/solutionAlias.

    Tool description:
        - Endpoint: webapi/Attribute/solutionAlias (POST)
        - Auth: Basic ()
        - Returns: dict with keys:
            - succes: bool
            - status_code: int      -> HTTP status code
            - raw_response: str     -> raw response text for auditing
            - error: str | None     -> error message if any
    
    Parameters:
        - solutionAlias (string): Solution system name where the container (record template) where the attribute is created.
        - owner (string): Container (record template) system name where the attribute is created.
        - alias (string): Unique system name of the attribute.
        - attribute_type (string): Attribute type ("String").
        - format (string): Display format name.
        - name (string): Human-readable name of the attribute.
        - description (string | none): Human-readable description of the attribute.
        - isUnique (bool | False): A parameter that displays whether the values of this attribute should be unique.
        - isTitle (bool | False): A parameter that displays whether the values of this attribute should be displayed as a title.
        - validationMaskRegex (str | None): A special fill mask for specific display formats.

    Notes:
        - This tool only calls Attribute (POST) and does not perform subsequent operations.
        
    Displaying formats mapping (API -> UI):
        - PlainText -> 
        - MarkedText -> 
        - HtmlText -> 
        - LicensePlateNumberRuMask -> 
        - IndexRuMask -> 
        - PassportRuMask -> 
        - INNMask -> 
        - OGRNMask -> 
        - IndividualINNMask -> 
        - PhoneRuMask -> 
        - EmailMask -> 
        - CustomMask -> 
    
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
    url = f"{base_url}/webapi/Attribute/{solutionAlias}"
    headers = _basic_headers()

    request_body: Dict[str, Any] = {
        "globalAlias": {
            "type": type,
            "owner": owner,
            "alias": alias
        },
        "type": attribute_type,
        "format": format,
        "name": name,
        "description": description,
        "isUnique": isUnique,
        "isTitle": isTitle,
        "validationMaskRegex": validationMaskRegex
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


solutionAlias = "Malatik"
owner = "Test"
alias = "Test3"
format = "PlainText"
name = "Test3"
type = "Undefined",
attribute_type = "String",
description = None,
isUnique = False,
isTitle = True,
validationMaskRegex = None
results = create_text_attribute.invoke({
    "solutionAlias": "Malatik",
    "owner": "Test",
    "alias": "Test3",
    "format": "PlainText",
    "name": "Test3",
    "type": "Undefined",
    "attribute_type": "String",
    "description": None,
    "isUnique": False,
    "isTitle": True,
    "validationMaskRegex": None})
print(results)