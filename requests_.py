from typing import Any, Dict, List, Optional
import json
import requests
import yaml
import base64

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

def _post_request(request_body: Dict[str, Any], endpoint: str) -> Dict[str, Any]:

    cfg = _load_server_config()
    base_url = cfg.get("base_url")
    url = f"{base_url}/{endpoint}"
    headers = _basic_headers()

    response = requests.post(
        url,
        headers=headers,
        data=json.dumps(request_body),
        timeout=30
    )
    response.raise_for_status()

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

def _put_request(request_body: Dict[str, Any], endpoint: str) -> Dict[str, Any]:

    cfg = _load_server_config()
    base_url = cfg.get("base_url")
    url = f"{base_url}/{endpoint}"
    headers = _basic_headers()

    response = requests.put(
        url,
        headers=headers,
        data=json.dumps(request_body),
        timeout=30
    )
    response.raise_for_status()

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

def _get_request(endpoint: str) -> Dict[str, Any]:

    cfg = _load_server_config()
    base_url = cfg.get("base_url")
    url = f"{base_url}/{endpoint}"
    headers = _basic_headers()

    response = requests.get(
        url,
        headers=headers,
        timeout=30
    )
    response.raise_for_status()

    # Avoid printing sensitive headers
    result: Dict[str, Any] = {
        "success": False,
        "base_url": url,
        "status_code": response.status_code,
        "raw_response": response.json(),
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
    return