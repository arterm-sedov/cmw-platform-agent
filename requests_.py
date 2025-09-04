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

def _check_response_for_errors(response_text: str) -> Optional[str]:
    """
    Check if the response body contains an error even when HTTP status is 200.
    
    Args:
        response_text: The raw response text from the API
        
    Returns:
        Error message if an error is found, None if response is successful
    """
    try:
        response_json = json.loads(response_text)
        if isinstance(response_json, dict) and response_json.get("success") is False:
            # API returned success: false, so this is actually an error
            return json.dumps(response_json, ensure_ascii=False)
    except (json.JSONDecodeError, AttributeError):
        # Response is not JSON or doesn't have expected structure, treat as success
        pass
    return None

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
        # Check if the response body contains an error
        error_message = _check_response_for_errors(response.text)
        if error_message:
            result["error"] = error_message
            return result
        
        result.update({"success": True})
        return result

    # Known error pattern: 500 with JSON body describing an issue (e.g., alias already exists)
    try:
        err_json = response.json()
        result["error"] = json.dumps(err_json, ensure_ascii=False)
    except Exception:
        result["error"] = response.text or f"HTTP {response.status_code}"
    return result

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
        # Check if the response body contains an error
        error_message = _check_response_for_errors(response.text)
        if error_message:
            result["error"] = error_message
            return result
        
        result.update({"success": True})
        return result

    # Known error pattern: 500 with JSON body describing an issue (e.g., alias already exists)
    try:
        err_json = response.json()
        result["error"] = json.dumps(err_json, ensure_ascii=False)
    except Exception:
        result["error"] = response.text or f"HTTP {response.status_code}"
    return result

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
    return result