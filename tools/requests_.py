"""
HTTP Requests Module with Pydantic Integration
==============================================

This module provides HTTP request functionality with comprehensive Pydantic validation
for the CMW Platform agent. It handles all HTTP operations (GET, POST, PUT, DELETE)
with type safety, error prevention, and consistent response handling.

Key Features:
- Pydantic validation for all HTTP responses
- Type safety to prevent NoneType concatenation errors
- Consistent error handling across all request types
- Configuration validation with proper error messages
- Backward compatibility with existing code

HTTP Methods:
- _get_request(): GET requests with JSON/text response handling
- _post_request(): POST requests with request body validation
- _put_request(): PUT requests with request body validation
- _delete_request(): DELETE requests with proper error handling

Configuration:
- _load_server_config(): Loads and validates server configuration
- _basic_headers(): Generates authentication headers

Author: AI Assistant
Date: January 18, 2025
Version: 1.0
"""

import base64
import json
import os
from typing import Any

from dotenv import load_dotenv
import requests

from .requests_models import HTTPResponse, RequestConfig

# Import session-aware config access
try:
    from agent_ng.session_manager import get_current_session_id, get_session_config
except Exception:
    get_config_for_session = None
    get_current_session_id = None


def _get_config_from_tab() -> dict[str, str] | None:
    """
    Get configuration values from the config tab's browser state.

    This function attempts to access the config tab instance from the main app
    and retrieve the current configuration values from the browser state.

    Returns:
        Dict with config values (url, username, password) or None if not available
    """
    # Prefer session-aware config store if available
    if get_session_config and get_current_session_id:
        session_id = get_current_session_id()
        try:
            return get_session_config(session_id)
        except Exception:
            return None
    return None


def _load_server_config() -> RequestConfig:
    """
    Load and validate server configuration based on CMW_USE_DOTENV flag.

    Configuration source behavior:
    - CMW_USE_DOTENV=true (default): Load from .env using python-dotenv
    - CMW_USE_DOTENV=false: Read values from ConfigTab (BrowserState snapshot)

    Required configuration values:
      - CMW_BASE_URL (required)
      - CMW_LOGIN (required)
      - CMW_PASSWORD (required)
      - CMW_TIMEOUT (optional; defaults to 30 if missing/invalid)

    The resulting values are validated via the `RequestConfig` Pydantic model,
    ensuring required parameters are present and correct types are enforced.

    Returns:
        RequestConfig: Validated configuration object with all required fields.

    Raises:
        RuntimeError: If required environment variables are missing or
            configuration validation fails.

    Environment variables example (PowerShell):
        $env:CMW_BASE_URL = "https://platform.comindware.com"
        $env:CMW_LOGIN = "your_username"
        $env:CMW_PASSWORD = "your_password"
        $env:CMW_TIMEOUT = "30"

    Example:
        >>> config = _load_server_config()
        >>> print(config.base_url)
        >>> print(config.timeout)
    """
    # Determine configuration source based on CMW_USE_DOTENV flag
    use_dotenv_flag = os.environ.get("CMW_USE_DOTENV", "true").lower() in (
        "1",
        "true",
        "yes",
    )

    if not use_dotenv_flag:
        # CMW_USE_DOTENV=false: Use config tab (BrowserState snapshot via server cache)
        config_values = _get_config_from_tab()
        if isinstance(config_values, dict) and config_values:
            base_url_env = config_values.get("url", "").strip()
            login_env = config_values.get("username", "").strip()
            password_env = config_values.get("password", "").strip()
    else:
        # CMW_USE_DOTENV=true: Load from .env file
        load_dotenv()
        base_url_env = os.environ.get("CMW_BASE_URL", "").strip()
        login_env = os.environ.get("CMW_LOGIN", "").strip()
        password_env = os.environ.get("CMW_PASSWORD", "").strip()

    # Get timeout from environment (always from env, regardless of source)
    timeout_env = os.environ.get("CMW_TIMEOUT", "").strip()

    # Validate required variables
    missing = []
    if not base_url_env:
        missing.append("CMW_BASE_URL")
    if not login_env:
        missing.append("CMW_LOGIN")
    if not password_env:
        missing.append("CMW_PASSWORD")
    if missing:
        missing_msg = "Missing required environment variables: " + ", ".join(missing)
        raise RuntimeError(missing_msg)

    # Parse timeout with default
    try:
        timeout_val = int(timeout_env) if timeout_env else 30
    except ValueError:
        timeout_val = 30

    # Build validated config
    try:
        return RequestConfig(
            base_url=base_url_env,
            login=login_env,
            password=password_env,
            timeout=timeout_val,
        )
    except Exception as e:
        err_msg = "Invalid server configuration"
        raise RuntimeError(err_msg) from e


def _basic_headers() -> dict[str, str]:
    """
    Generate basic authentication headers using validated configuration.

    This function creates HTTP headers for basic authentication using the
    validated server configuration. It encodes the credentials in base64
    and sets appropriate content type headers for JSON communication.

    Returns:
        Dict[str, str]: Dictionary containing HTTP headers for authentication
            - Authorization: Basic auth header with encoded credentials
            - Content-Type: application/json
            - Accept: application/json

    Example:
        >>> headers = _basic_headers()
        >>> print(headers["Authorization"])  # "Basic dXNlcjpwYXNzd29yZA=="
        >>> print(headers["Content-Type"])   # "application/json"
    """
    cfg = _load_server_config()

    # Ensure credentials are not None to prevent concatenation errors
    login = cfg.login or ""
    password = cfg.password or ""

    # Encode credentials in base64 for basic authentication
    credentials = base64.b64encode(f"{login}:{password}".encode("ascii")).decode(
        "ascii"
    )
    return {
        "Authorization": f"Basic {credentials}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _check_response_for_errors(response_text: str) -> str | None:
    """
    Check if the response body contains an error even when HTTP status is 200.

    This function performs additional validation on API responses to detect
    API-level errors that might be returned with HTTP 200 status codes.
    It's particularly useful for Comindware Platform API responses that may
    indicate errors through the response body rather than HTTP status codes.

    Args:
        response_text (str): The raw response text from the API

    Returns:
        Optional[str]: Error message if found, None if response is successful

    Error Detection Logic:
        - Parses response as JSON
        - Checks if response contains "success": false
        - Returns formatted error message if error detected
        - Returns None if response is valid or not JSON

    Example:
        >>> # Valid response
        >>> _check_response_for_errors('{"success": true, "data": "example"}')
        None

        >>> # Error response
        >>> _check_response_for_errors('{"success": false, "error": "Invalid request"}')
        '{"success": false, "error": "Invalid request"}'
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


def _post_request(request_body: dict[str, Any], endpoint: str) -> dict[str, Any]:
    """
    Make a POST request with Pydantic validation while keeping backward compatibility.
    Returns a dict for backward compatibility with existing code.
    """
    cfg = _load_server_config()
    url = f"{cfg.base_url}/{endpoint}"
    headers = _basic_headers()

    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(request_body), timeout=cfg.timeout
        )
        response.raise_for_status()

        # Parse response with proper error handling
        try:
            raw_response = response.json()
        except (json.JSONDecodeError, ValueError):
            raw_response = response.text

        # Create Pydantic model for validation
        http_response = HTTPResponse(
            success=response.status_code == 200,
            status_code=response.status_code,
            raw_response=raw_response,
            error=None,
            base_url=url,
        )

        # Additional API-level error checking for successful HTTP responses
        if http_response.success:
            api_error = _check_response_for_errors(response.text)
            if api_error:
                http_response.error = api_error
                http_response.success = False
    except requests.exceptions.Timeout as e:
        error_response = HTTPResponse(
            success=False,
            status_code=408,  # Request Timeout
            raw_response=None,
            error=f"Request timeout: {str(e)}",
            base_url=url,
        )
        result = error_response.model_dump()
        result["body"] = request_body
        return result
    except requests.exceptions.ConnectionError as e:
        error_response = HTTPResponse(
            success=False,
            status_code=503,  # Service Unavailable
            raw_response=None,
            error=f"Connection error: {str(e)}",
            base_url=url,
        )
        result = error_response.model_dump()
        result["body"] = request_body
        return result
    except requests.exceptions.RequestException as e:
        # Общий обработчик для всех остальных сетевых ошибок
        error_response = HTTPResponse(
            success=False,
            status_code=500,  # Internal Server Error
            raw_response=None,
            error=f"Request failed: {str(e)}",
            base_url=url,
        )
        result = error_response.model_dump()
        result["body"] = request_body
        return result
    else:
        # Return dict for backward compatibility (including body field)
        result = http_response.model_dump()
        result["body"] = request_body
        return result


def _put_request(request_body: dict[str, Any], endpoint: str) -> dict[str, Any]:
    """
    Make a PUT request with Pydantic validation while keeping backward compatibility.
    Returns a dict for backward compatibility with existing code.
    """
    cfg = _load_server_config()
    url = f"{cfg.base_url}/{endpoint}"
    headers = _basic_headers()

    try:
        response = requests.put(
            url, headers=headers, data=json.dumps(request_body), timeout=cfg.timeout
        )
        response.raise_for_status()

        # Parse response with proper error handling
        try:
            raw_response = response.json()
        except (json.JSONDecodeError, ValueError):
            raw_response = response.text

        # Create Pydantic model for validation
        http_response = HTTPResponse(
            success=response.status_code == 200,
            status_code=response.status_code,
            raw_response=raw_response,
            error=None,
            base_url=url,
        )

        # Additional API-level error checking for successful HTTP responses
        if http_response.success:
            api_error = _check_response_for_errors(response.text)
            if api_error:
                http_response.error = api_error
                http_response.success = False
    except requests.exceptions.Timeout as e:
        error_response = HTTPResponse(
            success=False,
            status_code=408,  # Request Timeout
            raw_response=None,
            error=f"Request timeout: {str(e)}",
            base_url=url,
        )
        result = error_response.model_dump()
        result["body"] = request_body
        return result
    except requests.exceptions.ConnectionError as e:
        error_response = HTTPResponse(
            success=False,
            status_code=503,  # Service Unavailable
            raw_response=None,
            error=f"Connection error: {str(e)}",
            base_url=url,
        )
        result = error_response.model_dump()
        result["body"] = request_body
        return result
    except requests.exceptions.RequestException as e:
        # Общий обработчик для всех остальных сетевых ошибок
        error_response = HTTPResponse(
            success=False,
            status_code=500,  # Internal Server Error
            raw_response=None,
            error=f"Request failed: {str(e)}",
            base_url=url,
        )
        result = error_response.model_dump()
        result["body"] = request_body
        return result
    else:
        # Return dict for backward compatibility (including body field)
        result = http_response.model_dump()
        result["body"] = request_body
        return result


def _get_request(endpoint: str) -> dict[str, Any]:
    """
    Make a GET request with Pydantic validation while keeping backward compatibility.
    Returns a dict for backward compatibility with existing code.
    """
    cfg = _load_server_config()
    url = f"{cfg.base_url}/{endpoint}"
    headers = _basic_headers()

    try:
        response = requests.get(url, headers=headers, timeout=cfg.timeout)
        response.raise_for_status()

        # Parse JSON response with proper error handling
        try:
            raw_response = response.json()
            error_msg = None
        except (json.JSONDecodeError, ValueError) as e:
            raw_response = response.text
            error_msg = f"Failed to parse JSON response: {str(e)}"

        # Create Pydantic model for validation
        http_response = HTTPResponse(
            success=response.status_code == 200,
            status_code=response.status_code,
            raw_response=raw_response,
            error=error_msg,
            base_url=url,
        )

        # Additional API-level error checking for successful HTTP responses
        if http_response.success and isinstance(raw_response, dict):
            api_error = _check_response_for_errors(
                json.dumps(raw_response, ensure_ascii=False)
            )
            if api_error:
                http_response.error = api_error
                http_response.success = False

        # Return dict for backward compatibility
        return http_response.model_dump()

    except requests.exceptions.Timeout as e:
        error_response = HTTPResponse(
            success=False,
            status_code=408,  # Request Timeout
            raw_response=None,
            error=f"Request timeout: {str(e)}",
            base_url=url,
        )
        return error_response.model_dump()
    except requests.exceptions.ConnectionError as e:
        error_response = HTTPResponse(
            success=False,
            status_code=503,  # Service Unavailable
            raw_response=None,
            error=f"Connection error: {str(e)}",
            base_url=url,
        )
        return error_response.model_dump()
    except requests.exceptions.RequestException as e:
        # Общий обработчик для всех остальных сетевых ошибок
        error_response = HTTPResponse(
            success=False,
            status_code=500,  # Internal Server Error
            raw_response=None,
            error=f"Request failed: {str(e)}",
            base_url=url,
        )
        return error_response.model_dump()


def _delete_request(endpoint: str) -> dict[str, Any]:
    """
    Make a DELETE request with Pydantic validation while keeping backward compatibility.
    Returns a dict for backward compatibility with existing code.
    """
    cfg = _load_server_config()
    url = f"{cfg.base_url}/{endpoint}"
    headers = _basic_headers()

    try:
        response = requests.delete(url, headers=headers, timeout=cfg.timeout)
        response.raise_for_status()

        # Create Pydantic model for validation
        http_response = HTTPResponse(
            success=response.status_code == 200,
            status_code=response.status_code,
            raw_response=None,  # DELETE requests typically don't have response body
            error=None,
            base_url=url,
        )

        # Return dict for backward compatibility
        return http_response.model_dump()

    except requests.exceptions.Timeout as e:
        error_response = HTTPResponse(
            success=False,
            status_code=408,  # Request Timeout
            raw_response=None,
            error=f"Request timeout: {str(e)}",
            base_url=url,
        )
        return error_response.model_dump()
    except requests.exceptions.ConnectionError as e:
        error_response = HTTPResponse(
            success=False,
            status_code=503,  # Service Unavailable
            raw_response=None,
            error=f"Connection error: {str(e)}",
            base_url=url,
        )
        return error_response.model_dump()
    except requests.exceptions.RequestException as e:
        # Общий обработчик для всех остальных сетевых ошибок
        error_response = HTTPResponse(
            success=False,
            status_code=500,  # Internal Server Error
            raw_response=None,
            error=f"Request failed: {str(e)}",
            base_url=url,
        )
        return error_response.model_dump()
