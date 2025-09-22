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

from typing import Any, Dict, List, Optional, Union
import json
import requests
import yaml
import base64
from .requests_models import HTTPResponse, APIResponse, RequestConfig

def _load_server_config() -> RequestConfig:
    """
    Load and validate server configuration using Pydantic validation.
    
    This function loads the server configuration from server_config.yml and
    validates it using the RequestConfig Pydantic model. It ensures all
    required configuration parameters are present and valid.
    
    Returns:
        RequestConfig: Validated configuration object with all required fields
        
    Raises:
        RuntimeError: If configuration file cannot be loaded or validation fails
        FileNotFoundError: If server_config.yml file is not found
        
    Configuration File Format:
        The server_config.yml file should contain:
        ```yaml
        base_url: "https://platform.comindware.com"
        login: "your_username"
        password: "your_password"
        timeout: 30  # optional, defaults to 30
        ```
    
    Example:
        >>> config = _load_server_config()
        >>> print(config.base_url)  # "https://platform.comindware.com"
        >>> print(config.timeout)   # 30
    """
    with open("server_config.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
        
    # Validate configuration using Pydantic to ensure all required fields are present
    try:
        return RequestConfig(
            base_url=cfg.get("base_url", ""),
            login=cfg.get("login", ""),
            password=cfg.get("password", ""),
            timeout=cfg.get("timeout", 30)
        )
    except Exception as e:
        raise RuntimeError(f"Invalid server configuration: {e}")

def _basic_headers() -> Dict[str, str]:
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
    # Encode credentials in base64 for basic authentication
    credentials = base64.b64encode(f"{cfg.login}:{cfg.password}".encode("ascii")).decode("ascii")
    return {
        "Authorization": f"Basic {credentials}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

def _check_response_for_errors(response_text: str) -> Optional[str]:
    """
    Check if the response body contains an error even when HTTP status is 200.
    
    This function performs additional validation on API responses to detect
    API-level errors that might be returned with HTTP 200 status codes.
    It's particularly useful for Comindware Platform API responses that may
    indicate errors through the response body rather than HTTP status codes.
    
    Args:
        response_text (str): The raw response text from the API
        
    Returns:
        Optional[str]: Error message if an error is found, None if response is successful
        
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

def _post_request(request_body: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
    """
    Make a POST request with Pydantic validation while maintaining backward compatibility.
    Returns a dict for backward compatibility with existing code.
    """
    cfg = _load_server_config()
    url = f"{cfg.base_url}/{endpoint}"
    headers = _basic_headers()

    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(request_body),
            timeout=cfg.timeout
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
            base_url=url
        )

        # Additional API-level error checking for successful HTTP responses
        if http_response.success:
            api_error = _check_response_for_errors(response.text)
            if api_error:
                http_response.error = api_error
                http_response.success = False

        # Return dict for backward compatibility (including body field)
        result = http_response.model_dump()
        
        result["body"] = request_body

        return result

    except requests.exceptions.Timeout as e:
        error_response = HTTPResponse(
            success=False,
            status_code=408,  # Request Timeout
            raw_response=None,
            error=f"Request timeout: {str(e)}",
            base_url=url
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
            base_url=url
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
            base_url=url
        )
        result = error_response.model_dump()
        result["body"] = request_body
        return result

def _put_request(request_body: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
    """
    Make a PUT request with Pydantic validation while maintaining backward compatibility.
    Returns a dict for backward compatibility with existing code.
    """
    cfg = _load_server_config()
    url = f"{cfg.base_url}/{endpoint}"
    headers = _basic_headers()

    try:
        response = requests.put(
            url,
            headers=headers,
            data=json.dumps(request_body),
            timeout=cfg.timeout
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
            base_url=url
        )

        # Additional API-level error checking for successful HTTP responses
        if http_response.success:
            api_error = _check_response_for_errors(response.text)
            if api_error:
                http_response.error = api_error
                http_response.success = False

        # Return dict for backward compatibility (including body field)
        result = http_response.model_dump()
        result["body"] = request_body
        return result

    except requests.exceptions.Timeout as e:
        error_response = HTTPResponse(
            success=False,
            status_code=408,  # Request Timeout
            raw_response=None,
            error=f"Request timeout: {str(e)}",
            base_url=url
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
            base_url=url
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
            base_url=url
        )
        result = error_response.model_dump()
        result["body"] = request_body
        return result

def _get_request(endpoint: str) -> Dict[str, Any]:
    """
    Make a GET request with Pydantic validation while maintaining backward compatibility.
    Returns a dict for backward compatibility with existing code.
    """
    cfg = _load_server_config()
    url = f"{cfg.base_url}/{endpoint}"
    headers = _basic_headers()

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=cfg.timeout
        )
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
            base_url=url
        )

        # Additional API-level error checking for successful HTTP responses
        if http_response.success and isinstance(raw_response, dict):
            api_error = _check_response_for_errors(json.dumps(raw_response, ensure_ascii=False))
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
            base_url=url
        )
        return error_response.model_dump()
    except requests.exceptions.ConnectionError as e:
        error_response = HTTPResponse(
            success=False,
            status_code=503,  # Service Unavailable
            raw_response=None,
            error=f"Connection error: {str(e)}",
            base_url=url
        )
        return error_response.model_dump()
    except requests.exceptions.RequestException as e:
    # Общий обработчик для всех остальных сетевых ошибок
        error_response = HTTPResponse(
            success=False,
            status_code=500,  # Internal Server Error
            raw_response=None,
            error=f"Request failed: {str(e)}",
            base_url=url
        )
        return error_response.model_dump()

def _delete_request(endpoint: str) -> Dict[str, Any]:
    """
    Make a DELETE request with Pydantic validation while maintaining backward compatibility.
    Returns a dict for backward compatibility with existing code.
    """
    cfg = _load_server_config()
    url = f"{cfg.base_url}/{endpoint}"
    headers = _basic_headers()

    try:
        response = requests.delete(
            url,
            headers=headers,
            timeout=cfg.timeout
        )
        response.raise_for_status()

        # Create Pydantic model for validation
        http_response = HTTPResponse(
            success=response.status_code == 200,
            status_code=response.status_code,
            raw_response=None,  # DELETE requests typically don't have response body
            error=None,
            base_url=url
        )

        # Return dict for backward compatibility
        return http_response.model_dump()

    except requests.exceptions.Timeout as e:
        error_response = HTTPResponse(
            success=False,
            status_code=408,  # Request Timeout
            raw_response=None,
            error=f"Request timeout: {str(e)}",
            base_url=url
        )
        return error_response.model_dump()
    except requests.exceptions.ConnectionError as e:
        error_response = HTTPResponse(
            success=False,
            status_code=503,  # Service Unavailable
            raw_response=None,
            error=f"Connection error: {str(e)}",
            base_url=url
        )
        return error_response.model_dump()
    except requests.exceptions.RequestException as e:
    # Общий обработчик для всех остальных сетевых ошибок
        error_response = HTTPResponse(
            success=False,
            status_code=500,  # Internal Server Error
            raw_response=None,
            error=f"Request failed: {str(e)}",
            base_url=url
        )
        return error_response.model_dump()