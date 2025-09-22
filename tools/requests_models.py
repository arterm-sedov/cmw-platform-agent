"""
Pydantic Models for HTTP Requests and Responses
==============================================

This module provides Pydantic models for type safety and validation in the requests module.
These models ensure consistent data structure, prevent runtime errors, and provide
comprehensive validation for HTTP operations and API responses.

Key Models:
- HTTPResponse: Validates HTTP response structure and prevents NoneType errors
- APIResponse: Validates Comindware Platform API response format
- RequestConfig: Validates server configuration with proper error handling

Usage:
    from tools.requests_models import HTTPResponse, APIResponse, RequestConfig
    
    # Validate HTTP response
    response_data = {"success": True, "status_code": 200, ...}
    http_response = HTTPResponse(**response_data)
    
    # Validate server configuration
    config = RequestConfig(base_url="https://api.example.com", ...)

Author: AI Assistant
Date: January 18, 2025
Version: 1.0
"""

from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, field_validator


class HTTPResponse(BaseModel):
    """
    Pydantic model for HTTP responses from the requests module.
    
    This model provides type safety and validation for all HTTP responses,
    preventing common runtime errors like NoneType concatenation. It ensures
    consistent structure across all HTTP operations and provides comprehensive
    validation for response data.
    
    Attributes:
        success (bool): Whether the HTTP request was successful (status 200)
        status_code (int): HTTP status code (validated to be 100-599)
        raw_response (Optional[Union[dict, str]]): Raw response data, either
            parsed JSON dict or text string. Automatically handles None values
            to prevent concatenation errors.
        error (Optional[str]): Error message if request failed, None if successful
        base_url (str): The URL that was requested for debugging purposes
    
    Validation:
        - status_code: Must be a valid HTTP status code (100-599)
        - raw_response: Automatically converts non-dict/str types to string
        - All fields are validated on instantiation
    
    Example:
        >>> response_data = {
        ...     "success": True,
        ...     "status_code": 200,
        ...     "raw_response": {"data": "example"},
        ...     "error": None,
        ...     "base_url": "https://api.example.com/endpoint"
        ... }
        >>> http_response = HTTPResponse(**response_data)
        >>> print(http_response.success)  # True
    """
    success: bool = Field(
        description="Whether the HTTP request was successful (status 200)",
        examples=[True, False]
    )
    status_code: int = Field(
        description="HTTP status code (100-599)",
        examples=[200, 404, 500]
    )
    raw_response: Optional[Union[dict, str]] = Field(
        default=None, 
        description="Raw response data (JSON dict or text). Prevents NoneType errors.",
        examples=[{"data": "example"}, "text response", None]
    )
    error: Optional[str] = Field(
        default=None, 
        description="Error message if request failed, None if successful",
        examples=[None, "Request failed: Connection timeout"]
    )
    base_url: str = Field(
        description="The URL that was requested for debugging and logging",
        examples=["https://api.example.com/endpoint"]
    )
    
    @field_validator("status_code")
    @classmethod
    def validate_status_code(cls, v: int) -> int:
        """
        Validate that status code is a valid HTTP status code.
        
        Ensures the status code is within the valid HTTP range (100-599).
        Automatically converts invalid status codes to 500 (Internal Server Error).
        
        Args:
            v (int): The status code to validate
            
        Returns:
            int: The validated status code (converted to 500 if invalid)
        """
        if not isinstance(v, int) or v < 100 or v > 599:
            # Convert invalid status codes to 500 (Internal Server Error)
            return 500
        return v
    
    @field_validator("raw_response", mode="before")
    @classmethod
    def validate_raw_response(cls, v: Any) -> Optional[Union[dict, str]]:
        """
        Ensure raw_response is either a dict, string, or None.
        
        This validator prevents NoneType concatenation errors by ensuring
        raw_response is always a safe type. Converts unexpected types to
        strings for safety.
        
        Args:
            v (Any): The raw response value to validate
            
        Returns:
            Optional[Union[dict, str]]: Validated response data
        """
        if v is None:
            return None
        if isinstance(v, (dict, str)):
            return v
        # Convert other types to string for safety to prevent NoneType errors
        return str(v)


class APIResponse(BaseModel):
    """
    Pydantic model for API responses from Comindware Platform.
    
    This model represents the structure of responses from the Comindware Platform API.
    It provides validation for API-level success indicators and response data,
    ensuring consistent handling of Platform-specific response formats.
    
    Attributes:
        response (Optional[Any]): The actual response data from the API.
            Can be any type depending on the API endpoint.
        success (Optional[bool]): API-level success indicator. May be None
            if not provided by the API.
        error (Optional[str]): API-level error message. None if successful.
    
    Note:
        This model is designed to be flexible to handle various Platform API
        response formats while providing basic validation and type safety.
    
    Example:
        >>> api_data = {
        ...     "response": {"applications": [{"name": "TestApp"}]},
        ...     "success": True,
        ...     "error": None
        ... }
        >>> api_response = APIResponse(**api_data)
        >>> print(api_response.success)  # True
    """
    response: Optional[Any] = Field(
        default=None,
        description="The actual response data from the API (any type)",
        examples=[{"data": "example"}, [1, 2, 3], "string", None]
    )
    success: Optional[bool] = Field(
        default=None,
        description="API-level success indicator (may be None if not provided)",
        examples=[True, False, None]
    )
    error: Optional[str] = Field(
        default=None,
        description="API-level error message, None if successful",
        examples=[None, "API error: Invalid request"]
    )
    
    @field_validator("response", mode="before")
    @classmethod
    def validate_response(cls, v: Any) -> Any:
        """
        Ensure response data is properly handled.
        
        This validator allows any type of response data while ensuring
        it's properly processed. No validation is applied to maintain
        flexibility for different API response formats.
        
        Args:
            v (Any): The response data to validate
            
        Returns:
            Any: The response data as-is (no validation applied)
        """
        return v


class RequestConfig(BaseModel):
    """
    Pydantic model for request configuration.
    
    This model ensures consistent and validated configuration handling across
    all HTTP requests. It validates server configuration parameters and provides
    proper error messages for invalid configurations.
    
    Attributes:
        base_url (str): Base URL for the API. Automatically strips trailing slashes.
        login (str): Login credentials for authentication. Must not be empty.
        password (str): Password credentials for authentication. Must not be empty.
        timeout (int): Request timeout in seconds. Defaults to 30, must be positive.
    
    Validation:
        - base_url: Must not be empty, automatically strips trailing slashes
        - login/password: Must not be empty after stripping whitespace
        - timeout: Must be a positive integer
    
    Example:
        >>> config = RequestConfig(
        ...     base_url="https://api.example.com/",
        ...     login="user@example.com",
        ...     password="secret123",
        ...     timeout=60
        ... )
        >>> print(config.base_url)  # "https://api.example.com"
    """
    base_url: str = Field(
        description="Base URL for the API (trailing slashes automatically removed)",
        examples=["https://api.example.com", "https://platform.comindware.com"]
    )
    login: str = Field(
        description="Login credentials for authentication (must not be empty)",
        examples=["user@example.com", "admin"]
    )
    password: str = Field(
        description="Password credentials for authentication (must not be empty)",
        examples=["secret123", "password"]
    )
    timeout: int = Field(
        default=30,
        description="Request timeout in seconds (must be positive)",
        examples=[30, 60, 120]
    )
    
    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """
        Ensure base_url is properly formatted.
        
        Validates that the base URL is not empty and automatically removes
        trailing slashes to ensure consistent URL formatting.
        
        Args:
            v (str): The base URL to validate
            
        Returns:
            str: The validated and formatted base URL
            
        Raises:
            ValueError: If base URL is empty or None
        """
        if not v or not v.strip():
            raise ValueError("Base URL cannot be empty")
        return v.rstrip("/")
    
    @field_validator("login", "password")
    @classmethod
    def validate_credentials(cls, v: str) -> str:
        """
        Ensure credentials are not empty.
        
        Validates that login and password credentials are not empty after
        stripping whitespace. This prevents authentication failures due to
        empty credentials.
        
        Args:
            v (str): The credential value to validate
            
        Returns:
            str: The validated and stripped credential
            
        Raises:
            ValueError: If credential is empty or None after stripping
        """
        if not v or not v.strip():
            raise ValueError("Credentials cannot be empty")
        return v.strip()
    
    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """
        Ensure timeout is a positive integer.
        
        Validates that the timeout value is a positive integer to prevent
        invalid timeout configurations that could cause request issues.
        
        Args:
            v (int): The timeout value to validate
            
        Returns:
            int: The validated timeout value
            
        Raises:
            ValueError: If timeout is not a positive integer
        """
        if not isinstance(v, int) or v <= 0:
            raise ValueError("Timeout must be a positive integer")
        return v
