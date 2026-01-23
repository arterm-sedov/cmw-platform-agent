"""
Error Handler Module
===================

This module provides comprehensive error handling for LLM API calls across
different providers. It extracts and classifies errors, provides recovery
suggestions, and handles retry logic.

Key Features:
- Provider-specific error classification
- HTTP status code extraction
- Retry timing extraction
- Error recovery suggestions
- Centralized error handling logic

Usage:
    error_handler = ErrorHandler()
    error_info = error_handler.classify_error(error, "gemini")
    if error_info['is_temporary']:
        # Handle retry logic
"""

import re
import json
import time
import numpy as np
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ErrorType(Enum):
    """Enumeration of error types"""
    INVALID_ARGUMENT = "invalid_argument"
    FAILED_PRECONDITION = "failed_precondition"
    PERMISSION_DENIED = "permission_denied"
    NOT_FOUND = "not_found"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    BAD_REQUEST = "bad_request"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    REQUEST_TOO_LARGE = "request_too_large"
    UNPROCESSABLE_ENTITY = "unprocessable_entity"
    FLEX_TIER_CAPACITY_EXCEEDED = "flex_tier_capacity_exceeded"
    REQUEST_CANCELLED = "request_cancelled"
    BAD_GATEWAY = "bad_gateway"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    INVALID_MESSAGE_ORDER = "invalid_message_order"
    TOKEN_LIMIT_EXCEEDED = "token_limit_exceeded"
    CONTEXT_TOO_LONG = "context_too_long"
    NETWORK_ERROR = "network_error"
    ROUTER_ERROR = "router_error"
    PAYMENT_REQUIRED = "payment_required"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """Structured error information"""
    error_type: ErrorType
    description: str
    suggested_action: str
    is_temporary: bool
    requires_config_change: bool
    retry_after: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    status_code: Optional[int] = None
    provider: Optional[str] = None


class ErrorHandler:
    """
    Comprehensive error handler for LLM API calls.

    This class provides centralized error handling across different LLM providers,
    with specific classification and recovery suggestions for each provider.
    """

    def __init__(self):
        """Initialize the error handler"""
        self.provider_failure_counts = {}  # Will be session-specific: {session_id: {provider: count}}
        self.max_failures_per_provider = 3
        self.failure_reset_time = 3600  # 1 hour

        # Initialize vector similarity components
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.error_patterns = self._initialize_error_patterns()
        self.similarity_threshold = 0.7

    def extract_http_status_code(self, error: Exception) -> Optional[int]:
        """Extract HTTP status code from various error types with enhanced patterns."""
        error_str = str(error)

        # Pattern 1: "HTTP 429" or "429 error" or "Error code: 429"
        status_match = re.search(r'(?:HTTP\s+(\d{3})|(\d{3})\s+error|Error\s+code:\s*(\d{3}))', error_str, re.IGNORECASE)
        if status_match:
            return int(status_match.group(1) or status_match.group(2) or status_match.group(3))

        # Pattern 2: "status: 429" or "code: 429" or "status_code: 429"
        status_match = re.search(r'(?:status|code|status_code)[:\s]*(\d{3})', error_str, re.IGNORECASE)
        if status_match:
            return int(status_match.group(1))

        # Pattern 3: Try to parse JSON error responses with more flexible patterns
        try:
            # Look for JSON structures with status/code fields
            json_patterns = [
                r'\{[^}]*"status"[^}]*(\d{3})[^}]*\}',
                r'\{[^}]*"code"[^}]*(\d{3})[^}]*\}',
                r'\{[^}]*"error"[^}]*"code"[^}]*(\d{3})[^}]*\}',
                r'\{[^}]*"error"[^}]*"status"[^}]*(\d{3})[^}]*\}'
            ]

            for pattern in json_patterns:
                json_match = re.search(pattern, error_str)
                if json_match:
                    return int(json_match.group(1))

            # Try to parse complete JSON error structures
            json_match = re.search(r'\{[^}]*"status"[^}]*\d{3}[^}]*\}|\{[^}]*"code"[^}]*\d{3}[^}]*\}', error_str)
            if json_match:
                error_data = json.loads(json_match.group(0))
                if 'status' in error_data:
                    return int(error_data['status'])
                elif 'code' in error_data:
                    return int(error_data['code'])
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Pattern 4: Look for HTTP status codes in URLs and error messages
        url_status_match = re.search(r'/(\d{3})/', error_str)
        if url_status_match:
            status = int(url_status_match.group(1))
            if 400 <= status <= 599:  # Valid HTTP error range
                return status

        # Pattern 5: Look for common HTTP status codes in the message (more comprehensive)
        for status in [400, 401, 402, 403, 404, 408, 413, 422, 429, 498, 499, 500, 502, 503, 504]:
            if str(status) in error_str:
                return status

        return None

    def extract_retry_after_timing(self, error_str: str) -> Optional[int]:
        """Extract retry-after timing from error messages."""
        patterns = [
            r'retry-after[:\s]*(\d+)',
            r'retry[:\s]*after[:\s]*(\d+)',
            r'wait[:\s]*(\d+)[:\s]*seconds?',
            r'(\d+)[:\s]*seconds?[:\s]*before[:\s]*retry'
        ]

        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def _initialize_error_patterns(self) -> Dict[str, List[str]]:
        """Initialize error patterns for vector similarity matching."""
        return {
            "token_limit": [
                "Error code: 413 - Request too large for model",
                "tokens per minute (TPM): Limit",
                "Requested",
                "please reduce your message size and try again",
                "token limit exceeded",
                "context length exceeded",
                "input too long"
            ],
            "rate_limit": [
                "Rate limit exceeded",
                "Too Many Requests",
                "rate_limit_exceeded",
                "free-models-per-day",
                "quota exceeded"
            ],
            "router_error": [
                "500 Server Error: Internal Server Error for url: https://router.huggingface.co",
                "router.huggingface.co",
                "Request ID: Root="
            ],
            "network_error": [
                "no healthy upstream",
                "network",
                "connection",
                "timeout",
                "connection refused"
            ],
            "auth_error": [
                "Unauthorized",
                "Invalid API key",
                "authentication failed",
                "permission denied",
                "access denied"
            ]
        }

    def _calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        try:
            # Fit and transform both texts
            texts = [text1, text2]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception:
            return 0.0

    def _is_token_limit_error(self, error: Exception, llm_type: str = "unknown") -> bool:
        """Check if the error is a token limit error using multiple detection methods."""
        error_str = str(error).lower()

        # Direct substring checks for efficiency
        token_indicators = [
            "413", "429", "token", "limit", "tokens per minute", 
            "truncated", "tpm", "router.huggingface.co", "402", 
            "payment required", "rate limit", "rate_limit", "context too long",
            "input too long", "message too large"
        ]

        if any(term in error_str for term in token_indicators):
            return True

        # Vector similarity check for complex patterns
        for pattern in self.error_patterns["token_limit"]:
            if self._calculate_cosine_similarity(error_str, pattern) > self.similarity_threshold:
                return True

        return False

    def _is_network_error(self, error: Exception) -> bool:
        """Check if the error is a network connectivity error."""
        error_str = str(error).lower()

        # Direct checks
        network_indicators = [
            "no healthy upstream", "network", "connection", "timeout",
            "connection refused", "dns", "resolve", "unreachable"
        ]

        if any(term in error_str for term in network_indicators):
            return True

        # Vector similarity check
        for pattern in self.error_patterns["network_error"]:
            if self._calculate_cosine_similarity(error_str, pattern) > self.similarity_threshold:
                return True

        return False

    def _is_router_error(self, error: Exception) -> bool:
        """Check if the error is a HuggingFace router error."""
        error_str = str(error).lower()

        # Direct checks
        if "router.huggingface.co" in error_str or "500 server error" in error_str:
            return True

        # Vector similarity check
        for pattern in self.error_patterns["router_error"]:
            if self._calculate_cosine_similarity(error_str, pattern) > self.similarity_threshold:
                return True

        return False

    def classify_gemini_error(self, status_code: int, error_str: str) -> Optional[ErrorInfo]:
        """Classify Gemini-specific error codes based on official documentation."""
        error_str_lower = error_str.lower()

        # 400 Bad Request - Focus on error codes first
        if status_code == 400:
            if 'invalid_argument' in error_str_lower:
                return ErrorInfo(
                    error_type=ErrorType.INVALID_ARGUMENT,
                    description='INVALID_ARGUMENT - Request body is malformed',
                    suggested_action='Check API reference for request format and supported versions',
                    is_temporary=False,
                    requires_config_change=True,
                    status_code=status_code,
                    provider='gemini'
                )
            elif 'failed_precondition' in error_str_lower or 'free tier' in error_str_lower:
                return ErrorInfo(
                    error_type=ErrorType.FAILED_PRECONDITION,
                    description='FAILED_PRECONDITION - Free tier not available in your country',
                    suggested_action='Enable billing on your project in Google AI Studio',
                    is_temporary=False,
                    requires_config_change=True,
                    status_code=status_code,
                    provider='gemini'
                )
            else:
                # Generic 400 error
                return ErrorInfo(
                    error_type=ErrorType.BAD_REQUEST,
                    description='Bad Request - Invalid request parameters',
                    suggested_action='Check request format and parameters',
                    is_temporary=False,
                    requires_config_change=True,
                    status_code=status_code,
                    provider='gemini'
                )

        # 401 Unauthorized
        elif status_code == 401:
            return ErrorInfo(
                error_type=ErrorType.UNAUTHORIZED,
                description='Unauthorized - Invalid API key or authentication',
                suggested_action='Check API key configuration and permissions',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='gemini'
            )

        # 402 Payment Required
        elif status_code == 402:
            return ErrorInfo(
                error_type=ErrorType.PAYMENT_REQUIRED,
                description='Payment Required - Billing account required',
                suggested_action='Enable billing on your Google Cloud project',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='gemini'
            )

        # 403 Forbidden
        elif status_code == 403:
            if 'permission_denied' in error_str_lower:
                return ErrorInfo(
                    error_type=ErrorType.PERMISSION_DENIED,
                    description='PERMISSION_DENIED - API key lacks required permissions',
                    suggested_action='Check API key access and authentication for tuned models',
                    is_temporary=False,
                    requires_config_change=True,
                    status_code=status_code,
                    provider='gemini'
                )
            else:
                return ErrorInfo(
                    error_type=ErrorType.FORBIDDEN,
                    description='Forbidden - Access denied',
                    suggested_action='Check API key permissions and access rights',
                    is_temporary=False,
                    requires_config_change=True,
                    status_code=status_code,
                    provider='gemini'
                )

        # 404 Not Found
        elif status_code == 404:
            return ErrorInfo(
                error_type=ErrorType.NOT_FOUND,
                description='NOT_FOUND - Requested resource was not found',
                suggested_action='Check if all parameters in your request are valid for your API version',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='gemini'
            )

        # 413 Request Entity Too Large
        elif status_code == 413:
            return ErrorInfo(
                error_type=ErrorType.REQUEST_TOO_LARGE,
                description='Request Entity Too Large - Request body exceeds size limits',
                suggested_action='Reduce request size or enable chunking',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='gemini'
            )

        # 422 Unprocessable Entity
        elif status_code == 422:
            return ErrorInfo(
                error_type=ErrorType.UNPROCESSABLE_ENTITY,
                description='Unprocessable Entity - Semantic errors in request',
                suggested_action='Check request parameters and model compatibility',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='gemini'
            )

        # 429 Too Many Requests
        elif status_code == 429:
            retry_after = self.extract_retry_after_timing(error_str)
            if 'resource_exhausted' in error_str_lower:
                return ErrorInfo(
                    error_type=ErrorType.RESOURCE_EXHAUSTED,
                    description='RESOURCE_EXHAUSTED - Rate limit exceeded',
                    suggested_action='Verify rate limits and request quota increase if needed',
                    is_temporary=True,
                    requires_config_change=False,
                    retry_after=retry_after,
                    status_code=status_code,
                    provider='gemini'
                )
            else:
                return ErrorInfo(
                    error_type=ErrorType.RATE_LIMIT_EXCEEDED,
                    description='Too Many Requests - Rate limit exceeded',
                    suggested_action='Wait before retrying or reduce request frequency',
                    is_temporary=True,
                    requires_config_change=False,
                    retry_after=retry_after,
                    status_code=status_code,
                    provider='gemini'
                )

        # 500 Internal Server Error
        elif status_code == 500:
            if 'internal' in error_str_lower or 'context' in error_str_lower:
                return ErrorInfo(
                    error_type=ErrorType.CONTEXT_TOO_LONG,
                    description='INTERNAL - Input context too long',
                    suggested_action='Reduce input context or switch to Gemini 1.5 Flash',
                    is_temporary=False,
                    requires_config_change=True,
                    status_code=status_code,
                    provider='gemini'
                )
            else:
                return ErrorInfo(
                    error_type=ErrorType.INTERNAL_ERROR,
                    description='Internal Server Error - Generic server error',
                    suggested_action='Try again later or contact support',
                    is_temporary=True,
                    requires_config_change=False,
                    status_code=status_code,
                    provider='gemini'
                )

        # 502 Bad Gateway
        elif status_code == 502:
            return ErrorInfo(
                error_type=ErrorType.BAD_GATEWAY,
                description='Bad Gateway - Invalid response from upstream server',
                suggested_action='Try again later or use different model',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider='gemini'
            )

        # 503 Service Unavailable
        elif status_code == 503:
            return ErrorInfo(
                error_type=ErrorType.SERVICE_UNAVAILABLE,
                description='UNAVAILABLE - Service temporarily overloaded',
                suggested_action='Switch to Gemini 1.5 Flash or wait and retry',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider='gemini'
            )

        # 504 Gateway Timeout
        elif status_code == 504:
            return ErrorInfo(
                error_type=ErrorType.DEADLINE_EXCEEDED,
                description='DEADLINE_EXCEEDED - Service unable to finish processing within deadline',
                suggested_action='Set larger timeout or reduce prompt size',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='gemini'
            )

        return None

    def classify_groq_error(self, status_code: int, error_str: str) -> Optional[ErrorInfo]:
        """Classify Groq-specific error codes based on official documentation."""
        error_str_lower = error_str.lower()

        if status_code == 400:
            return ErrorInfo(
                error_type=ErrorType.BAD_REQUEST,
                description='Bad Request - Invalid syntax in request',
                suggested_action='Check request format and parameters',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='groq'
            )

        elif status_code == 401:
            return ErrorInfo(
                error_type=ErrorType.UNAUTHORIZED,
                description='Unauthorized - Invalid authentication credentials',
                suggested_action='Check API key configuration',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='groq'
            )

        elif status_code == 404:
            return ErrorInfo(
                error_type=ErrorType.NOT_FOUND,
                description='Not Found - Requested resource not found',
                suggested_action='Check model name and endpoint configuration',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='groq'
            )

        elif status_code == 413:
            return ErrorInfo(
                error_type=ErrorType.REQUEST_TOO_LARGE,
                description='Request Entity Too Large - Request body exceeds size limits',
                suggested_action='Reduce request size or enable chunking',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='groq'
            )

        elif status_code == 422:
            return ErrorInfo(
                error_type=ErrorType.UNPROCESSABLE_ENTITY,
                description='Unprocessable Entity - Semantic errors in request',
                suggested_action='Check request parameters and model compatibility',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='groq'
            )

        elif status_code == 429:
            retry_after = self.extract_retry_after_timing(error_str)
            return ErrorInfo(
                error_type=ErrorType.RATE_LIMIT_EXCEEDED,
                description='Too Many Requests - Rate limit exceeded',
                suggested_action='Wait before retrying or reduce request frequency',
                is_temporary=True,
                requires_config_change=False,
                retry_after=retry_after,
                status_code=status_code,
                provider='groq'
            )

        elif status_code == 498:
            return ErrorInfo(
                error_type=ErrorType.FLEX_TIER_CAPACITY_EXCEEDED,
                description='Flex Tier Capacity Exceeded - Service at capacity',
                suggested_action='Wait for capacity to free up or upgrade to higher tier',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider='groq'
            )

        elif status_code == 499:
            return ErrorInfo(
                error_type=ErrorType.REQUEST_CANCELLED,
                description='Request Cancelled - Request was cancelled by caller',
                suggested_action='Retry the request if needed',
                is_temporary=False,
                requires_config_change=False,
                status_code=status_code,
                provider='groq'
            )

        elif status_code == 500:
            return ErrorInfo(
                error_type=ErrorType.INTERNAL_ERROR,
                description='Internal Server Error - Generic server error',
                suggested_action='Try again later or contact support',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider='groq'
            )

        elif status_code == 502:
            return ErrorInfo(
                error_type=ErrorType.BAD_GATEWAY,
                description='Bad Gateway - Invalid response from upstream server',
                suggested_action='Try again later or use different model',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider='groq'
            )

        elif status_code == 503:
            return ErrorInfo(
                error_type=ErrorType.SERVICE_UNAVAILABLE,
                description='Service Unavailable - Server not ready to handle request',
                suggested_action='Wait for maintenance to complete or try different model',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider='groq'
            )

        return None

    def classify_mistral_error(self, status_code: int, error_str: str) -> Optional[ErrorInfo]:
        """Classify Mistral-specific error codes based on official documentation."""
        error_str_lower = error_str.lower()

        # Check for Mistral-specific error codes first
        if 'invalid_request_message_order' in error_str_lower or '3230' in error_str:
            return ErrorInfo(
                error_type=ErrorType.INVALID_MESSAGE_ORDER,
                description='Invalid Request Message Order - Message sequence is incorrect',
                suggested_action='Reconstruct conversation with proper message ordering',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='mistral'
            )

        # Standard HTTP status codes for Mistral
        if status_code == 400:
            return ErrorInfo(
                error_type=ErrorType.BAD_REQUEST,
                description='Bad Request - Invalid or missing parameters',
                suggested_action='Check request parameters and format',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='mistral'
            )

        elif status_code == 401:
            return ErrorInfo(
                error_type=ErrorType.UNAUTHORIZED,
                description='Unauthorized - Invalid API key',
                suggested_action='Check API key configuration',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='mistral'
            )

        elif status_code == 403:
            return ErrorInfo(
                error_type=ErrorType.FORBIDDEN,
                description='Forbidden - Server refuses to authorize the request',
                suggested_action='Check API key permissions and access rights',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='mistral'
            )

        elif status_code == 404:
            return ErrorInfo(
                error_type=ErrorType.NOT_FOUND,
                description='Not Found - Requested resource could not be found',
                suggested_action='Check if the requested resource exists and is accessible',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='mistral'
            )

        elif status_code == 413:
            return ErrorInfo(
                error_type=ErrorType.REQUEST_TOO_LARGE,
                description='Request Entity Too Large - Request body exceeds size limits',
                suggested_action='Reduce request size or enable chunking',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='mistral'
            )

        elif status_code == 422:
            return ErrorInfo(
                error_type=ErrorType.UNPROCESSABLE_ENTITY,
                description='Unprocessable Entity - Semantic errors in request',
                suggested_action='Check request parameters and model compatibility',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='mistral'
            )

        elif status_code == 429:
            retry_after = self.extract_retry_after_timing(error_str)
            return ErrorInfo(
                error_type=ErrorType.RATE_LIMIT_EXCEEDED,
                description='Too Many Requests - Rate limit exceeded',
                suggested_action='Wait before retrying or reduce request frequency',
                is_temporary=True,
                requires_config_change=False,
                retry_after=retry_after,
                status_code=status_code,
                provider='mistral'
            )

        elif status_code == 500:
            return ErrorInfo(
                error_type=ErrorType.INTERNAL_ERROR,
                description='Internal Server Error - Generic server error',
                suggested_action='Try again later or contact support',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider='mistral'
            )

        elif status_code == 502:
            return ErrorInfo(
                error_type=ErrorType.BAD_GATEWAY,
                description='Bad Gateway - Invalid response from upstream server',
                suggested_action='Try again later or use different model',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider='mistral'
            )

        elif status_code == 503:
            return ErrorInfo(
                error_type=ErrorType.SERVICE_UNAVAILABLE,
                description='Service Unavailable - Server not ready to handle request',
                suggested_action='Wait for maintenance to complete or try different model',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider='mistral'
            )

        return None

    def classify_openrouter_error(self, status_code: int, error_str: str) -> Optional[ErrorInfo]:
        """Classify OpenRouter-specific error codes based on official documentation."""
        error_str_lower = error_str.lower()

        if status_code == 400:
            return ErrorInfo(
                error_type=ErrorType.BAD_REQUEST,
                description='Bad Request - Invalid or missing parameters',
                suggested_action='Check request parameters and format',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='openrouter'
            )

        elif status_code == 401:
            return ErrorInfo(
                error_type=ErrorType.UNAUTHORIZED,
                description='Unauthorized - Invalid API key',
                suggested_action='Check API key configuration',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='openrouter'
            )

        elif status_code == 403:
            return ErrorInfo(
                error_type=ErrorType.FORBIDDEN,
                description='Forbidden - Server refuses to authorize the request',
                suggested_action='Check API key permissions and access rights',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='openrouter'
            )

        elif status_code == 404:
            return ErrorInfo(
                error_type=ErrorType.NOT_FOUND,
                description='Not Found - Requested resource could not be found',
                suggested_action='Check if the requested resource exists and is accessible',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='openrouter'
            )

        elif status_code == 413:
            return ErrorInfo(
                error_type=ErrorType.REQUEST_TOO_LARGE,
                description='Request Entity Too Large - Request body exceeds size limits',
                suggested_action='Reduce request size or enable chunking',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='openrouter'
            )

        elif status_code == 422:
            return ErrorInfo(
                error_type=ErrorType.UNPROCESSABLE_ENTITY,
                description='Unprocessable Entity - Semantic errors in request',
                suggested_action='Check request parameters and model compatibility',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='openrouter'
            )

        elif status_code == 429:
            retry_after = self.extract_retry_after_timing(error_str)
            return ErrorInfo(
                error_type=ErrorType.RATE_LIMIT_EXCEEDED,
                description='Too Many Requests - Rate limit exceeded',
                suggested_action='Wait before retrying or reduce request frequency',
                is_temporary=True,
                requires_config_change=False,
                retry_after=retry_after,
                status_code=status_code,
                provider='openrouter'
            )

        elif status_code == 500:
            return ErrorInfo(
                error_type=ErrorType.INTERNAL_ERROR,
                description='Internal Server Error - Generic server error',
                suggested_action='Try again later or contact support',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider='openrouter'
            )

        elif status_code == 502:
            return ErrorInfo(
                error_type=ErrorType.BAD_GATEWAY,
                description='Bad Gateway - Invalid response from upstream server',
                suggested_action='Try again later or use different model',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider='openrouter'
            )

        elif status_code == 503:
            return ErrorInfo(
                error_type=ErrorType.SERVICE_UNAVAILABLE,
                description='Service Unavailable - Server not ready to handle request',
                suggested_action='Wait for maintenance to complete or try different model',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider='openrouter'
            )

        return None

    def classify_huggingface_error(self, status_code: int, error_str: str) -> Optional[ErrorInfo]:
        """Classify HuggingFace-specific error codes based on official documentation."""
        error_str_lower = error_str.lower()

        if status_code == 400:
            return ErrorInfo(
                error_type=ErrorType.BAD_REQUEST,
                description='Bad Request - Invalid or missing parameters',
                suggested_action='Check request parameters and format',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='huggingface'
            )

        elif status_code == 401:
            return ErrorInfo(
                error_type=ErrorType.UNAUTHORIZED,
                description='Unauthorized - Invalid API key',
                suggested_action='Check API key configuration',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='huggingface'
            )

        elif status_code == 403:
            return ErrorInfo(
                error_type=ErrorType.FORBIDDEN,
                description='Forbidden - Server refuses to authorize the request',
                suggested_action='Check API key permissions and access rights',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='huggingface'
            )

        elif status_code == 404:
            return ErrorInfo(
                error_type=ErrorType.NOT_FOUND,
                description='Not Found - Requested resource could not be found',
                suggested_action='Check if the requested resource exists and is accessible',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='huggingface'
            )

        elif status_code == 413:
            return ErrorInfo(
                error_type=ErrorType.REQUEST_TOO_LARGE,
                description='Request Entity Too Large - Request body exceeds size limits',
                suggested_action='Reduce request size or enable chunking',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='huggingface'
            )

        elif status_code == 422:
            return ErrorInfo(
                error_type=ErrorType.UNPROCESSABLE_ENTITY,
                description='Unprocessable Entity - Semantic errors in request',
                suggested_action='Check request parameters and model compatibility',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='huggingface'
            )

        elif status_code == 429:
            retry_after = self.extract_retry_after_timing(error_str)
            return ErrorInfo(
                error_type=ErrorType.RATE_LIMIT_EXCEEDED,
                description='Too Many Requests - Rate limit exceeded',
                suggested_action='Wait before retrying or reduce request frequency',
                is_temporary=True,
                requires_config_change=False,
                retry_after=retry_after,
                status_code=status_code,
                provider='huggingface'
            )

        elif status_code == 500:
            return ErrorInfo(
                error_type=ErrorType.INTERNAL_ERROR,
                description='Internal Server Error - Generic server error',
                suggested_action='Try again later or contact support',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider='huggingface'
            )

        elif status_code == 502:
            return ErrorInfo(
                error_type=ErrorType.BAD_GATEWAY,
                description='Bad Gateway - Invalid response from upstream server',
                suggested_action='Try again later or use different model',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider='huggingface'
            )

        elif status_code == 503:
            return ErrorInfo(
                error_type=ErrorType.SERVICE_UNAVAILABLE,
                description='Service Unavailable - Server not ready to handle request',
                suggested_action='Wait for maintenance to complete or try different model',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider='huggingface'
            )

        return None

    def classify_gigachat_error(self, status_code: int, error_str: str) -> Optional[ErrorInfo]:
        """Classify GigaChat-specific error codes based on official documentation."""
        error_str_lower = error_str.lower()

        if status_code == 400:
            return ErrorInfo(
                error_type=ErrorType.BAD_REQUEST,
                description='Bad Request - Invalid or missing parameters',
                suggested_action='Check request parameters and format',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='gigachat'
            )

        elif status_code == 401:
            return ErrorInfo(
                error_type=ErrorType.UNAUTHORIZED,
                description='Unauthorized - Invalid API key',
                suggested_action='Check API key configuration',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='gigachat'
            )

        elif status_code == 403:
            return ErrorInfo(
                error_type=ErrorType.FORBIDDEN,
                description='Forbidden - Server refuses to authorize the request',
                suggested_action='Check API key permissions and access rights',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='gigachat'
            )

        elif status_code == 404:
            return ErrorInfo(
                error_type=ErrorType.NOT_FOUND,
                description='Not Found - Requested resource could not be found',
                suggested_action='Check if the requested resource exists and is accessible',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='gigachat'
            )

        elif status_code == 413:
            return ErrorInfo(
                error_type=ErrorType.REQUEST_TOO_LARGE,
                description='Request Entity Too Large - Request body exceeds size limits',
                suggested_action='Reduce request size or enable chunking',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='gigachat'
            )

        elif status_code == 422:
            return ErrorInfo(
                error_type=ErrorType.UNPROCESSABLE_ENTITY,
                description='Unprocessable Entity - Semantic errors in request',
                suggested_action='Check request parameters and model compatibility',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider='gigachat'
            )

        elif status_code == 429:
            retry_after = self.extract_retry_after_timing(error_str)
            return ErrorInfo(
                error_type=ErrorType.RATE_LIMIT_EXCEEDED,
                description='Too Many Requests - Rate limit exceeded',
                suggested_action='Wait before retrying or reduce request frequency',
                is_temporary=True,
                requires_config_change=False,
                retry_after=retry_after,
                status_code=status_code,
                provider='gigachat'
            )

        elif status_code == 500:
            return ErrorInfo(
                error_type=ErrorType.INTERNAL_ERROR,
                description='Internal Server Error - Generic server error',
                suggested_action='Try again later or contact support',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider='gigachat'
            )

        elif status_code == 502:
            return ErrorInfo(
                error_type=ErrorType.BAD_GATEWAY,
                description='Bad Gateway - Invalid response from upstream server',
                suggested_action='Try again later or use different model',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider='gigachat'
            )

        elif status_code == 503:
            return ErrorInfo(
                error_type=ErrorType.SERVICE_UNAVAILABLE,
                description='Service Unavailable - Server not ready to handle request',
                suggested_action='Wait for maintenance to complete or try different model',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider='gigachat'
            )

        return None

    def classify_error_by_status_code(self, status_code: int, error: Exception, llm_name: str) -> Optional[ErrorInfo]:
        """Classify error based on official HTTP status codes."""
        error_str = str(error)

        # Check for provider-specific error status codes first
        if 'gemini' in llm_name.lower():
            return self.classify_gemini_error(status_code, error_str)
        elif 'groq' in llm_name.lower():
            return self.classify_groq_error(status_code, error_str)
        elif 'mistral' in llm_name.lower():
            return self.classify_mistral_error(status_code, error_str)
        elif 'openrouter' in llm_name.lower():
            return self.classify_openrouter_error(status_code, error_str)
        elif 'huggingface' in llm_name.lower():
            return self.classify_huggingface_error(status_code, error_str)
        elif 'gigachat' in llm_name.lower():
            return self.classify_gigachat_error(status_code, error_str)

        # Generic HTTP status code handling
        if status_code == 400:
            return ErrorInfo(
                error_type=ErrorType.BAD_REQUEST,
                description='Bad Request - Invalid or missing parameters',
                suggested_action='Check request parameters and format',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider=llm_name.lower()
            )
        elif status_code == 401:
            return ErrorInfo(
                error_type=ErrorType.UNAUTHORIZED,
                description='Unauthorized - Invalid authentication credentials',
                suggested_action='Check API key configuration',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider=llm_name.lower()
            )
        elif status_code == 403:
            return ErrorInfo(
                error_type=ErrorType.FORBIDDEN,
                description='Forbidden - Server refuses to authorize the request',
                suggested_action='Check API key permissions and access rights',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider=llm_name.lower()
            )
        elif status_code == 404:
            return ErrorInfo(
                error_type=ErrorType.NOT_FOUND,
                description='Not Found - Requested resource could not be found',
                suggested_action='Check if the requested resource exists and is accessible',
                is_temporary=False,
                requires_config_change=True,
                status_code=status_code,
                provider=llm_name.lower()
            )
        elif status_code == 429:
            retry_after = self.extract_retry_after_timing(error_str)
            return ErrorInfo(
                error_type=ErrorType.RATE_LIMIT_EXCEEDED,
                description='Too Many Requests - Rate limit exceeded',
                suggested_action='Wait before retrying or reduce request frequency',
                is_temporary=True,
                requires_config_change=False,
                retry_after=retry_after,
                status_code=status_code,
                provider=llm_name.lower()
            )
        elif status_code == 500:
            return ErrorInfo(
                error_type=ErrorType.INTERNAL_ERROR,
                description='Internal Server Error - Generic server error',
                suggested_action='Try again later or contact support',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider=llm_name.lower()
            )
        elif status_code == 502:
            return ErrorInfo(
                error_type=ErrorType.BAD_GATEWAY,
                description='Bad Gateway - Invalid response from upstream server',
                suggested_action='Try again later or use different model',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider=llm_name.lower()
            )
        elif status_code == 503:
            return ErrorInfo(
                error_type=ErrorType.SERVICE_UNAVAILABLE,
                description='Service Unavailable - Server not ready to handle request',
                suggested_action='Wait for maintenance to complete or try different model',
                is_temporary=True,
                requires_config_change=False,
                status_code=status_code,
                provider=llm_name.lower()
            )

        return None

    def classify_error_by_message(self, error_message: str, llm_name: str) -> Optional[ErrorInfo]:
        """Fallback classification based on error message content."""
        error_lower = error_message.lower()

        # Only use this for very specific, well-known error patterns
        if 'location is not supported' in error_lower:
            return ErrorInfo(
                error_type=ErrorType.FAILED_PRECONDITION,
                description='Location Not Supported - Service not available in your region',
                suggested_action='Use a different provider or VPN',
                is_temporary=False,
                requires_config_change=True,
                provider=llm_name.lower()
            )
        elif 'token limit' in error_lower or 'context length' in error_lower:
            return ErrorInfo(
                error_type=ErrorType.TOKEN_LIMIT_EXCEEDED,
                description='Token Limit Exceeded - Input exceeds maximum context length',
                suggested_action='Reduce input length or enable chunking',
                is_temporary=False,
                requires_config_change=True,
                provider=llm_name.lower()
            )
        elif 'rate limit' in error_lower or 'too many requests' in error_lower:
            return ErrorInfo(
                error_type=ErrorType.RATE_LIMIT_EXCEEDED,
                description='Rate Limit Exceeded - Too many requests',
                suggested_action='Wait before retrying or reduce request frequency',
                is_temporary=True,
                requires_config_change=False,
                provider=llm_name.lower()
            )

        return None

    def classify_error(self, error: Exception, llm_name: str) -> ErrorInfo:
        """
        Main error classification method that tries all classification strategies.
        Prioritizes error codes over keyword parsing for reliability.

        Args:
            error: The exception that occurred
            llm_name: Name of the LLM provider

        Returns:
            ErrorInfo object with classification results
        """
        error_str = str(error)
        error_lower = error_str.lower()

        # 1. PRIORITY: Try to extract HTTP status code first (most reliable)
        status_code = self.extract_http_status_code(error)

        if status_code:
            error_info = self.classify_error_by_status_code(status_code, error, llm_name)
            if error_info:
                return error_info

        # 2. Check for specific error patterns using vector similarity (from agent.py)
        if self._is_token_limit_error(error, llm_name):
            return ErrorInfo(
                error_type=ErrorType.TOKEN_LIMIT_EXCEEDED,
                description='Token Limit Exceeded - Input exceeds maximum context length',
                suggested_action='Reduce input length or enable chunking',
                is_temporary=False,
                requires_config_change=True,
                provider=llm_name.lower()
            )

        if self._is_network_error(error):
            return ErrorInfo(
                error_type=ErrorType.NETWORK_ERROR,
                description='Network Error - Connectivity issue with provider',
                suggested_action='Check network connection and try again',
                is_temporary=True,
                requires_config_change=False,
                provider=llm_name.lower()
            )

        if self._is_router_error(error):
            return ErrorInfo(
                error_type=ErrorType.ROUTER_ERROR,
                description='Router Error - HuggingFace router service error',
                suggested_action='Try a different provider or wait for service recovery',
                is_temporary=True,
                requires_config_change=False,
                provider=llm_name.lower()
            )

        # 3. Check for Mistral-specific message ordering errors
        if 'mistral' in llm_name.lower() and ('invalid_request_message_order' in error_lower or '3230' in error_str):
            return ErrorInfo(
                error_type=ErrorType.INVALID_MESSAGE_ORDER,
                description='Invalid Request Message Order - Message sequence is incorrect',
                suggested_action='Reconstruct conversation with proper message ordering',
                is_temporary=False,
                requires_config_change=True,
                provider=llm_name.lower()
            )

        # 4. Fallback to message-based classification for edge cases
        error_info = self.classify_error_by_message(error_str, llm_name)
        if error_info:
            return error_info

        # 5. Last resort: Check for specific error patterns that might indicate status codes
        if 'invalid_argument' in error_lower or 'malformed' in error_lower:
            return ErrorInfo(
                error_type=ErrorType.INVALID_ARGUMENT,
                description='Invalid Argument - Request body is malformed',
                suggested_action='Check API reference for request format and supported versions',
                is_temporary=False,
                requires_config_change=True,
                provider=llm_name.lower()
            )
        elif 'unauthorized' in error_lower:
            return ErrorInfo(
                error_type=ErrorType.UNAUTHORIZED,
                description='Unauthorized - Invalid authentication credentials',
                suggested_action='Check API key configuration',
                is_temporary=False,
                requires_config_change=True,
                provider=llm_name.lower()
            )
        elif 'not found' in error_lower:
            return ErrorInfo(
                error_type=ErrorType.NOT_FOUND,
                description='Not Found - Requested resource could not be found',
                suggested_action='Check if the requested resource exists and is accessible',
                is_temporary=False,
                requires_config_change=True,
                provider=llm_name.lower()
            )

        # 6. Default unknown error
        return ErrorInfo(
            error_type=ErrorType.UNKNOWN,
            description=f'Unknown error: {error_str[:200]}...',
            suggested_action='Check logs and contact support if issue persists',
            is_temporary=True,
            requires_config_change=False,
            provider=llm_name.lower()
        )

    def handle_provider_failure(self, provider_type: str, error_type: str = "general", session_id: str = "default") -> bool:
        """
        Track provider failures with simple retry limits.

        Args:
            provider_type: Provider name (e.g., "mistral", "gemini", "groq")
            error_type: Type of error that occurred
            session_id: Session ID for isolation

        Returns:
            True if provider should be temporarily skipped, False otherwise
        """
        current_time = time.time()
        key = f"{session_id}_{provider_type}_{error_type}"

        if key not in self.provider_failure_counts:
            self.provider_failure_counts[key] = {
                'count': 0,
                'first_failure': current_time,
                'last_failure': current_time
            }

        failure_info = self.provider_failure_counts[key]

        # Reset if enough time has passed
        if current_time - failure_info['first_failure'] > self.failure_reset_time:
            failure_info['count'] = 0
            failure_info['first_failure'] = current_time

        failure_info['count'] += 1
        failure_info['last_failure'] = current_time

        # Skip provider if too many failures
        return failure_info['count'] >= self.max_failures_per_provider

    def should_skip_provider_temporarily(self, provider_type: str, session_id: str = "default") -> bool:
        """Check if a provider should be temporarily skipped due to failures."""
        current_time = time.time()
        session_prefix = f"{session_id}_{provider_type}"

        for key, failure_info in self.provider_failure_counts.items():
            if key.startswith(session_prefix):
                # Reset if enough time has passed
                if current_time - failure_info['first_failure'] > self.failure_reset_time:
                    failure_info['count'] = 0
                    failure_info['first_failure'] = current_time

                # Skip if too many failures
                if failure_info['count'] >= self.max_failures_per_provider:
                    return True

        return False

    def reset_provider_failures(self, provider_type: str = None, session_id: str = "default"):
        """Reset failure counts for a provider or all providers."""
        if provider_type:
            session_prefix = f"{session_id}_{provider_type}"
            keys_to_remove = [k for k in self.provider_failure_counts.keys() if k.startswith(session_prefix)]
            for key in keys_to_remove:
                del self.provider_failure_counts[key]
        else:
            # Reset all failures for this session
            session_keys = [k for k in self.provider_failure_counts.keys() if k.startswith(f"{session_id}_")]
            for key in session_keys:
                del self.provider_failure_counts[key]

    def get_provider_failure_stats(self, session_id: str = "default") -> Dict[str, Any]:
        """Get statistics about provider failures for a specific session."""
        current_time = time.time()
        stats = {}
        session_prefix = f"{session_id}_"

        for key, failure_info in self.provider_failure_counts.items():
            if key.startswith(session_prefix):
                # Remove session prefix to get provider and error type
                remaining_key = key[len(session_prefix):]
                parts = remaining_key.split('_', 1)
                if len(parts) >= 2:
                    provider, error_type = parts[0], parts[1]
                else:
                    provider, error_type = remaining_key, "general"

                if provider not in stats:
                    stats[provider] = {}

                stats[provider][error_type] = {
                    'count': failure_info['count'],
                    'first_failure': failure_info['first_failure'],
                    'last_failure': failure_info['last_failure'],
                    'time_since_last': current_time - failure_info['last_failure'],
                    'should_skip': failure_info['count'] >= self.max_failures_per_provider
                }

        return stats


# Global error handler instance
_error_handler = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance (singleton pattern)"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def reset_error_handler():
    """Reset the global error handler (useful for testing)"""
    global _error_handler
    _error_handler = None
