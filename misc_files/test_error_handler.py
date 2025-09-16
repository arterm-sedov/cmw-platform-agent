"""
Test script for Error Handler Module
===================================

This script tests the Error Handler functionality to ensure it works correctly.
"""

import sys
from error_handler import get_error_handler, ErrorType, reset_error_handler


class MockError(Exception):
    """Mock error for testing"""
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def test_error_handler():
    """Test the Error Handler functionality"""
    print("Testing Error Handler...")
    
    # Reset handler for clean test
    reset_error_handler()
    handler = get_error_handler()
    
    # Test 1: HTTP status code extraction
    print("\n1. Testing HTTP status code extraction...")
    test_errors = [
        MockError("HTTP 429 Too Many Requests"),
        MockError("Rate limit exceeded (429 error)"),
        MockError('{"status": 500, "message": "Internal error"}'),
        MockError("Bad request with 400 status"),
        MockError("Some random error")
    ]
    
    for error in test_errors:
        status_code = handler.extract_http_status_code(error)
        print(f"  Error: {str(error)[:50]}... -> Status: {status_code}")
    
    # Test 2: Retry timing extraction
    print("\n2. Testing retry timing extraction...")
    test_messages = [
        "Rate limit exceeded. Retry after 60 seconds",
        "Too many requests. Wait 30 seconds before retry",
        "Service unavailable. Retry-After: 120",
        "Some error without timing info"
    ]
    
    for message in test_messages:
        retry_after = handler.extract_retry_after_timing(message)
        print(f"  Message: {message} -> Retry after: {retry_after}")
    
    # Test 3: Provider-specific error classification
    print("\n3. Testing provider-specific error classification...")
    test_cases = [
        (MockError("INVALID_ARGUMENT: Request body is malformed"), "gemini", 400),
        (MockError("Rate limit exceeded"), "groq", 429),
        (MockError("Invalid request message order"), "mistral", 400),
        (MockError("Unauthorized"), "openrouter", 401),
        (MockError("Not found"), "huggingface", 404),
    ]
    
    for error, provider, status_code in test_cases:
        # Manually set the status code for testing
        error.status_code = status_code
        error_info = handler.classify_error(error, provider)
        print(f"  Provider: {provider}, Status: {status_code}")
        print(f"    Type: {error_info.error_type.value}")
        print(f"    Description: {error_info.description}")
        print(f"    Temporary: {error_info.is_temporary}")
        print(f"    Action: {error_info.suggested_action}")
        print()
    
    # Test 4: Provider failure tracking
    print("\n4. Testing provider failure tracking...")
    test_providers = ["gemini", "groq", "mistral"]
    
    for provider in test_providers:
        for i in range(5):  # Simulate multiple failures
            should_skip = handler.handle_provider_failure(provider, "rate_limit")
            print(f"  {provider} failure {i+1}: should_skip = {should_skip}")
    
    # Test 5: Provider failure stats
    print("\n5. Testing provider failure stats...")
    stats = handler.get_provider_failure_stats()
    print(f"  Failure stats: {stats}")
    
    # Test 6: Reset functionality
    print("\n6. Testing reset functionality...")
    handler.reset_provider_failures("gemini")
    stats_after_reset = handler.get_provider_failure_stats()
    print(f"  Stats after reset: {stats_after_reset}")
    
    # Test 7: Error classification edge cases
    print("\n7. Testing error classification edge cases...")
    edge_cases = [
        (MockError("Location is not supported"), "gemini"),
        (MockError("Token limit exceeded"), "groq"),
        (MockError("Some unknown error"), "mistral"),
    ]
    
    for error, provider in edge_cases:
        error_info = handler.classify_error(error, provider)
        print(f"  Provider: {provider}")
        print(f"    Type: {error_info.error_type.value}")
        print(f"    Description: {error_info.description}")
        print()
    
    print("✓ Error Handler test completed!")


if __name__ == "__main__":
    test_error_handler()
