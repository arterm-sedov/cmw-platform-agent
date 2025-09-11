#!/usr/bin/env python3
"""
Test script for the enhanced error handler.
Tests error code extraction, vector similarity, and provider-specific error handling.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from error_handler import ErrorHandler, ErrorType
import traceback

def test_status_code_extraction():
    """Test HTTP status code extraction with various error formats."""
    print("🧪 Testing HTTP status code extraction...")
    
    handler = ErrorHandler()
    
    # Test cases for different error formats
    test_cases = [
        # HTTP status in various formats
        ("HTTP 429 error", 429),
        ("Error code: 413", 413),
        ("status: 500", 500),
        ("code: 401", 401),
        ("status_code: 404", 404),
        
        # JSON error responses
        ('{"error": {"code": 422}}', 422),
        ('{"status": 503}', 503),
        ('{"error": {"message": "Bad request", "code": 400}}', 400),
        
        # URL patterns
        ("https://api.example.com/500/", 500),
        ("Internal Server Error 502", 502),
        
        # Edge cases
        ("No status code here", None),
        ("", None),
    ]
    
    for error_msg, expected_status in test_cases:
        class MockError(Exception):
            def __str__(self):
                return error_msg
        
        try:
            result = handler.extract_http_status_code(MockError())
            if result == expected_status:
                print(f"✅ '{error_msg}' -> {result}")
            else:
                print(f"❌ '{error_msg}' -> Expected {expected_status}, got {result}")
        except Exception as e:
            print(f"❌ Error processing '{error_msg}': {e}")

def test_vector_similarity():
    """Test vector similarity-based error detection."""
    print("\n🧪 Testing vector similarity error detection...")
    
    handler = ErrorHandler()
    
    # Test token limit error detection
    token_errors = [
        "Error code: 413 - Request too large for model",
        "tokens per minute (TPM): Limit 6000, Requested 9681",
        "token limit exceeded",
        "context length exceeded",
        "input too long"
    ]
    
    for error_msg in token_errors:
        class MockError(Exception):
            def __str__(self):
                return error_msg
        
        try:
            is_token_error = handler._is_token_limit_error(MockError())
            print(f"✅ Token error '{error_msg[:50]}...' -> {is_token_error}")
        except Exception as e:
            print(f"❌ Error processing token error: {e}")
    
    # Test network error detection
    network_errors = [
        "no healthy upstream",
        "connection timeout",
        "network unreachable",
        "connection refused"
    ]
    
    for error_msg in network_errors:
        class MockError(Exception):
            def __str__(self):
                return error_msg
        
        try:
            is_network_error = handler._is_network_error(MockError())
            print(f"✅ Network error '{error_msg}' -> {is_network_error}")
        except Exception as e:
            print(f"❌ Error processing network error: {e}")

def test_provider_specific_errors():
    """Test provider-specific error classification."""
    print("\n🧪 Testing provider-specific error classification...")
    
    handler = ErrorHandler()
    
    # Test Gemini errors
    gemini_errors = [
        (400, "INVALID_ARGUMENT: Request body is malformed"),
        (401, "Unauthorized: Invalid API key"),
        (403, "PERMISSION_DENIED: API key lacks required permissions"),
        (404, "NOT_FOUND: Requested resource was not found"),
        (429, "RESOURCE_EXHAUSTED: Rate limit exceeded"),
        (500, "INTERNAL: Input context too long"),
        (503, "UNAVAILABLE: Service temporarily overloaded"),
        (504, "DEADLINE_EXCEEDED: Service unable to finish processing")
    ]
    
    for status_code, error_msg in gemini_errors:
        class MockError(Exception):
            def __str__(self):
                return error_msg
        
        try:
            error_info = handler.classify_gemini_error(status_code, error_msg)
            if error_info:
                print(f"✅ Gemini {status_code}: {error_info.error_type.value}")
            else:
                print(f"❌ Gemini {status_code}: No classification")
        except Exception as e:
            print(f"❌ Error processing Gemini {status_code}: {e}")
    
    # Test Mistral message ordering error
    mistral_error = "invalid_request_message_order: Message sequence is incorrect"
    class MockMistralError(Exception):
        def __str__(self):
            return mistral_error
    
    try:
        error_info = handler.classify_error(MockMistralError(), "mistral")
        if error_info and error_info.error_type == ErrorType.INVALID_MESSAGE_ORDER:
            print(f"✅ Mistral message ordering error detected: {error_info.error_type.value}")
        else:
            print(f"❌ Mistral message ordering error not detected")
    except Exception as e:
        print(f"❌ Error processing Mistral error: {e}")

def test_comprehensive_error_classification():
    """Test the main error classification method with various error types."""
    print("\n🧪 Testing comprehensive error classification...")
    
    handler = ErrorHandler()
    
    test_errors = [
        # Status code errors
        ("HTTP 429 error", "groq"),
        ("Error code: 413", "huggingface"),
        ("status: 500", "gemini"),
        
        # Token limit errors
        ("tokens per minute (TPM): Limit 6000, Requested 9681", "groq"),
        ("token limit exceeded", "openrouter"),
        
        # Network errors
        ("no healthy upstream", "groq"),
        ("connection timeout", "mistral"),
        
        # Router errors
        ("500 Server Error: Internal Server Error for url: https://router.huggingface.co", "huggingface"),
        
        # Auth errors
        ("Unauthorized: Invalid API key", "openrouter"),
        ("PERMISSION_DENIED: API key lacks required permissions", "gemini"),
        
        # Unknown errors
        ("Some random error message", "unknown"),
    ]
    
    for error_msg, provider in test_errors:
        class MockError(Exception):
            def __str__(self):
                return error_msg
        
        try:
            error_info = handler.classify_error(MockError(), provider)
            print(f"✅ {provider}: {error_info.error_type.value} - {error_info.description[:50]}...")
        except Exception as e:
            print(f"❌ Error processing {provider} error: {e}")

def test_provider_failure_tracking():
    """Test provider failure tracking functionality."""
    print("\n🧪 Testing provider failure tracking...")
    
    handler = ErrorHandler()
    
    # Test failure tracking
    provider = "test_provider"
    
    # Should not skip initially
    should_skip = handler.should_skip_provider_temporarily(provider)
    print(f"✅ Initial skip status: {should_skip}")
    
    # Add some failures
    for i in range(3):
        handler.handle_provider_failure(provider, "rate_limit")
    
    # Should skip after max failures
    should_skip = handler.should_skip_provider_temporarily(provider)
    print(f"✅ After failures skip status: {should_skip}")
    
    # Test failure stats
    stats = handler.get_provider_failure_stats()
    print(f"✅ Failure stats: {stats}")
    
    # Reset failures
    handler.reset_provider_failures(provider)
    should_skip = handler.should_skip_provider_temporarily(provider)
    print(f"✅ After reset skip status: {should_skip}")

def main():
    """Run all tests."""
    print("🚀 Starting Enhanced Error Handler Tests\n")
    
    try:
        test_status_code_extraction()
        test_vector_similarity()
        test_provider_specific_errors()
        test_comprehensive_error_classification()
        test_provider_failure_tracking()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
