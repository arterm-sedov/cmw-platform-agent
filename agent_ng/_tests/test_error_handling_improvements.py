#!/usr/bin/env python3
"""
Test script for lean error handling improvements.

This script tests the lean error handling approach that passes through
all errors to users without hardcoded workarounds.
"""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agent_ng.utils import ensure_valid_answer


def test_ensure_valid_answer():
    """Test the ensure_valid_answer function"""
    print("Testing ensure_valid_answer...")
    
    # Test cases - only "No answer provided" when literally no answer
    test_cases = [
        (None, "No answer provided"),
        ("", "No answer provided"),
        ("   ", "No answer provided"),
        ("Valid response", "Valid response"),
        (123, "123"),
        ({"error": "test"}, "{'error': 'test'}"),
        ("Error: Tool failed", "Error: Tool failed"),
        ("LLM Error: Context length exceeded", "LLM Error: Context length exceeded"),
    ]
    
    for input_val, expected in test_cases:
        result = ensure_valid_answer(input_val)
        print(f"  Input: {repr(input_val)} -> Output: {repr(result)}")
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("✅ ensure_valid_answer tests passed")


def test_error_passthrough():
    """Test that errors are passed through without modification"""
    print("\nTesting error passthrough...")
    
    # Test that all error types are passed through as-is
    error_cases = [
        "Error: Tool 'unknown_tool' not found",
        "Error executing tool 'test_tool': Connection timeout",
        "LLM Error: This endpoint's maximum context length is 163840 tokens",
        "Error: API key not found",
        "Exception: Network connection failed",
    ]
    
    for error_msg in error_cases:
        result = ensure_valid_answer(error_msg)
        print(f"  Error: {error_msg}")
        print(f"  Result: {result}")
        assert result == error_msg, f"Error should be passed through unchanged: {error_msg}"
        print("  ✅ Error passed through correctly")
    
    print("✅ Error passthrough tests passed")


def test_no_answer_scenarios():
    """Test that 'No answer provided' only appears when there's literally no answer"""
    print("\nTesting 'No answer provided' scenarios...")
    
    # Only these should result in "No answer provided"
    no_answer_cases = [
        (None, "No answer provided"),
        ("", "No answer provided"),
        ("   ", "No answer provided"),
    ]
    
    for input_val, expected in no_answer_cases:
        result = ensure_valid_answer(input_val)
        print(f"  Input: {repr(input_val)} -> Output: {repr(result)}")
        assert result == expected, f"Expected {expected}, got {result}"
    
    # These should NOT result in "No answer provided"
    has_answer_cases = [
        (" ", " "),  # Single space is not empty
        ("\n", "\n"),  # Newline is not empty
        ("0", "0"),  # Zero is not empty
        ("false", "false"),  # False string is not empty
    ]
    
    for input_val, expected in has_answer_cases:
        result = ensure_valid_answer(input_val)
        print(f"  Input: {repr(input_val)} -> Output: {repr(result)}")
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("✅ 'No answer provided' scenarios tests passed")


def main():
    """Run all tests"""
    print("🧪 Testing Lean Error Handling")
    print("=" * 50)
    
    try:
        test_ensure_valid_answer()
        test_error_passthrough()
        test_no_answer_scenarios()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Lean error handling is working correctly.")
        print("\nKey principles:")
        print("- All errors are passed through to users without modification")
        print("- 'No answer provided' only when there's literally no answer")
        print("- No hardcoded workarounds or specific error handling")
        print("- Agent continues working even when tools return errors")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
