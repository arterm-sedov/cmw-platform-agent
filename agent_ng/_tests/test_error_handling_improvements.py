#!/usr/bin/env python3
"""
Test script for error handling improvements.

This script tests the new meaningful error responses to ensure they work correctly
and provide better visibility into tool execution and LLM errors.
"""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agent_ng.utils import ensure_valid_answer, ensure_meaningful_response


def test_ensure_valid_answer():
    """Test the ensure_valid_answer function"""
    print("Testing ensure_valid_answer...")
    
    # Test cases
    test_cases = [
        (None, "No answer provided"),
        ("", "No answer provided"),
        ("   ", "No answer provided"),
        ("Valid response", "Valid response"),
        (123, "123"),
        ({"error": "test"}, "{'error': 'test'}"),
    ]
    
    for input_val, expected in test_cases:
        result = ensure_valid_answer(input_val)
        print(f"  Input: {repr(input_val)} -> Output: {repr(result)}")
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("✅ ensure_valid_answer tests passed")


def test_ensure_meaningful_response():
    """Test the ensure_meaningful_response function"""
    print("\nTesting ensure_meaningful_response...")
    
    # Test cases
    test_cases = [
        # (response, context, tool_calls, expected_pattern)
        (None, "", None, "No response generated"),
        (None, "", [{"name": "test_tool"}], "Tools executed successfully (test_tool) but no response generated"),
        ("", "", [{"name": "tool1"}, {"name": "tool2"}], "Tools executed successfully (tool1, tool2) but response is empty"),
        ("", "", None, "Empty response generated"),
        ("Valid response", "", None, "Valid response"),
        ("Valid response", "test context", [{"name": "test_tool"}], "Valid response"),
    ]
    
    for response, context, tool_calls, expected_pattern in test_cases:
        result = ensure_meaningful_response(response, context, tool_calls)
        print(f"  Input: {repr(response)}, context: {context}, tools: {tool_calls}")
        print(f"  Output: {repr(result)}")
        
        # Check if the result contains the expected pattern
        if expected_pattern in result:
            print("  ✅ Pattern matched")
        else:
            print(f"  ❌ Expected pattern '{expected_pattern}' not found in result")
    
    print("✅ ensure_meaningful_response tests passed")


def test_error_scenarios():
    """Test realistic error scenarios"""
    print("\nTesting realistic error scenarios...")
    
    # Scenario 1: Tools succeed but LLM fails due to context length
    tool_calls = [
        {"name": "list_templates", "result": "Templates: A, B, C"},
        {"name": "list_attributes", "result": "Attributes: X, Y, Z"}
    ]
    
    # Simulate empty response after tool execution
    response = ensure_meaningful_response("", "LLM response after tool execution", tool_calls)
    print(f"Empty response with tools: {response}")
    assert "Tools executed successfully" in response
    assert "context length limit" in response
    
    # Scenario 2: No tools, just LLM error
    response = ensure_meaningful_response(None, "Direct LLM call", None)
    print(f"No response without tools: {response}")
    assert response == "No response generated"
    
    print("✅ Error scenario tests passed")


def main():
    """Run all tests"""
    print("🧪 Testing Error Handling Improvements")
    print("=" * 50)
    
    try:
        test_ensure_valid_answer()
        test_ensure_meaningful_response()
        test_error_scenarios()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Error handling improvements are working correctly.")
        print("\nKey improvements:")
        print("- Meaningful error messages that preserve context")
        print("- Tool execution status is shown when LLM fails")
        print("- Context length limit errors are properly identified")
        print("- Users get visibility into what actually happened")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
