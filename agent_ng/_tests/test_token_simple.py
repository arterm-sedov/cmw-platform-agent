"""
Simple Token Test
================

Test token counting without full agent initialization.
"""

def test_token_counter_import():
    """Test that token counter can be imported"""
    try:
        from agent_ng.token_counter import get_token_tracker, TokenCount
        print("✅ Token counter imports successfully")
        return True
    except Exception as e:
        print(f"❌ Token counter import failed: {e}")
        return False

def test_token_tracker_creation():
    """Test that token tracker can be created"""
    try:
        from agent_ng.token_counter import get_token_tracker
        tracker = get_token_tracker()
        print("✅ Token tracker created successfully")
        print(f"   - Has tiktoken_counter: {hasattr(tracker, 'tiktoken_counter')}")
        print(f"   - Has api_counter: {hasattr(tracker, 'api_counter')}")
        return True
    except Exception as e:
        print(f"❌ Token tracker creation failed: {e}")
        return False

def test_token_counting():
    """Test basic token counting"""
    try:
        from agent_ng.token_counter import get_token_tracker
        tracker = get_token_tracker()
        
        # Test counting a simple message
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content="Hello world")]
        
        token_count = tracker.count_prompt_tokens(messages)
        print(f"✅ Token counting works: {token_count.total_tokens} tokens")
        print(f"   - Source: {token_count.source}")
        print(f"   - Estimated: {token_count.is_estimated}")
        return True
    except Exception as e:
        print(f"❌ Token counting failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing token counting integration...")
    print()
    
    success = True
    success &= test_token_counter_import()
    print()
    success &= test_token_tracker_creation()
    print()
    success &= test_token_counting()
    print()
    
    if success:
        print("🎉 All tests passed! Token counting is working.")
    else:
        print("❌ Some tests failed. Check the errors above.")
