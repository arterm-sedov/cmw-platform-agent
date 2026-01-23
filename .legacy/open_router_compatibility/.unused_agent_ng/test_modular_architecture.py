"""
Test script for the modular architecture
=======================================

This script tests all the new modular components to ensure they work together correctly.
"""

import time
from trace_manager import get_trace_manager, reset_trace_manager
from token_manager import get_token_manager, reset_token_manager
from message_processor import get_message_processor, reset_message_processor
from tool_manager import get_tool_manager, reset_tool_manager
from vector_similarity import get_vector_similarity_manager, reset_vector_similarity_manager
from stats_manager import get_stats_manager, reset_stats_manager
from response_processor import get_response_processor, reset_response_processor
# from streaming_manager import get_streaming_manager, reset_streaming_manager  # Moved to .unused


def test_all_modules():
    """Test all modular components"""
    print("🧪 Testing Modular Architecture Components...")

    # Test 1: Trace Manager
    print("\n1. Testing Trace Manager...")
    trace_mgr = get_trace_manager()
    trace_mgr.init_question("What is 2 + 2?")
    trace_mgr.add_debug_output("Testing debug output", "test")
    trace_mgr.finalize_question({'answer': '4', 'llm_used': 'test'})
    trace_stats = trace_mgr.get_stats()
    print(f"  ✅ Trace Manager: {trace_stats['total_questions']} questions tracked")

    # Test 2: Token Manager
    print("\n2. Testing Token Manager...")
    token_mgr = get_token_manager()
    tokens = token_mgr.estimate_tokens("Hello world, this is a test message.")
    chunks = token_mgr.create_chunks(["Short text", "Another short text"], 10)
    print(f"  ✅ Token Manager: {tokens} tokens estimated, {len(chunks)} chunks created")

    # Test 3: Message Processor
    print("\n3. Testing Message Processor...")
    msg_proc = get_message_processor()
    messages = msg_proc.format_messages_simple("What is 5 + 5?")
    print(f"  ✅ Message Processor: {len(messages)} messages formatted")

    # Test 4: Tool Manager
    print("\n4. Testing Tool Manager...")
    tool_mgr = get_tool_manager()
    available_tools = tool_mgr.get_tool_names()
    print(f"  ✅ Tool Manager: {len(available_tools)} tools available")

    # Test 5: Vector Similarity Manager
    print("\n5. Testing Vector Similarity Manager...")
    vsm = get_vector_similarity_manager(enabled=False)  # Disable for testing
    is_match, similarity = vsm.answers_match("The answer is 4", "4")
    print(f"  ✅ Vector Similarity: Match={is_match}, Similarity={similarity:.2f}")

    # Test 6: Stats Manager
    print("\n6. Testing Stats Manager...")
    stats_mgr = get_stats_manager()
    stats_mgr.track_llm_usage("gemini", "success", 1.5)
    stats_mgr.track_conversation("test_conv", "Test question", "Test answer", "gemini")
    stats = stats_mgr.get_comprehensive_stats()
    print(f"  ✅ Stats Manager: {stats['system_stats']['total_requests']} requests tracked")

    # Test 7: Response Processor
    print("\n7. Testing Response Processor...")
    resp_proc = get_response_processor()
    from langchain_core.messages import AIMessage
    test_response = AIMessage(content="The answer is 42")
    answer = resp_proc.extract_final_answer(test_response)
    print(f"  ✅ Response Processor: Extracted answer: '{answer}'")

    # Test 8: Streaming Manager (moved to .unused)
    print("\n8. Testing Streaming Manager...")
    print("  ⚠️ Streaming Manager moved to .unused - using native LangChain streaming")

    print("\n🎉 All modules tested successfully!")
    return True


def test_module_integration():
    """Test how modules work together"""
    print("\n🔗 Testing Module Integration...")

    # Create a simple workflow
    trace_mgr = get_trace_manager()
    token_mgr = get_token_manager()
    msg_proc = get_message_processor()
    tool_mgr = get_tool_manager()
    stats_mgr = get_stats_manager()
    resp_proc = get_response_processor()

    # Simulate a question processing workflow
    question = "What is 10 multiplied by 5?"

    # 1. Initialize trace
    trace_mgr.init_question(question)

    # 2. Format messages
    messages = msg_proc.format_messages_simple(question)

    # 3. Estimate tokens
    token_count = token_mgr.estimate_messages_tokens(messages)

    # 4. Track in stats
    stats_mgr.track_llm_usage("test_llm", "success", 0.5)

    # 5. Simulate response processing
    from langchain_core.messages import AIMessage
    response = AIMessage(content="The answer is 50")
    answer = resp_proc.extract_final_answer(response)

    # 6. Finalize trace
    trace_mgr.finalize_question({
        'answer': answer,
        'llm_used': 'test_llm',
        'reference': '50'
    })

    # 7. Track conversation
    stats_mgr.track_conversation("integration_test", question, answer, "test_llm")

    print(f"  ✅ Integration test completed:")
    print(f"    - Question: {question}")
    print(f"    - Answer: {answer}")
    print(f"    - Tokens: {token_count}")
    print(f"    - Messages: {len(messages)}")

    return True


def test_module_reset():
    """Test module reset functionality"""
    print("\n🔄 Testing Module Reset...")

    # Test individual resets
    reset_trace_manager()
    reset_token_manager()
    reset_message_processor()
    reset_tool_manager()
    reset_vector_similarity_manager()
    reset_stats_manager()
    reset_response_processor()
    # reset_streaming_manager()  # Moved to .unused

    print("  ✅ All modules reset successfully")
    return True


def main():
    """Main test function"""
    print("🚀 Starting Modular Architecture Tests...")

    try:
        # Test individual modules
        test_all_modules()

        # Test module integration
        test_module_integration()

        # Test module reset
        test_module_reset()

        print("\n🎉 All tests passed! Modular architecture is working correctly.")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
