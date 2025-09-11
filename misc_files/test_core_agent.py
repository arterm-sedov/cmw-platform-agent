"""
Test script for Core Agent Module
=================================

This script tests the Core Agent functionality to ensure it works correctly.
"""

import time
from core_agent import get_agent, reset_agent


def test_core_agent():
    """Test the Core Agent functionality"""
    print("Testing Core Agent...")
    
    # Reset agent for clean test
    reset_agent()
    agent = get_agent()
    
    # Test 1: Basic question processing
    print("\n1. Testing basic question processing...")
    try:
        response = agent.process_question("What is 2 + 2?")
        print(f"  Question: What is 2 + 2?")
        print(f"  Answer: {response.answer}")
        print(f"  LLM Used: {response.llm_used}")
        print(f"  Confidence: {response.confidence}")
        print(f"  Execution Time: {response.execution_time:.2f}s")
        print(f"  Tool Calls: {len(response.tool_calls)}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 2: Conversation management
    print("\n2. Testing conversation management...")
    try:
        # First message
        response1 = agent.process_question("Hello, my name is Alice", conversation_id="test_conv")
        print(f"  First message: {response1.answer[:50]}...")
        
        # Second message in same conversation
        response2 = agent.process_question("What is my name?", conversation_id="test_conv")
        print(f"  Second message: {response2.answer[:50]}...")
        
        # Check conversation history
        history = agent.get_conversation_history("test_conv")
        print(f"  Conversation length: {len(history)} messages")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 3: Streaming responses
    print("\n3. Testing streaming responses...")
    try:
        print("  Streaming response:")
        for event in agent.process_question_stream("Tell me a short story about a cat"):
            if event["event_type"] == "answer":
                print(f"    Answer: {event['content'][:100]}...")
            elif event["event_type"] in ["start", "success", "error", "warning"]:
                print(f"    {event['event_type'].upper()}: {event['content']}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 4: Agent statistics
    print("\n4. Testing agent statistics...")
    try:
        stats = agent.get_agent_stats()
        print(f"  Total questions: {stats['total_questions']}")
        print(f"  Active conversations: {stats['active_conversations']}")
        print(f"  Tools available: {stats['tools_available']}")
        print(f"  Vector similarity enabled: {stats['vector_similarity_enabled']}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 5: Conversation statistics
    print("\n5. Testing conversation statistics...")
    try:
        conv_stats = agent.get_conversation_stats("test_conv")
        print(f"  Message count: {conv_stats['message_count']}")
        print(f"  User messages: {conv_stats['user_messages']}")
        print(f"  Assistant messages: {conv_stats['assistant_messages']}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 6: Clear conversation
    print("\n6. Testing conversation clearing...")
    try:
        agent.clear_conversation("test_conv")
        history_after_clear = agent.get_conversation_history("test_conv")
        print(f"  Conversation after clear: {len(history_after_clear)} messages")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 7: Error handling
    print("\n7. Testing error handling...")
    try:
        # Try with invalid LLM sequence
        response = agent.process_question("Test question", llm_sequence=["invalid_provider"])
        print(f"  Response with invalid provider: {response.answer[:50]}...")
    except Exception as e:
        print(f"  Error (expected): {e}")
    
    print("\n✓ Core Agent test completed!")


if __name__ == "__main__":
    test_core_agent()
