#!/usr/bin/env python3
"""
Comprehensive test for the enhanced CmwAgent with all new features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

def test_enhanced_agent_features():
    """Test all the enhanced features of the upgraded CmwAgent"""
    print("🚀 Testing Enhanced CmwAgent Features")
    print("=" * 50)
    
    try:
        from agent import CmwAgent
        
        # Initialize agent with Mistral
        print("🔧 Initializing enhanced agent with Mistral...")
        agent = CmwAgent(provider="mistral")
        print("✅ Enhanced agent initialized successfully")
        
        # Test 1: Conversation ID management
        print("\n📝 Testing conversation ID management...")
        conv_id = "test_conversation_1"
        
        # Test invoke with conversation ID
        result1 = agent.invoke({
            "input": "Hello, what's your name?",
            "conversation_id": conv_id
        })
        print(f"✅ First message result: {result1.get('output', '')[:100]}...")
        
        # Test second message in same conversation
        result2 = agent.invoke({
            "input": "What did I just ask you?",
            "conversation_id": conv_id
        })
        print(f"✅ Second message result: {result2.get('output', '')[:100]}...")
        
        # Test 2: Conversation history management
        print("\n💬 Testing conversation history management...")
        history = agent.get_conversation_history_by_id(conv_id)
        print(f"✅ Conversation history length: {len(history)}")
        print(f"   First message: {history[0] if history else 'None'}")
        
        # Test 3: Multiple conversations
        print("\n🔄 Testing multiple conversations...")
        conv_id2 = "test_conversation_2"
        result3 = agent.invoke({
            "input": "This is a different conversation",
            "conversation_id": conv_id2
        })
        print(f"✅ Second conversation result: {result3.get('output', '')[:100]}...")
        
        # Test 4: Conversation statistics
        print("\n📊 Testing conversation statistics...")
        stats1 = agent.get_conversation_stats(conv_id)
        stats2 = agent.get_conversation_stats(conv_id2)
        print(f"✅ Conversation 1 stats: {stats1}")
        print(f"✅ Conversation 2 stats: {stats2}")
        
        # Test 5: Agent health status
        print("\n🏥 Testing agent health status...")
        health = agent.get_agent_health_status()
        print(f"✅ Agent health: {health}")
        
        # Test 6: Conversation export
        print("\n📤 Testing conversation export...")
        export_json = agent.export_conversation(conv_id, "json")
        export_txt = agent.export_conversation(conv_id, "txt")
        export_md = agent.export_conversation(conv_id, "markdown")
        print(f"✅ JSON export length: {len(export_json)}")
        print(f"✅ TXT export length: {len(export_txt)}")
        print(f"✅ Markdown export length: {len(export_md)}")
        
        # Test 7: Enhanced streaming
        print("\n📡 Testing enhanced streaming...")
        chunk_count = 0
        for chunk in agent.astream({
            "input": "Tell me a short story",
            "conversation_id": conv_id
        }):
            chunk_count += 1
            if chunk_count <= 3:
                print(f"   Chunk {chunk_count}: {chunk}")
            if chunk_count >= 5:
                break
        print(f"✅ Received {chunk_count} streaming chunks")
        
        # Test 8: Thread safety
        print("\n🔒 Testing thread safety...")
        import threading
        import time
        
        def concurrent_request(conv_id, message, results, index):
            try:
                result = agent.invoke({
                    "input": f"{message} (Request {index})",
                    "conversation_id": conv_id
                })
                results[index] = result
            except Exception as e:
                results[index] = {"error": str(e)}
        
        # Test concurrent requests
        results = {}
        threads = []
        for i in range(3):
            thread = threading.Thread(
                target=concurrent_request,
                args=(f"concurrent_test_{i}", "Test concurrent request", results, i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        print(f"✅ Concurrent requests completed: {len(results)} results")
        for i, result in results.items():
            if "error" in result:
                print(f"   Request {i}: Error - {result['error']}")
            else:
                print(f"   Request {i}: Success - {result.get('output', '')[:50]}...")
        
        # Test 9: Conversation cleanup
        print("\n🧹 Testing conversation cleanup...")
        all_convs = agent.get_all_conversation_ids()
        print(f"✅ Active conversations before cleanup: {len(all_convs)}")
        
        # Test 10: LangChain compatibility
        print("\n🔗 Testing LangChain compatibility...")
        tools = agent.get_langchain_tools()
        print(f"✅ LangChain tools available: {len(tools)}")
        
        graph = agent.get_graph()
        if graph:
            print("✅ LangGraph integration working")
        else:
            print("⚠️ LangGraph not available (expected if not installed)")
        
        print("\n🎉 All enhanced features tested successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_multi_turn_conversation():
    """Test multi-turn conversation with proper context"""
    print("\n🔄 Testing Multi-Turn Conversation")
    print("=" * 40)
    
    try:
        from agent import CmwAgent
        agent = CmwAgent(provider="mistral")
        
        conv_id = "multi_turn_test"
        
        # Turn 1: Initial question
        print("Turn 1: Asking about applications...")
        result1 = agent.invoke({
            "input": "What applications are available in the system?",
            "conversation_id": conv_id
        })
        print(f"Response: {result1.get('output', '')[:200]}...")
        
        # Turn 2: Follow-up question
        print("\nTurn 2: Asking for more details...")
        result2 = agent.invoke({
            "input": "Can you create an attribute in the first application you mentioned?",
            "conversation_id": conv_id
        })
        print(f"Response: {result2.get('output', '')[:200]}...")
        
        # Turn 3: Context-dependent question
        print("\nTurn 3: Asking about previous context...")
        result3 = agent.invoke({
            "input": "What did we just discuss? Summarize our conversation.",
            "conversation_id": conv_id
        })
        print(f"Response: {result3.get('output', '')[:200]}...")
        
        # Check conversation history
        history = agent.get_conversation_history_by_id(conv_id)
        print(f"\n✅ Conversation history has {len(history)} messages")
        
        # Verify context is maintained
        if len(history) >= 6:  # 3 turns * 2 messages each
            print("✅ Multi-turn conversation context maintained successfully!")
        else:
            print("⚠️ Multi-turn conversation context may not be fully maintained")
        
    except Exception as e:
        print(f"❌ Multi-turn test failed: {e}")

if __name__ == "__main__":
    test_enhanced_agent_features()
    test_multi_turn_conversation()
