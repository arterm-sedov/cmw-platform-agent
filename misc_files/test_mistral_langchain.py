#!/usr/bin/env python3
"""
Test script for LangChain compatibility methods with Mistral LLM
Run this in WSL to test the new LangChain-compatible methods
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

def test_mistral_langchain_methods():
    """Test the new LangChain-compatible methods with Mistral"""
    print("🧪 Testing LangChain Compatibility Methods with Mistral")
    print("=" * 60)
    
    # Check if MISTRAL_API_KEY is set
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        print("❌ MISTRAL_API_KEY not found in environment variables")
        print("Please set your Mistral API key in .env file or environment")
        return
    
    print(f"✅ MISTRAL_API_KEY found: {mistral_key[:10]}...")
    
    try:
        # Import your agent
        from agent import CmwAgent
        print("✅ Successfully imported CmwAgent")
        
        # Initialize agent with Mistral
        print("\n🔧 Initializing agent with Mistral...")
        agent = CmwAgent(provider="mistral")
        print("✅ Mistral agent initialized successfully")
        
        # Test 1: invoke method
        print("\n📝 Testing invoke method with Mistral...")
        test_input = {
            "input": "What applications are available?",
            "chat_history": []
        }
        
        try:
            result = agent.invoke(test_input)
            print(f"✅ invoke() result: {result}")
            print(f"   Output type: {type(result)}")
            print(f"   Has 'output' key: {'output' in result}")
        except Exception as e:
            print(f"❌ invoke() failed: {e}")
        
        # Test 2: astream method
        print("\n📡 Testing astream method with Mistral...")
        try:
            chunks = []
            for chunk in agent.astream(test_input):
                chunks.append(chunk)
                if len(chunks) <= 3:  # Show first few chunks
                    print(f"   Chunk {len(chunks)}: {chunk}")
            
            print(f"✅ astream() completed, received {len(chunks)} chunks")
        except Exception as e:
            print(f"❌ astream() failed: {e}")
        
        # Test 3: get_langchain_tools method
        print("\n🔧 Testing get_langchain_tools method...")
        try:
            tools = agent.get_langchain_tools()
            print(f"✅ get_langchain_tools() returned {len(tools)} tools")
            if tools:
                print(f"   First tool: {tools[0].name if hasattr(tools[0], 'name') else 'Unknown'}")
        except Exception as e:
            print(f"❌ get_langchain_tools() failed: {e}")
        
        # Test 4: get_graph method (if LangGraph is available)
        print("\n🕸️ Testing get_graph method...")
        try:
            graph = agent.get_graph()
            if graph:
                print("✅ get_graph() returned a graph successfully")
                print(f"   Graph type: {type(graph)}")
            else:
                print("⚠️ get_graph() returned None (LangGraph not available)")
        except Exception as e:
            print(f"❌ get_graph() failed: {e}")
        
        # Test 5: Multi-turn conversation simulation
        print("\n💬 Testing multi-turn conversation with Mistral...")
        try:
            # First turn
            turn1_input = {
                "input": "What applications are available?",
                "chat_history": []
            }
            turn1_result = agent.invoke(turn1_input)
            print(f"✅ Turn 1 completed: {turn1_result.get('output', '')[:100]}...")
            
            # Second turn with context
            turn2_input = {
                "input": "Create an attribute in the first application",
                "chat_history": [
                    {"role": "user", "content": "What applications are available?"},
                    {"role": "assistant", "content": turn1_result.get('output', '')}
                ]
            }
            turn2_result = agent.invoke(turn2_input)
            print(f"✅ Turn 2 completed: {turn2_result.get('output', '')[:100]}...")
            
        except Exception as e:
            print(f"❌ Multi-turn conversation failed: {e}")
        
        print("\n🎉 Mistral LangChain compatibility test completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the correct directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_quick_mistral():
    """Quick test with Mistral"""
    print("\n🚀 Quick Mistral Test")
    print("=" * 30)
    
    try:
        from agent import CmwAgent
        agent = CmwAgent(provider="mistral")
        
        # Quick test
        result = agent.invoke({"input": "Hello", "chat_history": []})
        print(f"✅ Quick test result: {result.get('output', '')[:50]}...")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")

if __name__ == "__main__":
    test_mistral_langchain_methods()
    test_quick_mistral()
