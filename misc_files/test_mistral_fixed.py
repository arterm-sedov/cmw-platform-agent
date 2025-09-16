#!/usr/bin/env python3
"""
Fixed test script for LangChain compatibility methods with Mistral LLM
This version properly handles the generator responses
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

def test_mistral_langchain_methods():
    """Test the new LangChain-compatible methods with Mistral"""
    print("🧪 Testing LangChain Compatibility Methods with Mistral (Fixed)")
    print("=" * 70)
    
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
        
        # Test 1: invoke method (fixed)
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
            
            # Check if the output is a generator (which would be wrong)
            if isinstance(result.get('output'), str) and '<generator' in result.get('output', ''):
                print("⚠️  Warning: invoke() returned a generator string instead of actual response")
            else:
                print(f"   Response preview: {str(result.get('output', ''))[:100]}...")
                
        except Exception as e:
            print(f"❌ invoke() failed: {e}")
        
        # Test 2: astream method (with better error handling)
        print("\n📡 Testing astream method with Mistral...")
        try:
            chunks = []
            chunk_count = 0
            for chunk in agent.astream(test_input):
                chunks.append(chunk)
                chunk_count += 1
                if chunk_count <= 3:  # Show first few chunks
                    print(f"   Chunk {chunk_count}: {chunk}")
                if chunk_count >= 10:  # Limit to prevent too much output
                    print("   ... (limiting output)")
                    break
            
            print(f"✅ astream() completed, received {len(chunks)} chunks")
        except Exception as e:
            print(f"❌ astream() failed: {e}")
        
        # Test 3: get_langchain_tools method (fixed)
        print("\n🔧 Testing get_langchain_tools method...")
        try:
            tools = agent.get_langchain_tools()
            print(f"✅ get_langchain_tools() returned {len(tools)} tools")
            if tools:
                print(f"   First tool: {tools[0].name if hasattr(tools[0], 'name') else 'Unknown'}")
            else:
                print("   No tools returned (this might be expected)")
        except Exception as e:
            print(f"❌ get_langchain_tools() failed: {e}")
        
        # Test 4: get_graph method
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
        
        # Test 5: Multi-turn conversation simulation (fixed)
        print("\n💬 Testing multi-turn conversation with Mistral...")
        try:
            # First turn
            turn1_input = {
                "input": "What applications are available?",
                "chat_history": []
            }
            turn1_result = agent.invoke(turn1_input)
            turn1_response = turn1_result.get('output', '')
            print(f"✅ Turn 1 completed: {str(turn1_response)[:100]}...")
            
            # Second turn with context
            turn2_input = {
                "input": "Create an attribute in the first application",
                "chat_history": [
                    {"role": "user", "content": "What applications are available?"},
                    {"role": "assistant", "content": str(turn1_response)}
                ]
            }
            turn2_result = agent.invoke(turn2_input)
            turn2_response = turn2_result.get('output', '')
            print(f"✅ Turn 2 completed: {str(turn2_response)[:100]}...")
            
            # Test streaming version too
            print("\n📡 Testing streaming multi-turn conversation...")
            turn3_input = {
                "input": "What did we just discuss?",
                "chat_history": [
                    {"role": "user", "content": "What applications are available?"},
                    {"role": "assistant", "content": str(turn1_response)},
                    {"role": "user", "content": "Create an attribute in the first application"},
                    {"role": "assistant", "content": str(turn2_response)}
                ]
            }
            
            print("Streaming response:")
            for i, chunk in enumerate(agent.astream(turn3_input)):
                if i < 5:  # Show first 5 chunks
                    print(f"   Chunk {i+1}: {chunk}")
                elif i == 5:
                    print("   ... (continuing to stream)")
                    break
            
        except Exception as e:
            print(f"❌ Multi-turn conversation failed: {e}")
        
        print("\n🎉 Mistral LangChain compatibility test completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the correct directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_direct_agent_call():
    """Test the agent directly to see what it returns"""
    print("\n🔍 Testing Direct Agent Call")
    print("=" * 30)
    
    try:
        from agent import CmwAgent
        agent = CmwAgent(provider="mistral")
        
        # Test direct call
        print("Testing agent('Hello')...")
        result = agent("Hello")
        print(f"Result type: {type(result)}")
        
        if hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
            print("Result is a generator, consuming it...")
            chunks = []
            for chunk in result:
                chunks.append(chunk)
                if len(chunks) >= 3:
                    break
            print(f"Generated {len(chunks)} chunks")
            print(f"First chunk: {chunks[0] if chunks else 'None'}")
        else:
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"❌ Direct call failed: {e}")

if __name__ == "__main__":
    test_mistral_langchain_methods()
    test_direct_agent_call()
