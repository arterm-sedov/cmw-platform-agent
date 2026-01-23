#!/usr/bin/env python3
"""
Simple test to verify the fixed invoke() method works correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

def test_simple_invoke():
    """Test the fixed invoke method with a simple question"""
    print("🧪 Testing Fixed invoke() Method")
    print("=" * 40)

    try:
        from agent import CmwAgent

        # Initialize agent with Mistral
        print("🔧 Initializing agent...")
        agent = CmwAgent(provider="mistral")
        print("✅ Agent initialized")

        # Test invoke with a simple question
        print("\n📝 Testing invoke() with simple question...")
        result = agent.invoke({
            "input": "Hello, how are you?",
            "chat_history": []
        })

        print(f"✅ invoke() result type: {type(result)}")
        print(f"✅ Has 'output' key: {'output' in result}")
        print(f"✅ Output preview: {str(result.get('output', ''))[:200]}...")

        # Check if it's actually a string response, not a generator
        output = result.get('output', '')
        if '<generator' in str(output):
            print("❌ Still returning generator string!")
        else:
            print("✅ Successfully returning actual response!")

    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_simple_invoke()
