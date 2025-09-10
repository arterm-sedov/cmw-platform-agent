#!/usr/bin/env python3
"""
Example usage of Mistral AI integration in the Comindware Analyst Copilot
"""

import os
from dotenv import load_dotenv
from agent import CmwAgent

# Load environment variables
load_dotenv()

def example_mistral_usage():
    """Example of using the agent with Mistral AI integration"""
    
    # Check if Mistral API key is available
    if not os.getenv('MISTRAL_API_KEY'):
        print("❌ MISTRAL_API_KEY not found in environment variables")
        print("Please set MISTRAL_API_KEY in your .env file")
        return
    
    print("🚀 Initializing agent with Mistral AI integration...")
    
    # Initialize the agent
    agent = CmwAgent(provider="groq")  # provider doesn't matter, it will try all LLMs
    
    # Check if Mistral was successfully initialized
    if "mistral" in agent.llm_provider_names:
        print("✅ Mistral AI successfully integrated!")
        print(f"Available LLMs: {agent.llm_provider_names}")
    else:
        print("⚠️ Mistral AI not found in initialized LLMs")
        print(f"Available LLMs: {agent.llm_provider_names}")
        return
    
    # Example questions to test
    test_questions = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "What are the main benefits of renewable energy?"
    ]
    
    print("\n🧪 Testing Mistral AI with example questions...")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 30)
        
        try:
            # Get answer from agent
            result = agent(question)
            
            print(f"Answer: {result.get('answer', 'No answer provided')}")
            print(f"LLM Used: {result.get('llm_used', 'Unknown')}")
            print(f"Similarity Score: {result.get('similarity_score', 'N/A')}")
            
            if result.get('error'):
                print(f"Error: {result['error']}")
                
        except Exception as e:
            print(f"❌ Error processing question: {e}")
        
        print("-" * 30)

def test_mistral_specific():
    """Test using Mistral AI specifically"""
    
    if not os.getenv('MISTRAL_API_KEY'):
        print("❌ MISTRAL_API_KEY not found")
        return
    
    print("\n🎯 Testing Mistral AI specifically...")
    
    try:
        from langchain_mistralai.chat_models import ChatMistralAI
        from langchain_core.messages import HumanMessage
        
        # Initialize Mistral AI directly
        chat = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0,
            max_tokens=200
        )
        
        # Test message
        messages = [HumanMessage(content="What is artificial intelligence?")]
        response = chat.invoke(messages)
        
        print(f"✅ Direct Mistral AI response: {response.content}")
        
    except Exception as e:
        print(f"❌ Direct Mistral AI test failed: {e}")

if __name__ == "__main__":
    print("🎯 Mistral AI Integration Example")
    print("=" * 40)
    
    # Test direct Mistral usage
    test_mistral_specific()
    
    # Test agent integration
    example_mistral_usage()
    
    print("\n" + "=" * 40)
    print("✅ Example completed!")
