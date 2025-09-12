#!/usr/bin/env python3
"""
Test script to verify all tools are properly loaded
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_ng.core_agent import CoreAgent

def test_tool_loading():
    """Test that all tools are properly loaded"""
    print("🧪 Testing tool loading...")
    
    try:
        # Initialize the core agent
        agent = CoreAgent()
        
        print(f"\n📊 Tool Loading Results:")
        print(f"Total tools loaded: {len(agent.tools)}")
        
        # Categorize tools
        langchain_tools = []
        function_tools = []
        
        for tool in agent.tools:
            if hasattr(tool, 'name') and hasattr(tool, 'description'):
                langchain_tools.append(tool.name)
            else:
                function_tools.append(tool.__name__ if hasattr(tool, '__name__') else str(tool))
        
        print(f"\n🔧 LangChain Tools ({len(langchain_tools)}):")
        for tool_name in sorted(langchain_tools):
            print(f"  - {tool_name}")
        
        print(f"\n⚙️ Function Tools ({len(function_tools)}):")
        for tool_name in sorted(function_tools):
            print(f"  - {tool_name}")
        
        # Check for specific tool categories
        cmw_tools = [t for t in agent.tools if any(keyword in str(t) for keyword in ['attribute', 'application', 'template'])]
        math_tools = [t for t in agent.tools if any(keyword in str(t) for keyword in ['add', 'multiply', 'divide', 'subtract', 'power', 'sqrt'])]
        search_tools = [t for t in agent.tools if any(keyword in str(t) for keyword in ['search', 'wiki', 'web', 'arxiv'])]
        
        print(f"\n📋 Tool Categories:")
        print(f"  - CMW Platform tools: {len(cmw_tools)}")
        print(f"  - Math tools: {len(math_tools)}")
        print(f"  - Search tools: {len(search_tools)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing tool loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tool_loading()
    if success:
        print("\n✅ Tool loading test completed successfully!")
    else:
        print("\n❌ Tool loading test failed!")
        sys.exit(1)
