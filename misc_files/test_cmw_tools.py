#!/usr/bin/env python3
"""
Test script to verify CMW tools are accessible to the agent
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_ng.core_agent import CoreAgent

def test_cmw_tools():
    """Test that CMW tools are accessible"""
    print("🧪 Testing CMW tools accessibility...")
    
    try:
        # Initialize the core agent
        agent = CoreAgent()
        
        # Check for specific CMW tools
        cmw_tool_names = [
            'list_applications', 'list_templates', 'list_attributes',
            'edit_or_create_text_attribute', 'get_text_attribute',
            'edit_or_create_date_time_attribute', 'get_date_time_attribute',
            'edit_or_create_numeric_attribute', 'get_numeric_attribute',
            'edit_or_create_record_attribute', 'get_record_attribute',
            'edit_or_create_image_attribute', 'get_image_attribute',
            'edit_or_create_drawing_attribute', 'get_drawing_attribute',
            'edit_or_create_document_attribute', 'get_document_attribute',
            'edit_or_create_duration_attribute', 'get_duration_attribute',
            'edit_or_create_account_attribute', 'get_account_attribute',
            'edit_or_create_boolean_attribute', 'get_boolean_attribute',
            'edit_or_create_role_attribute', 'get_role_attribute',
            'delete_attribute', 'archive_or_unarchive_attribute'
        ]
        
        print(f"\n🔍 Checking for CMW tools...")
        found_tools = []
        missing_tools = []
        
        for tool_name in cmw_tool_names:
            # Check if tool exists in the tools list
            tool_found = False
            for tool in agent.tools:
                if hasattr(tool, 'name') and tool.name == tool_name:
                    found_tools.append(tool_name)
                    tool_found = True
                    break
                elif hasattr(tool, '__name__') and tool.__name__ == tool_name:
                    found_tools.append(tool_name)
                    tool_found = True
                    break
            
            if not tool_found:
                missing_tools.append(tool_name)
        
        print(f"\n✅ Found CMW tools ({len(found_tools)}):")
        for tool in sorted(found_tools):
            print(f"  - {tool}")
        
        if missing_tools:
            print(f"\n❌ Missing CMW tools ({len(missing_tools)}):")
            for tool in sorted(missing_tools):
                print(f"  - {tool}")
        
        # Test a simple CMW tool call
        print(f"\n🧪 Testing tool call...")
        try:
            # Find the list_applications tool
            list_apps_tool = None
            for tool in agent.tools:
                if hasattr(tool, 'name') and tool.name == 'list_applications':
                    list_apps_tool = tool
                    break
            
            if list_apps_tool:
                print(f"✅ Found list_applications tool: {list_apps_tool.name}")
                print(f"   Description: {list_apps_tool.description}")
                print(f"   Args schema: {list_apps_tool.args_schema}")
            else:
                print("❌ list_applications tool not found")
                
        except Exception as e:
            print(f"❌ Error testing tool call: {e}")
        
        return len(missing_tools) == 0
        
    except Exception as e:
        print(f"❌ Error testing CMW tools: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cmw_tools()
    if success:
        print("\n✅ CMW tools test completed successfully!")
    else:
        print("\n❌ CMW tools test failed!")
        sys.exit(1)
