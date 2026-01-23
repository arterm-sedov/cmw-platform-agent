#!/usr/bin/env python3
"""
Script to test all tools to ensure they work properly
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_tool(tool_name, tool_func, test_args=None):
    """Test a single tool"""
    try:
        if test_args is None:
            test_args = {}

        print(f"🧪 Testing {tool_name}...")
        result = tool_func.invoke(test_args)

        if isinstance(result, dict) and result.get('success', False):
            print(f"✅ {tool_name}: SUCCESS")
            return True
        elif isinstance(result, dict) and 'error' in result:
            print(f"❌ {tool_name}: ERROR - {result.get('error', 'Unknown error')}")
            return False
        else:
            print(f"⚠️  {tool_name}: UNEXPECTED RESULT - {type(result)}")
            return False
    except Exception as e:
        print(f"❌ {tool_name}: EXCEPTION - {str(e)}")
        return False

def main():
    """Test all available tools"""
    print("🔧 Testing all tools...\n")

    # Test applications tools
    try:
        from tools.applications_tools.tool_list_applications import list_applications
        test_tool("list_applications", list_applications)
    except Exception as e:
        print(f"❌ Failed to import list_applications: {e}")

    try:
        from tools.applications_tools.tool_list_templates import list_templates
        test_tool("list_templates", list_templates, {
            'application_system_name': 'systemSolution',
            'template_type': 'record'
        })
    except Exception as e:
        print(f"❌ Failed to import list_templates: {e}")

    # Test templates tools
    try:
        from tools.templates_tools.tool_list_attributes import list_attributes
        test_tool("list_attributes", list_attributes, {
            'application_system_name': 'systemSolution',
            'template_system_name': 'Contact'
        })
    except Exception as e:
        print(f"❌ Failed to import list_attributes: {e}")

    # Test some attribute tools
    try:
        from tools.attributes_tools.tools_text_attribute import edit_or_create_text_attribute
        test_tool("edit_or_create_text_attribute", edit_or_create_text_attribute, {
            'operation': 'create',
            'name': 'TestField',
            'system_name': 'testfield',
            'application_system_name': 'systemSolution',
            'template_system_name': 'Contact',
            'display_format': 'PlainText',
            'max_length': 255
        })
    except Exception as e:
        print(f"❌ Failed to import edit_or_create_text_attribute: {e}")

    print("\n🎉 Tool testing completed!")

if __name__ == "__main__":
    main()