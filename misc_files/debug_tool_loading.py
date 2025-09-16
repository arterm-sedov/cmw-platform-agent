#!/usr/bin/env python3
"""
Debug script to understand why CMW platform tools aren't being loaded
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def debug_tool_loading():
    """Debug the tool loading process"""
    print("🔍 Debugging tool loading...")
    
    try:
        import tools.tools as tools_module
        
        print(f"✅ Tools module imported successfully")
        print(f"📁 Module path: {tools_module.__file__}")
        
        # Check a few CMW platform tools
        cmw_tools = ['list_applications', 'list_templates', 'edit_or_create_text_attribute']
        
        for tool_name in cmw_tools:
            if hasattr(tools_module, tool_name):
                tool = getattr(tools_module, tool_name)
                print(f"\n🔧 {tool_name}:")
                print(f"   callable: {callable(tool)}")
                print(f"   has name: {hasattr(tool, 'name')}")
                print(f"   has description: {hasattr(tool, 'description')}")
                print(f"   module: {getattr(tool, '__module__', 'no module')}")
                print(f"   type: {type(tool).__name__}")
                
                if hasattr(tool, 'name'):
                    print(f"   tool name: {tool.name}")
                if hasattr(tool, 'description'):
                    print(f"   tool description: {tool.description[:100]}...")
            else:
                print(f"❌ {tool_name} not found in tools module")
        
        # Now test the filtering logic
        print(f"\n🧪 Testing filtering logic:")
        tool_list = []
        
        for name, obj in tools_module.__dict__.items():
            if (callable(obj) and 
                not name.startswith("_") and 
                not isinstance(obj, type) and  # Exclude classes
                hasattr(obj, '__module__') and  # Must have __module__ attribute
                obj.__module__ == 'tools.tools' and  # Must be from tools module
                name not in ["CmwAgent", "CodeInterpreter"]):  # Exclude specific classes
                
                # Check if it's a proper tool object (has the tool attributes)
                if hasattr(obj, 'name') and hasattr(obj, 'description'):
                    # This is a proper @tool decorated function
                    tool_list.append(obj)
                    print(f"✅ LangChain tool: {name}")
                elif callable(obj) and not name.startswith("_"):
                    # This is a regular function that might be a tool
                    # Only include if it's not an internal function
                    if not name.startswith("_") and name not in [
                        # Exclude built-in types and classes
                        'int', 'str', 'float', 'bool', 'list', 'dict', 'tuple', 'Any', 'BaseModel', 'Field', 'field_validator'
                    ]:
                        tool_list.append(obj)
                        print(f"✅ Function tool: {name}")
        
        print(f"\n📊 Total tools found: {len(tool_list)}")
        print(f"🔧 Tool names: {[getattr(t, 'name', getattr(t, '__name__', str(type(t).__name__))) for t in tool_list[:10]]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_tool_loading()
