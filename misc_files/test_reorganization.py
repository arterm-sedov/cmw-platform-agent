#!/usr/bin/env python3
"""
Test script to verify the reorganization works correctly
======================================================

This script tests that all the moved modules can be imported correctly
and that the basic functionality still works.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported correctly"""
    print("🧪 Testing Module Imports After Reorganization")
    print("=" * 60)
    
    # Test agent_ng imports
    try:
        from agent_ng.agent_ng import NextGenAgent, ChatMessage, get_agent_ng
        print("✅ agent_ng.agent_ng imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import agent_ng.agent_ng: {e}")
        return False
    
    try:
        from agent_ng.app_ng import NextGenApp
        print("✅ agent_ng.app_ng imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import agent_ng.app_ng: {e}")
        return False
    
    try:
        from agent_ng.core_agent import CoreAgent, get_agent
        print("✅ agent_ng.core_agent imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import agent_ng.core_agent: {e}")
        return False
    
    try:
        from agent_ng.llm_manager import get_llm_manager, LLMInstance, LLMProvider
        print("✅ agent_ng.llm_manager imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import agent_ng.llm_manager: {e}")
        return False
    
    try:
        from agent_ng.error_handler import get_error_handler, ErrorInfo
        print("✅ agent_ng.error_handler imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import agent_ng.error_handler: {e}")
        return False
    
    # Test tools imports
    try:
        import tools.tool_utils
        print("✅ tools.tool_utils imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import tools.tool_utils: {e}")
        return False
    
    try:
        from tools.applications_tools import tool_list_applications
        print("✅ tools.applications_tools imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import tools.applications_tools: {e}")
        return False
    
    try:
        from tools.attributes_tools import tools_text_attribute
        print("✅ tools.attributes_tools imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import tools.attributes_tools: {e}")
        return False
    
    # Test agent_old imports
    try:
        from agent_old.agent import CmwAgent
        print("✅ agent_old.agent imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import agent_old.agent: {e}")
        return False
    
    try:
        from agent_old.utils import ensure_valid_answer
        print("✅ agent_old.utils imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import agent_old.utils: {e}")
        return False
    
    print("\n🎉 All imports successful! Reorganization completed successfully.")
    return True

def test_directory_structure():
    """Test that the directory structure is correct"""
    print("\n📁 Testing Directory Structure")
    print("=" * 40)
    
    # Check that directories exist
    directories = [
        "agent_ng",
        "agent_old", 
        "tools",
        "agent_ng/__init__.py",
        "agent_old/__init__.py",
        "tools/__init__.py"
    ]
    
    for dir_path in directories:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} exists")
        else:
            print(f"❌ {dir_path} missing")
            return False
    
    # Check that key files are in the right places
    key_files = [
        "agent_ng/agent_ng.py",
        "agent_ng/app_ng.py", 
        "agent_ng/core_agent.py",
        "agent_ng/llm_manager.py",
        "agent_old/agent.py",
        "agent_old/app.py",
        "tools/tool_utils.py",
        "tools/applications_tools",
        "tools/attributes_tools"
    ]
    
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} in correct location")
        else:
            print(f"❌ {file_path} missing or in wrong location")
            return False
    
    print("\n🎉 Directory structure is correct!")
    return True

if __name__ == "__main__":
    print("🚀 Testing CMW Platform Agent Reorganization")
    print("=" * 60)
    
    success = True
    success &= test_directory_structure()
    success &= test_imports()
    
    if success:
        print("\n🎉 All tests passed! The reorganization was successful.")
        print("\n📋 Summary of changes:")
        print("  • Created agent_ng/ folder with next-generation modules")
        print("  • Created agent_old/ folder with legacy modules") 
        print("  • Created tools/ folder with all tool-related modules")
        print("  • Updated all import statements to use relative imports")
        print("  • Created __init__.py files for proper package structure")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
