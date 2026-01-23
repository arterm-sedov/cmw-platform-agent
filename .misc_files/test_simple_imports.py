#!/usr/bin/env python3
"""
Simple test to verify imports work after reorganization
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_simple_imports():
    """Test basic imports"""
    print("🧪 Testing Simple Imports After Reorganization")
    print("=" * 50)

    try:
        # Test agent_ng module
        import agent_ng.agent_ng
        print("✅ agent_ng.agent_ng imported")
    except Exception as e:
        print(f"❌ agent_ng.agent_ng failed: {e}")
        return False

    try:
        # Test agent_old module
        import agent_old.agent
        print("✅ agent_old.agent imported")
    except Exception as e:
        print(f"❌ agent_old.agent failed: {e}")
        return False

    try:
        # Test tools module
        import tools.tool_utils
        print("✅ tools.tool_utils imported")
    except Exception as e:
        print(f"❌ tools.tool_utils failed: {e}")
        return False

    print("\n🎉 All basic imports successful!")
    return True

if __name__ == "__main__":
    test_simple_imports()
