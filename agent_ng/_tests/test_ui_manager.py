"""
Test UI Manager Integration
=========================

Test the UI Manager integration to ensure it works correctly with the modular app.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_ui_manager_import():
    """Test that UI Manager can be imported correctly"""
    try:
        from agent_ng.ui_manager import UIManager, get_ui_manager
        print("✅ UI Manager import successful")
        return True
    except ImportError as e:
        print(f"❌ UI Manager import failed: {e}")
        return False

def test_ui_manager_creation():
    """Test that UI Manager can be created"""
    try:
        from agent_ng.ui_manager import get_ui_manager

        ui_manager = get_ui_manager()
        print("✅ UI Manager creation successful")
        return True
    except Exception as e:
        print(f"❌ UI Manager creation failed: {e}")
        return False

def test_modular_app_with_ui_manager():
    """Test that the modular app works with UI Manager"""
    try:
        from agent_ng.app_ng_modular import NextGenApp

        # Create app instance
        app = NextGenApp()
        print("✅ Modular app creation with UI Manager successful")

        # Check that UI Manager is initialized
        if hasattr(app, 'ui_manager') and app.ui_manager is not None:
            print("✅ UI Manager properly initialized in app")
        else:
            print("❌ UI Manager not properly initialized in app")
            return False

        return True
    except Exception as e:
        print(f"❌ Modular app with UI Manager failed: {e}")
        return False

def test_ui_manager_components():
    """Test that UI Manager can handle components"""
    try:
        from agent_ng.ui_manager import get_ui_manager

        ui_manager = get_ui_manager()

        # Test component management
        ui_manager.components = {"test": "component"}

        components = ui_manager.get_components()
        if "test" in components:
            print("✅ UI Manager component management working")
        else:
            print("❌ UI Manager component management failed")
            return False

        return True
    except Exception as e:
        print(f"❌ UI Manager component test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing UI Manager Integration")
    print("=" * 50)

    tests = [
        test_ui_manager_import,
        test_ui_manager_creation,
        test_ui_manager_components,
        test_modular_app_with_ui_manager,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! UI Manager integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
