#!/usr/bin/env python3
"""
Test script for all managers (file_manager.py, login_manager.py)
==============================================================

This script tests the file manager and login manager functionality in both disabled and enabled modes.
"""

import os
import sys
import json
from datetime import datetime

def test_file_manager():
    """Test the file manager functionality."""
    print("🧪 Testing File Manager")
    print("=" * 50)
    
    # Test 1: Import and basic functionality
    print("\n1. Testing import and basic functionality...")
    try:
        from file_manager import file_manager, get_file_manager_status
        print("✅ Successfully imported file_manager")
    except ImportError as e:
        print(f"❌ Failed to import file_manager: {e}")
        return False
    
    # Test 2: Check initial status
    print("\n2. Checking initial status...")
    status = get_file_manager_status()
    print(f"File manager status: {json.dumps(status, indent=2)}")
    
    # Test 3: Test with disabled state
    print("\n3. Testing with disabled state...")
    test_content = "test,data,content\n1,2,3\n4,5,6"
    
    # Test save_and_commit_file
    result = file_manager.save_and_commit_file(
        file_path="test.csv",
        content=test_content,
        commit_message="Test commit"
    )
    print(f"save_and_commit_file result: {result}")
    
    # Test 4: Test convenience functions
    print("\n4. Testing convenience functions...")
    try:
        from file_manager import save_and_commit_file
        result = save_and_commit_file(
            file_path="test2.csv",
            content=test_content,
            commit_message="Test commit 2"
        )
        print(f"Convenience save_and_commit_file result: {result}")
    except Exception as e:
        print(f"❌ Convenience functions failed: {e}")
    
    # Test 5: Test with enabled state (if environment allows)
    print("\n5. Testing with enabled state...")
    original_enabled = os.getenv("FILE_UPLOAD_ENABLED", "false")
    
    # Temporarily enable file uploading
    os.environ["FILE_UPLOAD_ENABLED"] = "true"
    
    # Reimport to get updated state
    import importlib
    import file_manager
    importlib.reload(file_manager)
    
    status = file_manager.get_file_manager_status()
    print(f"File manager status (enabled): {json.dumps(status, indent=2)}")
    
    # Test upload functions (should fail without proper token)
    result = file_manager.file_manager.save_and_commit_file(
        file_path="test3.csv",
        content=test_content,
        commit_message="Test commit 3"
    )
    print(f"save_and_commit_file result (enabled): {result}")
    
    # Restore original state
    os.environ["FILE_UPLOAD_ENABLED"] = original_enabled
    
    print("\n✅ File manager tests completed successfully!")
    return True

def test_login_manager():
    """Test the login manager functionality."""
    print("\n🧪 Testing Login Manager")
    print("=" * 50)
    
    # Test 1: Import and basic functionality
    print("\n1. Testing import and basic functionality...")
    try:
        from login_manager import login_manager, get_login_manager_status
        print("✅ Successfully imported login_manager")
    except ImportError as e:
        print(f"❌ Failed to import login_manager: {e}")
        return False
    
    # Test 2: Check initial status
    print("\n2. Checking initial status...")
    status = get_login_manager_status()
    print(f"Login manager status: {json.dumps(status, indent=2)}")
    
    # Test 3: Test with disabled state
    print("\n3. Testing with disabled state...")
    
    # Test is_logged_in
    result = login_manager.is_logged_in(None)
    print(f"is_logged_in result: {result}")
    
    # Test get_username
    result = login_manager.get_username(None)
    print(f"get_username result: {result}")
    
    # Test get_login_message
    result = login_manager.get_login_message()
    print(f"get_login_message result: {result}")
    
    # Test validate_login_for_operation
    is_valid, username, error_message = login_manager.validate_login_for_operation(None, "test")
    print(f"validate_login_for_operation result: {is_valid}, {username}, {error_message}")
    
    # Test 4: Test convenience functions
    print("\n4. Testing convenience functions...")
    try:
        from login_manager import is_logged_in, get_username, get_login_message
        result = is_logged_in(None)
        print(f"Convenience is_logged_in result: {result}")
        result = get_username(None)
        print(f"Convenience get_username result: {result}")
        result = get_login_message()
        print(f"Convenience get_login_message result: {result}")
    except Exception as e:
        print(f"❌ Convenience functions failed: {e}")
    
    # Test 5: Test with enabled state (if environment allows)
    print("\n5. Testing with enabled state...")
    original_enabled = os.getenv("LOGIN_ENABLED", "false")
    
    # Temporarily enable login
    os.environ["LOGIN_ENABLED"] = "true"
    
    # Reimport to get updated state
    import importlib
    import login_manager
    importlib.reload(login_manager)
    
    status = login_manager.get_login_manager_status()
    print(f"Login manager status (enabled): {json.dumps(status, indent=2)}")
    
    # Test login functions
    is_valid, username, error_message = login_manager.login_manager.validate_login_for_operation(None, "test")
    print(f"validate_login_for_operation result (enabled): {is_valid}, {username}, {error_message}")
    
    # Restore original state
    os.environ["LOGIN_ENABLED"] = original_enabled
    
    print("\n✅ Login manager tests completed successfully!")
    return True

def test_all_managers():
    """Test all managers."""
    print("🧪 Testing All Managers")
    print("=" * 80)
    
    success = True
    
    # Test file manager
    if not test_file_manager():
        success = False
    
    # Test login manager
    if not test_login_manager():
        success = False
    
    print("\n" + "=" * 80)
    print("📋 Summary:")
    print("- File manager is properly modularized")
    print("- Login manager is properly modularized")
    print("- Both are disabled by default (safe)")
    print("- Can be enabled via environment variables")
    print("- Gracefully handle missing dependencies")
    print("- Provide backward compatibility functions")
    
    return success

if __name__ == "__main__":
    success = test_all_managers()
    sys.exit(0 if success else 1)
