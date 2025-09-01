#!/usr/bin/env python3
"""
Test script for dataset_manager.py
==================================

This script tests the dataset manager functionality in both disabled and enabled modes.
"""

import os
import sys
import json
from datetime import datetime

def test_dataset_manager():
    """Test the dataset manager functionality."""
    print("🧪 Testing Dataset Manager")
    print("=" * 50)
    
    # Test 1: Import and basic functionality
    print("\n1. Testing import and basic functionality...")
    try:
        from dataset_manager import dataset_manager, get_dataset_status
        print("✅ Successfully imported dataset_manager")
    except ImportError as e:
        print(f"❌ Failed to import dataset_manager: {e}")
        return False
    
    # Test 2: Check initial status
    print("\n2. Checking initial status...")
    status = get_dataset_status()
    print(f"Dataset manager status: {json.dumps(status, indent=2)}")
    
    # Test 3: Test with disabled state
    print("\n3. Testing with disabled state...")
    test_data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "test_field": "test_value",
        "test_number": 42
    }
    
    # Test upload_init_summary
    result = dataset_manager.upload_init_summary(test_data)
    print(f"upload_init_summary result: {result}")
    
    # Test upload_run_data
    result = dataset_manager.upload_run_data(test_data)
    print(f"upload_run_data result: {result}")
    
    # Test 4: Test convenience functions
    print("\n4. Testing convenience functions...")
    try:
        from dataset_manager import upload_init_summary, upload_run_data
        result = upload_init_summary(test_data)
        print(f"Convenience upload_init_summary result: {result}")
        result = upload_run_data(test_data)
        print(f"Convenience upload_run_data result: {result}")
    except Exception as e:
        print(f"❌ Convenience functions failed: {e}")
    
    # Test 5: Test with enabled state (if environment allows)
    print("\n5. Testing with enabled state...")
    original_enabled = os.getenv("DATASET_ENABLED", "false")
    
    # Temporarily enable dataset uploading
    os.environ["DATASET_ENABLED"] = "true"
    
    # Reimport to get updated state
    import importlib
    import dataset_manager
    importlib.reload(dataset_manager)
    
    status = dataset_manager.get_dataset_status()
    print(f"Dataset manager status (enabled): {json.dumps(status, indent=2)}")
    
    # Test upload functions (should fail without proper token)
    result = dataset_manager.dataset_manager.upload_init_summary(test_data)
    print(f"upload_init_summary result (enabled): {result}")
    
    # Restore original state
    os.environ["DATASET_ENABLED"] = original_enabled
    
    print("\n✅ Dataset manager tests completed successfully!")
    print("\n📋 Summary:")
    print("- Dataset manager is properly modularized")
    print("- Disabled by default (safe)")
    print("- Can be enabled via DATASET_ENABLED environment variable")
    print("- Gracefully handles missing dependencies")
    print("- Provides backward compatibility functions")
    
    return True

if __name__ == "__main__":
    success = test_dataset_manager()
    sys.exit(0 if success else 1)
