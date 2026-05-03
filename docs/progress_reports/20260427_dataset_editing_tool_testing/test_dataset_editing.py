#!/usr/bin/env python3
"""
Test script for dataset editing tool in CMW Platform agent.
Tests the edit_or_create_dataset tool with MaintenancePlans dataset in FacilityManagement app.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import tools
from tools.templates_tools.tools_dataset import (
    edit_or_create_dataset,
    get_dataset,
    list_datasets,
)


def test_list_datasets():
    """Test listing datasets for FacilityManagement/MaintenancePlans."""
    logger.info("=" * 80)
    logger.info("TEST 1: List datasets for FacilityManagement template")
    logger.info("=" * 80)

    try:
        result = list_datasets.invoke({
            "application_system_name": "FacilityManagement",
            "template_system_name": "MaintenancePlans"
        })
        logger.info(f"Result: {json.dumps(result, indent=2)}")
        return result
    except Exception as e:
        logger.error(f"Error listing datasets: {e}", exc_info=True)
        return None


def test_get_dataset():
    """Test fetching the MaintenancePlans dataset."""
    logger.info("=" * 80)
    logger.info("TEST 2: Get MaintenancePlans dataset (defaultList)")
    logger.info("=" * 80)

    try:
        result = get_dataset.invoke({
            "application_system_name": "FacilityManagement",
            "template_system_name": "MaintenancePlans",
            "dataset_system_name": "defaultList"
        })
        logger.info(f"Result: {json.dumps(result, indent=2, default=str)}")
        return result
    except Exception as e:
        logger.error(f"Error getting dataset: {e}", exc_info=True)
        return None


def test_edit_dataset_rename_column():
    """Test editing dataset - rename a column."""
    logger.info("=" * 80)
    logger.info("TEST 3: Edit dataset - rename column")
    logger.info("=" * 80)

    try:
        result = edit_or_create_dataset.invoke({
            "operation": "edit",
            "application_system_name": "FacilityManagement",
            "template_system_name": "MaintenancePlans",
            "dataset_system_name": "defaultList",
            "columns": {
                "Title": {"name": "Plan Title (Updated)"}
            }
        })
        logger.info(f"Result: {json.dumps(result, indent=2)}")
        return result
    except Exception as e:
        logger.error(f"Error editing dataset: {e}", exc_info=True)
        return None


def test_edit_dataset_hide_column():
    """Test editing dataset - hide a column."""
    logger.info("=" * 80)
    logger.info("TEST 4: Edit dataset - hide column")
    logger.info("=" * 80)

    try:
        result = edit_or_create_dataset.invoke({
            "operation": "edit",
            "application_system_name": "FacilityManagement",
            "template_system_name": "MaintenancePlans",
            "dataset_system_name": "defaultList",
            "columns": {
                "isDisabled": {"isHidden": True}
            }
        })
        logger.info(f"Result: {json.dumps(result, indent=2)}")
        return result
    except Exception as e:
        logger.error(f"Error editing dataset: {e}", exc_info=True)
        return None


def test_edit_dataset_add_sorting():
    """Test editing dataset - add sorting."""
    logger.info("=" * 80)
    logger.info("TEST 5: Edit dataset - add sorting")
    logger.info("=" * 80)

    try:
        result = edit_or_create_dataset.invoke({
            "operation": "edit",
            "application_system_name": "FacilityManagement",
            "template_system_name": "MaintenancePlans",
            "dataset_system_name": "defaultList",
            "sorting": [
                {
                    "propertyPath": [
                        {
                            "type": "Attribute",
                            "owner": "MaintenancePlans",
                            "alias": "Title"
                        }
                    ],
                    "direction": "Asc",
                    "nullValuesOnTop": False
                }
            ]
        })
        logger.info(f"Result: {json.dumps(result, indent=2)}")
        return result
    except Exception as e:
        logger.error(f"Error editing dataset: {e}", exc_info=True)
        return None


def test_edit_dataset_multiple_changes():
    """Test editing dataset - multiple changes at once."""
    logger.info("=" * 80)
    logger.info("TEST 6: Edit dataset - multiple changes")
    logger.info("=" * 80)

    try:
        result = edit_or_create_dataset.invoke({
            "operation": "edit",
            "application_system_name": "FacilityManagement",
            "template_system_name": "MaintenancePlans",
            "dataset_system_name": "defaultList",
            "name": "Maintenance Plans (Updated)",
            "is_default": True,
            "columns": {
                "Title": {"name": "Plan Title"},
                "isDisabled": {"isHidden": False}
            }
        })
        logger.info(f"Result: {json.dumps(result, indent=2)}")
        return result
    except Exception as e:
        logger.error(f"Error editing dataset: {e}", exc_info=True)
        return None


def main():
    """Run all tests."""
    logger.info("Starting dataset editing tool tests...")
    logger.info(f"Python version: {sys.version}")

    results = {}

    # Test 1: List datasets
    results['list_datasets'] = test_list_datasets()

    # Test 2: Get dataset
    results['get_dataset'] = test_get_dataset()

    # Test 3: Edit - rename column
    results['edit_rename_column'] = test_edit_dataset_rename_column()

    # Test 4: Edit - hide column
    results['edit_hide_column'] = test_edit_dataset_hide_column()

    # Test 5: Edit - add sorting
    results['edit_add_sorting'] = test_edit_dataset_add_sorting()

    # Test 6: Edit - multiple changes
    results['edit_multiple_changes'] = test_edit_dataset_multiple_changes()

    # Summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    for test_name, result in results.items():
        if result:
            success = result.get('success', False)
            status = "✓ PASS" if success else "✗ FAIL"
            logger.info(f"{status}: {test_name}")
            if not success:
                logger.info(f"  Error: {result.get('error', 'Unknown error')}")
        else:
            logger.info(f"✗ FAIL: {test_name} (Exception occurred)")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
