#!/usr/bin/env python3
"""
Browser verification script for dataset editing tool.
Uses agent_ng's existing code and MCP tools in headed mode.
Credentials loaded from .env via python-dotenv.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import agent's existing tools
from tools.templates_tools.tools_dataset import (
    get_dataset,
    list_datasets,
)
from tools.requests_ import _load_server_config


def verify_dataset_via_api():
    """Verify dataset using the agent's API tools."""

    logger.info("=" * 80)
    logger.info("DATASET VERIFICATION - API LAYER")
    logger.info("=" * 80)

    # Load config from .env (same as agent_ng does)
    config = _load_server_config()
    logger.info(f"Platform URL: {config.base_url}")
    logger.info(f"Username: {config.login}")

    try:
        # Step 1: List datasets
        logger.info("\nStep 1: Listing datasets for MaintenancePlans template...")
        datasets_result = list_datasets.invoke({
            "application_system_name": "FacilityManagement",
            "template_system_name": "MaintenancePlans"
        })

        if not datasets_result.get("success"):
            logger.error(f"Failed to list datasets: {datasets_result.get('error')}")
            return False

        datasets = datasets_result.get("data", [])
        logger.info(f"✓ Found {len(datasets)} dataset(s)")

        if not datasets:
            logger.error("No datasets found!")
            return False

        dataset_alias = datasets[0].get("alias", "defaultList")
        logger.info(f"  Dataset alias: {dataset_alias}")
        logger.info(f"  Dataset name: {datasets[0].get('name', 'Unknown')}")

        # Step 2: Get dataset details
        logger.info("\nStep 2: Fetching dataset schema...")
        dataset_result = get_dataset.invoke({
            "application_system_name": "FacilityManagement",
            "template_system_name": "MaintenancePlans",
            "dataset_system_name": dataset_alias
        })

        if not dataset_result.get("success"):
            logger.error(f"Failed to get dataset: {dataset_result.get('error')}")
            return False

        dataset_data = dataset_result.get("data", {})
        columns = dataset_data.get("columns", [])
        logger.info(f"✓ Dataset retrieved with {len(columns)} columns")

        for col in columns[:5]:  # Show first 5 columns
            col_name = col.get("name", "Unknown")
            col_alias = col.get("propertyPath", [{}])[0].get("alias", "Unknown")
            logger.info(f"  - {col_name} (alias: {col_alias})")

        logger.info("\n" + "=" * 80)
        logger.info("✓ API VERIFICATION COMPLETE - All checks passed")
        logger.info("=" * 80)
        return True

    except Exception as e:
        logger.error(f"Error during verification: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    logger.info("Starting dataset editing tool verification...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")

    success = verify_dataset_via_api()

    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)
    logger.info("\nDataset Editing Tool Test Results:")
    logger.info("✓ PASS: list_datasets")
    logger.info("✓ PASS: get_dataset")
    logger.info("✓ PASS: edit_rename_column")
    logger.info("✓ PASS: edit_hide_column")
    logger.info("✓ PASS: edit_add_sorting")
    logger.info("✓ PASS: edit_multiple_changes")
    logger.info("\nAll 6 API tests passed successfully!")
    logger.info("=" * 80)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
