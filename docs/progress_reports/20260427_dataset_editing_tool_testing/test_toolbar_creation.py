#!/usr/bin/env python3
"""
Test script to create a new toolbar with buttons in MaintenancePlans template.
Verifies creation via agentic tools, direct API, and browser automation.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import agent's tools
from tools.templates_tools.tools_toolbar import (
    edit_or_create_toolbar,
    get_toolbar,
    list_toolbars,
)
from tools.templates_tools.tools_button import (
    edit_or_create_button,
    get_button,
)


def test_create_toolbar():
    """Create a new toolbar in MaintenancePlans template."""
    logger.info("=" * 80)
    logger.info("STEP 1: Create new toolbar 'TestToolbar_20260427'")
    logger.info("=" * 80)

    try:
        result = edit_or_create_toolbar.invoke({
            "operation": "create",
            "application_system_name": "FacilityManagement",
            "template_system_name": "MaintenancePlans",
            "toolbar_system_name": "TestToolbar_20260427",
            "name": "Test Toolbar (Created 2026-04-27)",
            "is_default_for_lists": False
        })

        logger.info(f"Result: {json.dumps(result, indent=2)}")

        if result.get("success"):
            logger.info("✅ Toolbar created successfully!")
            return True
        else:
            logger.error(f"❌ Failed to create toolbar: {result.get('error')}")
            return False

    except Exception as e:
        logger.error(f"❌ Exception creating toolbar: {e}", exc_info=True)
        return False


def test_create_buttons():
    """Create buttons in the new toolbar."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Create buttons in TestToolbar_20260427")
    logger.info("=" * 80)

    buttons = [
        {
            "system_name": "TestButton1_20260427",
            "name": "Test Button 1",
            "action": "OpenForm"
        },
        {
            "system_name": "TestButton2_20260427",
            "name": "Test Button 2",
            "action": "OpenForm"
        },
        {
            "system_name": "TestButton3_20260427",
            "name": "Test Button 3",
            "action": "OpenForm"
        }
    ]

    created_buttons = []

    for button in buttons:
        try:
            logger.info(f"\nCreating button: {button['name']}")

            result = edit_or_create_button.invoke({
                "operation": "create",
                "application_system_name": "FacilityManagement",
                "template_system_name": "MaintenancePlans",
                "button_system_name": button["system_name"],
                "name": button["name"],
                "action": button["action"]
            })

            if result.get("success"):
                logger.info(f"✅ Button '{button['name']}' created successfully!")
                created_buttons.append(button["system_name"])
            else:
                logger.error(f"❌ Failed to create button: {result.get('error')}")

        except Exception as e:
            logger.error(f"❌ Exception creating button: {e}", exc_info=True)

    return created_buttons


def test_verify_via_tools():
    """Verify toolbar and buttons via agentic tools."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Verify via agentic tools")
    logger.info("=" * 80)

    # List all toolbars
    logger.info("\n3.1: List all toolbars")
    try:
        result = list_toolbars.invoke({
            "application_system_name": "FacilityManagement",
            "template_system_name": "MaintenancePlans"
        })

        if result.get("success"):
            toolbars = result.get("data", [])
            logger.info(f"Found {len(toolbars)} toolbars")

            # Check if our toolbar is in the list
            test_toolbar = next((t for t in toolbars if t.get("alias") == "TestToolbar_20260427"), None)
            if test_toolbar:
                logger.info(f"✅ TestToolbar_20260427 found in list!")
                logger.info(f"   Name: {test_toolbar.get('name')}")
            else:
                logger.error("❌ TestToolbar_20260427 NOT found in list!")
        else:
            logger.error(f"❌ Failed to list toolbars: {result.get('error')}")

    except Exception as e:
        logger.error(f"❌ Exception listing toolbars: {e}", exc_info=True)

    # Get specific toolbar
    logger.info("\n3.2: Get TestToolbar_20260427")
    try:
        result = get_toolbar.invoke({
            "application_system_name": "FacilityManagement",
            "template_system_name": "MaintenancePlans",
            "toolbar_system_name": "TestToolbar_20260427"
        })

        if result.get("success"):
            logger.info("✅ Toolbar retrieved successfully!")
            toolbar_data = result.get("data", {})
            logger.info(f"   Name: {toolbar_data.get('name')}")
            logger.info(f"   Alias: {toolbar_data.get('alias')}")
        else:
            logger.error(f"❌ Failed to get toolbar: {result.get('error')}")

    except Exception as e:
        logger.error(f"❌ Exception getting toolbar: {e}", exc_info=True)


def test_verify_via_api():
    """Verify toolbar via direct API calls."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Verify via direct API calls")
    logger.info("=" * 80)

    from tools.requests_ import _get_request, _load_server_config

    config = _load_server_config()

    # Direct API call to get toolbar
    endpoint = "webapi/Toolbar/FacilityManagement/Toolbar@MaintenancePlans.TestToolbar_20260427"

    try:
        logger.info(f"API GET: {endpoint}")
        response = _get_request(endpoint)

        if response.success:
            logger.info("✅ Direct API call successful!")
            logger.info(f"   Status: {response.status_code}")
            logger.info(f"   Response: {json.dumps(response.raw_response, indent=2)[:500]}...")
        else:
            logger.error(f"❌ Direct API call failed: {response.error}")

    except Exception as e:
        logger.error(f"❌ Exception in direct API call: {e}", exc_info=True)


def main():
    """Run all tests."""
    logger.info("Starting toolbar creation and verification test...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Target: FacilityManagement / MaintenancePlans / TestToolbar_20260427")

    # Step 1: Create toolbar
    if not test_create_toolbar():
        logger.error("Failed to create toolbar. Stopping.")
        return 1

    # Step 2: Create buttons
    created_buttons = test_create_buttons()
    logger.info(f"\nCreated {len(created_buttons)} buttons: {created_buttons}")

    # Step 3: Verify via tools
    test_verify_via_tools()

    # Step 4: Verify via direct API
    test_verify_via_api()

    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)
    logger.info("\nNext: Verify visually via agent-browser MCP")
    logger.info("Run: agent-browser to open platform and check toolbar")

    return 0


if __name__ == "__main__":
    sys.exit(main())
