#!/usr/bin/env python3
"""
Diagnostic script to explore available applications, templates, and datasets.
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

from tools.applications_tools.tool_list_applications import list_applications
from tools.applications_tools.tool_list_templates import list_templates
from tools.templates_tools.tools_dataset import list_datasets


def explore_platform():
    """Explore available applications, templates, and datasets."""

    logger.info("=" * 80)
    logger.info("STEP 1: List all applications")
    logger.info("=" * 80)

    try:
        apps_result = list_applications.invoke({})
        logger.info(f"Applications result: {json.dumps(apps_result, indent=2, default=str)}")

        if apps_result.get('success') and apps_result.get('data'):
            apps = apps_result['data']
            logger.info(f"Found {len(apps)} applications")

            for app in apps[:5]:  # Show first 5
                app_name = app.get('name', 'Unknown')
                app_alias = app.get('alias', 'Unknown')
                logger.info(f"  - {app_name} (alias: {app_alias})")

                # Try to list templates for this app
                logger.info(f"\n  STEP 2: List templates for {app_alias}")
                logger.info("  " + "=" * 76)

                try:
                    templates_result = list_templates.invoke({
                        "application_system_name": app_alias
                    })

                    if templates_result.get('success') and templates_result.get('data'):
                        templates = templates_result['data']
                        logger.info(f"  Found {len(templates)} templates")

                        for tpl in templates[:3]:  # Show first 3 templates
                            tpl_name = tpl.get('name', 'Unknown')
                            tpl_alias = tpl.get('alias', 'Unknown')
                            logger.info(f"    - {tpl_name} (alias: {tpl_alias})")

                            # Try to list datasets for this template
                            logger.info(f"\n    STEP 3: List datasets for {tpl_alias}")
                            logger.info("    " + "=" * 72)

                            try:
                                datasets_result = list_datasets.invoke({
                                    "application_system_name": app_alias,
                                    "template_system_name": tpl_alias
                                })

                                if datasets_result.get('success') and datasets_result.get('data'):
                                    datasets = datasets_result['data']
                                    logger.info(f"    Found {len(datasets)} datasets")

                                    for ds in datasets[:3]:  # Show first 3 datasets
                                        ds_name = ds.get('name', 'Unknown')
                                        ds_alias = ds.get('alias', 'Unknown')
                                        logger.info(f"      - {ds_name} (alias: {ds_alias})")
                                else:
                                    logger.info(f"    No datasets or error: {datasets_result.get('error', 'Unknown')}")
                            except Exception as e:
                                logger.error(f"    Error listing datasets: {e}")
                    else:
                        logger.info(f"  No templates or error: {templates_result.get('error', 'Unknown')}")
                except Exception as e:
                    logger.error(f"  Error listing templates: {e}")
        else:
            logger.error(f"Failed to list applications: {apps_result.get('error', 'Unknown')}")
    except Exception as e:
        logger.error(f"Error listing applications: {e}", exc_info=True)


if __name__ == "__main__":
    explore_platform()
