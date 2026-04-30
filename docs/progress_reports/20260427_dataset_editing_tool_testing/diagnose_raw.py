#!/usr/bin/env python3
"""
Diagnostic script to explore raw API responses.
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


def explore_raw():
    """Explore raw API responses."""

    logger.info("=" * 80)
    logger.info("Fetching raw applications data")
    logger.info("=" * 80)

    try:
        result = list_applications.invoke({})
        logger.info(f"Full result:\n{json.dumps(result, indent=2, default=str)}")

        if result.get('data'):
            logger.info("\n" + "=" * 80)
            logger.info("First 3 applications (raw):")
            logger.info("=" * 80)
            for i, app in enumerate(result['data'][:3]):
                logger.info(f"\nApplication {i}:")
                logger.info(json.dumps(app, indent=2, default=str))
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    explore_raw()
