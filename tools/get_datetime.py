"""Datetime utilities for getting current date/time."""

from datetime import datetime
import json
import logging
import os
from zoneinfo import ZoneInfo

from langchain.tools import tool
from pydantic import BaseModel, Field


def _get_default_timezone() -> str:
    """Get default timezone from environment or fallback to UTC."""
    return os.environ.get("DEFAULT_TIMEZONE", "UTC")


logger = logging.getLogger(__name__)

DEFAULT_TIMEZONE = _get_default_timezone()


def _get_current_datetime_dict(timezone: str | None = None) -> dict:
    """Get current datetime as structured dictionary.

    This helper function is used by both the get_current_datetime tool and the system prompt
    to ensure consistent datetime data across the application.

    Args:
        timezone: Optional IANA timezone name. If None, uses default timezone.

    Returns:
        Dictionary with structured datetime information.
    """
    tz_str = timezone or DEFAULT_TIMEZONE

    try:
        tz = ZoneInfo(tz_str)
    except Exception:
        logger.warning(
            "Invalid timezone %s, falling back to %s", tz_str, DEFAULT_TIMEZONE
        )
        tz = ZoneInfo(DEFAULT_TIMEZONE)
        tz_str = DEFAULT_TIMEZONE

    now = datetime.now(tz)

    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    weekday_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    return {
        "iso_format": now.isoformat(),
        "timezone": tz_str,
        "timestamp": int(now.timestamp()),
        "month_name": month_names[now.month - 1],
        "weekday": weekday_names[now.weekday()],
    }


class GetDateTimeSchema(BaseModel):
    """Schema for getting current date and time information."""

    timezone: str | None = Field(
        default=None,
        description="Optional IANA timezone name (e.g., 'Europe/Moscow', 'UTC', 'America/New_York'). "
        "If not specified, uses the default timezone.",
    )


@tool("get_current_datetime", args_schema=GetDateTimeSchema)
def get_current_datetime(
    timezone: str | None = None,
) -> str:
    """Get the current date and time information.

    **Use when:**
    - User asks about current date, time, day of week, or month
    - User asks about time periods (e.g., "December release", "this month's updates")
    - Need to filter or search information by periods, date ranges
    - User asks "when" questions requiring current time context
    - Need to determine temporal context

    **Examples:**
    - "что в декабрьском выпуске" → Get current date to determine if December is past/present/future
    - "Что изменилось в этом месяце?" → Get current month, then search for recent updates
    - "Когда был последний релиз?" → Get current date to compare with release dates

    **Timezone handling:**
    - By default, returns time in UTC
    - Specify a different timezone if the user asks for time in a specific location
    - Common timezones: UTC, Europe/Moscow, America/New_York, Asia/Dubai

    Returns:
        JSON with date/time information:
        {
          "iso_format": "2024-12-09T15:30:45+03:00",
          "timezone": "Europe/Moscow",
          "timestamp": 1702121445,
          "month_name": "December",
          "weekday": "Monday"
        }
    """
    try:
        result = _get_current_datetime_dict(timezone=timezone)
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    except Exception as exc:
        logger.exception("Error getting current datetime: %s", exc)
        return json.dumps(
            {"error": f"Failed to get current datetime: {exc!s}", "datetime": None},
            ensure_ascii=False,
        )

