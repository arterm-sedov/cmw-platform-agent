"""Datetime context injection for user messages.

Lean pattern from cmw-rag - injects current datetime in every user message
to avoid unnecessary tool calls while keeping tool available for explicit queries.
"""

import json

from tools.get_datetime import _get_current_datetime_dict

DEFAULT_TIMEZONE = "UTC"


def get_datetime_context(timezone: str | None = None) -> str:
    """Build datetime context string for user message injection.

    Args:
        timezone: Optional IANA timezone. Uses default if not specified.

    Returns:
        Formatted datetime context string wrapped in XML tags.
    """
    dt = _get_current_datetime_dict(timezone or DEFAULT_TIMEZONE)
    return (
        "<current_date>\n"
        "Current date/time:\n"
        f"{json.dumps(dt, ensure_ascii=False, separators=(',', ':'))}\n"
        "</current_date>\n\n"
    )


def wrap_user_message(
    user_message: str,
    timezone: str | None = None,
    include_datetime: bool | None = None,  # noqa: FBT001
) -> str:
    """Wrap user message with datetime context.

    Args:
        user_message: Original user message
        timezone: Optional timezone for datetime
        include_datetime: Whether to inject datetime context (default: True)

    Returns:
        Wrapped user message with datetime context if enabled.
    """
    if include_datetime is False:
        return user_message

    dt_context = get_datetime_context(timezone)
    return f"{dt_context}{user_message}"
