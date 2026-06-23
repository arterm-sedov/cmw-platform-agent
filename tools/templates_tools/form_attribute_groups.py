"""Attribute filtering and grouping helpers for generated forms."""

from __future__ import annotations

from typing import Any

SYSTEM_ATTRIBUTE_NAMES = {
    "id",
    "_creator",
    "_creationdate",
    "_lastmodifier",
    "_lastwritedate",
    "_processes",
}


def _first_present(source: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    return default


def normalize_attribute_result(attribute: dict[str, Any]) -> dict[str, Any]:
    """Normalize API/tool attribute output to a small common shape."""
    system_name = _first_present(
        attribute,
        ("Attribute system name", "attribute_system_name", "systemName", "alias"),
        "",
    )
    name = _first_present(attribute, ("Name", "name", "displayName"), system_name)
    attr_type = _first_present(attribute, ("Attribute type", "type", "attributeType"), "")
    return {
        "system_name": str(system_name or ""),
        "name": str(name or system_name or ""),
        "type": str(attr_type or ""),
        "is_system": bool(
            _first_present(attribute, ("Is system", "isSystem"), default=False)
        ),
        "archived": bool(
            _first_present(attribute, ("Archived", "isDisabled"), default=False)
        ),
        "raw": attribute,
    }


def is_system_attribute(attribute: dict[str, Any]) -> bool:
    """Return True for system or archived attributes excluded from generated forms."""
    normalized = normalize_attribute_result(attribute)
    system_name = normalized["system_name"]
    return (
        normalized["is_system"]
        or normalized["archived"]
        or system_name.startswith("_")
        or system_name.lower() in SYSTEM_ATTRIBUTE_NAMES
    )


def _group_name(attribute: dict[str, Any]) -> str:
    haystack = f"{attribute['system_name']} {attribute['name']}".lower()
    if any(term in haystack for term in ("code", "number", "name", "title", "date")):
        return "Main information"
    if any(term in haystack for term in ("account", "role", "assignee", "owner", "participant")):
        return "Participants"
    if any(term in haystack for term in ("decision", "status", "result")):
        return "Decision and status"
    if any(term in haystack for term in ("comment", "description", "note")):
        return "Comments"
    return "Other"


def infer_form_groups(attributes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group non-system attributes deterministically for form generation."""
    buckets: dict[str, list[dict[str, Any]]] = {
        "Main information": [],
        "Participants": [],
        "Decision and status": [],
        "Comments": [],
        "Other": [],
    }
    for attribute in attributes:
        normalized = normalize_attribute_result(attribute)
        if not normalized["system_name"] or is_system_attribute(attribute):
            continue
        buckets[_group_name(normalized)].append(normalized)
    return [
        {"title": title, "attributes": grouped}
        for title, grouped in buckets.items()
        if grouped
    ]
