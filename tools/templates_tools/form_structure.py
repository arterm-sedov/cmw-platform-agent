"""Pure helpers for inspecting CMW form structures."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


def iter_form_nodes(obj: Any) -> Iterator[dict[str, Any]]:
    """Yield every dictionary node in a form tree."""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from iter_form_nodes(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_form_nodes(item)


def list_field_components(form: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all form nodes whose type is FieldComponent."""
    return [node for node in iter_form_nodes(form) if node.get("type") == "FieldComponent"]


def count_field_components(form: dict[str, Any] | None) -> int:
    """Return the number of visible field components."""
    if not isinstance(form, dict):
        return 0
    return len(list_field_components(form))


def _aliases_from_property_path(property_path: Any) -> Iterator[str]:
    if isinstance(property_path, dict):
        alias = property_path.get("alias")
        if isinstance(alias, str) and alias:
            yield alias
    elif isinstance(property_path, list):
        for item in property_path:
            yield from _aliases_from_property_path(item)


def list_referenced_attribute_aliases(form: dict[str, Any]) -> set[str]:
    """Extract attribute aliases from FieldComponent.propertyPath values."""
    aliases: set[str] = set()
    for field in list_field_components(form):
        aliases.update(_aliases_from_property_path(field.get("propertyPath")))
    return aliases


def form_has_visible_fields(form: dict[str, Any] | None) -> bool:
    """Return True when a form contains at least one FieldComponent."""
    return count_field_components(form) > 0


def assert_no_stale_tokens(form: dict[str, Any], stale_tokens: list[str]) -> None:
    """Raise ValueError if any source token remains after copying."""
    serialized = json.dumps(form, ensure_ascii=False)
    remaining = [token for token in stale_tokens if token and token in serialized]
    if remaining:
        msg = "Copied form still contains stale source tokens: " + ", ".join(remaining)
        raise ValueError(msg)


def assert_references_known_attributes(
    form: dict[str, Any],
    known_attribute_aliases: set[str],
    *,
    allow_system_attributes: bool = False,
) -> None:
    """Validate that FieldComponent.propertyPath aliases exist in template attrs."""
    aliases = list_referenced_attribute_aliases(form)
    if allow_system_attributes:
        aliases = {alias for alias in aliases if not alias.startswith("_")}
    missing = sorted(aliases - known_attribute_aliases)
    if missing:
        msg = "Form references unknown attributes: " + ", ".join(missing)
        raise ValueError(msg)
