"""
Provider registry.

Maps a short provider key (``"openrouter"``, ``"polza"``, ``"yandex"``,
``"gigachat"``, ...) to an adapter factory. The engine uses this to pick
the right adapter for a given :class:`ImageModelConfig`.

A built-in ``"openrouter"`` entry is registered at import time. Additional
providers plug in via :func:`register_provider`, typically from a side-car
module (e.g. ``image_providers/polza.py``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .openrouter import OpenRouterProvider

if TYPE_CHECKING:
    from collections.abc import Callable

    from .base import ImageProvider

logger = logging.getLogger(__name__)

# Factory = zero-arg callable returning a fresh provider instance. Using a
# factory (instead of storing an instance) keeps credentials loaded lazily
# and tests easy to isolate via monkeypatching.
_FACTORIES: dict[str, Callable[[], ImageProvider]] = {}


def register_provider(name: str, factory: Callable[[], ImageProvider]) -> None:
    """Register a provider factory under ``name``.

    Overwrites any existing registration — useful for tests and for
    swapping in an alternative credential source.
    """
    if not isinstance(name, str) or not name:
        msg = "provider name must be a non-empty string"
        raise ValueError(msg)
    if name in _FACTORIES:
        logger.debug("Replacing image provider factory for %r", name)
    _FACTORIES[name] = factory


def get_provider(name: str) -> ImageProvider | None:
    """Instantiate and return the provider registered under ``name``.

    Returns ``None`` when no adapter is registered for that name, so the
    engine can surface a clear error without crashing.
    """
    factory = _FACTORIES.get(name)
    if factory is None:
        return None
    try:
        return factory()
    except Exception:
        logger.exception("Image provider %r failed to initialize", name)
        return None


def list_providers() -> list[str]:
    """Return the list of registered provider keys (for diagnostics)."""
    return sorted(_FACTORIES.keys())


# --- Built-in registrations ------------------------------------------------

register_provider("openrouter", OpenRouterProvider)


__all__ = [
    "get_provider",
    "list_providers",
    "register_provider",
]
