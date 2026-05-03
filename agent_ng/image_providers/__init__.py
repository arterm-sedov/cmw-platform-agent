"""
Image-generation provider adapters.

Pluggable backends for text-to-image generation. Each adapter implements
:class:`ImageProvider` and is registered under a short key (``"openrouter"``,
``"polza"``, ``"yandex"``, ...) that matches ``ImageModelConfig.provider``.

The engine selects an adapter by looking up the model's ``provider`` field
in the registry. Adding a new backend is a three-step change:

1. Implement a subclass of :class:`ImageProvider` in this package.
2. Register it via :func:`register_provider` (or let the built-in
   ``openrouter`` adapter remain the default).
3. Add entries to :data:`agent_ng.image_models.IMAGE_MODELS` with the
   matching ``provider`` value.

No changes are required in :class:`agent_ng.image_engine.ImageEngine` or in
the ``generate_ai_image`` tool.
"""

from __future__ import annotations

from .base import ImageGenerationResult, ImageProvider, ImageRequest
from .openrouter import OpenRouterProvider
from .polza import PolzaProvider
from .registry import get_provider, list_providers, register_provider

__all__ = [
    "ImageGenerationResult",
    "ImageProvider",
    "ImageRequest",
    "OpenRouterProvider",
    "PolzaProvider",
    "get_provider",
    "list_providers",
    "register_provider",
]
