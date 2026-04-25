"""
Shared data types and abstract base for image-generation provider adapters.

``ImageRequest`` is the provider-agnostic input. ``ImageGenerationResult``
is the provider-agnostic output (binary bytes + normalized usage metadata).
``ImageProvider`` is the contract every adapter must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_ng.image_models import ImageModelConfig


@dataclass(frozen=True)
class ImageRequest:
    """Provider-agnostic image-generation request.

    Attributes:
        prompt: Text description of the desired image.
        config: Metadata for the target model (carries modalities,
            ``supports_image_config`` flag, etc.).
        aspect_ratio: Optional ``"W:H"`` ratio. Providers that don't
            support aspect control may ignore this.
        image_size: Optional resolution tier (e.g. ``"1K"``, ``"2K"``,
            ``"4K"``). Providers that don't support size control may
            ignore this.
    """

    prompt: str
    config: ImageModelConfig
    aspect_ratio: str | None = None
    image_size: str | None = None


@dataclass
class ImageGenerationResult:
    """Provider-agnostic outcome of a single image-generation request.

    On success, ``image_bytes`` holds the decoded binary payload and
    ``mime_type`` is set. On failure, ``error`` carries a human-readable
    message and ``image_bytes`` is ``None``.

    Cost and token counts are normalized to the fields below; providers
    that don't report some values leave them ``None``.
    """

    success: bool
    image_bytes: bytes | None = None
    mime_type: str | None = None
    model: str | None = None
    cost: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    generation_id: str | None = None
    error: str | None = None


class ImageProvider(ABC):
    """Abstract adapter for a text-to-image backend.

    Implementations should be stateless beyond their configuration
    (credentials, base URL, timeouts) and safe to instantiate per-call.
    """

    #: Registry key for this adapter; must match ``ImageModelConfig.provider``.
    name: str = ""

    @abstractmethod
    def generate(self, request: ImageRequest) -> ImageGenerationResult:
        """Generate a single image from a text prompt.

        Implementations must not raise for expected failures (HTTP
        errors, bad responses, unsupported options). They should return
        an ``ImageGenerationResult(success=False, error=...)`` instead.
        """


__all__ = [
    "ImageGenerationResult",
    "ImageProvider",
    "ImageRequest",
]
