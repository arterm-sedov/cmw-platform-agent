"""
Image generation engine — provider-agnostic orchestrator.

Resolves a model slug to its :class:`ImageModelConfig`, picks the matching
adapter from :mod:`agent_ng.image_providers`, and delegates the actual call.

The engine itself knows nothing about HTTP or any specific backend. Adding
a new provider (Polza.ai, Yandex ART, GigaChat, ...) only requires:

1. Implementing :class:`agent_ng.image_providers.base.ImageProvider`.
2. Calling :func:`agent_ng.image_providers.register_provider` with a
   short name (e.g. ``"polza"``).
3. Adding entries to :data:`agent_ng.image_models.IMAGE_MODELS` with
   that provider's name in the ``provider`` field.

Neither this module nor the ``generate_ai_image`` tool need changes.

Example:
    >>> engine = ImageEngine()
    >>> result = engine.generate("A minimalist blue workflow icon")
    >>> if result.success:
    ...     Path("icon.png").write_bytes(result.image_bytes)
"""

from __future__ import annotations

import logging
import os

try:
    from .image_models import get_default_model, get_model_config
    from .image_providers import ImageGenerationResult, ImageRequest, get_provider
    from .image_providers.openrouter import OpenRouterProvider
    from .key_resolution import get_provider_api_key
    from .session_manager import get_current_session_id
except ImportError:  # pragma: no cover — fallback for script / test harnesses
    from agent_ng.image_models import (  # type: ignore[no-redef]
        get_default_model,
        get_model_config,
    )
    from agent_ng.image_providers import (  # type: ignore[no-redef]
        ImageGenerationResult,
        ImageRequest,
        get_provider,
    )
    from agent_ng.image_providers.openrouter import (  # type: ignore[no-redef]
        OpenRouterProvider,
    )
    from agent_ng.key_resolution import get_provider_api_key
    from agent_ng.session_manager import get_current_session_id
logger = logging.getLogger(__name__)


class ImageEngine:
    """Resolve a model and delegate to its provider adapter.

    Kept stateless on purpose: the adapter-selection decision happens per
    call (via the model registry) and adapter instances are cheap to
    construct. Thread-safe as a consequence.

    Constructor kwargs are forwarded for backward-compatibility with the
    original direct-HTTP implementation but only the ``openrouter`` adapter
    currently uses them. For provider-specific credentials in the future,
    use :func:`agent_ng.image_providers.register_provider` with a factory
    that closes over its own config.
    """

    # Kept for backward compatibility with callers that read these.
    api_key: str | None
    base_url: str

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        session_id: str | None = None,
    ) -> None:
        # Resolve OpenRouter key for backward-compat / per-instance override.
        # When api_key is explicitly passed (tests or custom callers) we require
        # it and bind a dedicated adapter. Otherwise, key validation is deferred
        # to the provider at generate-time so that Polza-only deployments work
        # without an OpenRouter key.
        effective_key: str | None = None
        if api_key is not None:
            effective_key = get_provider_api_key(
                provider="openrouter",
                override_key=api_key,
                session_id=session_id or get_current_session_id(),
            )
            if not effective_key:
                msg = (
                    "OPENROUTER_API_KEY is required (pass api_key= or set "
                    "the environment variable)."
                )
                raise ValueError(msg)
        else:
            # Lazy: let the chosen provider validate its own key.
            effective_key = get_provider_api_key(
                provider="openrouter",
                session_id=session_id or get_current_session_id(),
            )
            # If no OpenRouter key, still OK — Polza models don't need it.

        self.api_key = effective_key
        self.base_url = (base_url or "https://openrouter.ai/api/v1").rstrip(
            "/"
        ) + "/chat/completions"
        self.timeout = timeout

        # Per-instance adapter override: when the caller passes explicit
        # credentials we bind a dedicated OpenRouter adapter to this engine
        # rather than mutating the global registry (which would leak creds
        # across test cases and concurrent callers).
        self._instance_openrouter: OpenRouterProvider | None = None
        if (api_key is not None or base_url is not None) and effective_key:
            self._instance_openrouter = OpenRouterProvider(
                api_key=effective_key, base_url=base_url, timeout=timeout
            )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        aspect_ratio: str | None = None,
        image_size: str | None = None,
        reference_images: list[str] | None = None,
    ) -> ImageGenerationResult:
        """Generate a single image via the resolved provider.

        Provider resolution order:
        1. ``IMAGE_GEN_PROVIDER`` env var — explicit override; must be in the
           model's ``providers`` list (or the request is rejected with an error).
        2. ``config.providers[0]`` — the model's default/preferred provider.

        No automatic fallback cascade: if the selected provider fails, the
        failure is returned immediately.  The ``providers`` list is a registry
        of *available* providers for a model, not a cascade chain.
        """
        resolved_model = model or get_default_model()
        config = get_model_config(resolved_model)
        if config is None:
            return ImageGenerationResult(
                success=False,
                error=(
                    f"Unknown image model: {resolved_model!r}. "
                    "Add it to agent_ng/image_models.py first."
                ),
            )

        # Resolve which single provider to use.
        forced = os.getenv("IMAGE_GEN_PROVIDER", "").strip()
        if forced:
            if forced not in config.providers:
                return ImageGenerationResult(
                    success=False,
                    model=resolved_model,
                    error=(
                        f"Provider {forced!r} (IMAGE_GEN_PROVIDER) is not available "
                        f"for model {resolved_model!r}. "
                        f"Available providers: {config.providers}"
                    ),
                )
            provider_name = forced
        else:
            provider_name = config.providers[0]

        if self._instance_openrouter is not None and provider_name == "openrouter":
            provider = self._instance_openrouter
        else:
            provider = get_provider(provider_name)

        if provider is None:
            return ImageGenerationResult(
                success=False,
                model=resolved_model,
                error=(
                    f"No adapter registered for provider {provider_name!r}. "
                    "See agent_ng/image_providers for how to add one."
                ),
            )

        # Clip reference images to what this model supports.
        clipped_refs: list[str] | None = None
        if reference_images:
            max_imgs = config.max_reference_images
            if max_imgs > 0:
                clipped_refs = reference_images[:max_imgs]
                if len(clipped_refs) < len(reference_images):
                    logger.warning(
                        "Model %s supports at most %d reference image(s); "
                        "%d provided, clipped to %d.",
                        resolved_model,
                        max_imgs,
                        len(reference_images),
                        len(clipped_refs),
                    )
            else:
                logger.warning(
                    "Model %s does not support reference images; "
                    "ignoring %d provided.",
                    resolved_model,
                    len(reference_images),
                )

        return provider.generate(
            ImageRequest(
                prompt=prompt,
                config=config,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                reference_images=clipped_refs,
            )
        )


__all__ = ["ImageEngine", "ImageGenerationResult"]
