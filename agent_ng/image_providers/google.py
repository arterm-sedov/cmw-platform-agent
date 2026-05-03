"""
Google Gemini native image-generation adapter.

Implements :class:`agent_ng.image_providers.base.ImageProvider` against the
Google Gen AI SDK (``google-genai``).

Why not OpenRouter for Gemini?  OpenRouter and Polza both proxy to Google's
backend, which helps users in geo-restricted regions.  This adapter connects
directly and is the right choice when the server has unrestricted access to
Google's API (e.g. hosting outside Russia / CIS).

API key: ``GEMINI_KEY`` env var (same variable used by the LLM adapter).
SDK: ``google-genai`` (``pip install google-genai``).

Cost reporting: the Google API returns token counts in ``usage_metadata``
but not a dollar amount.  The adapter falls back to
``config.price_per_generation_usd`` for a static cost estimate when that
field is set (populated in the image model registry from Polza's published
per-request prices).  Without it, ``cost`` is ``None`` in the result.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

from dotenv import load_dotenv

from .base import ImageGenerationResult, ImageProvider, ImageRequest

try:
    from ..key_resolution import get_provider_api_key
except ImportError:
    from agent_ng.key_resolution import get_provider_api_key  # type: ignore[no-redef]

load_dotenv()
logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120.0


class GoogleGeminiProvider(ImageProvider):
    """Image provider backed by the Google Gen AI SDK (direct Gemini API)."""

    name = "google"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        key = api_key or get_provider_api_key("gemini") or os.getenv("GEMINI_KEY")
        if not key:
            msg = (
                "GEMINI_KEY is required for the google image provider "
                "(pass api_key= or set the GEMINI_KEY environment variable)."
            )
            raise ValueError(msg)
        self.api_key: str = key
        self.timeout: float = timeout

        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:
            msg = (
                "google-genai package is required for the google image provider. "
                "Install it with: pip install google-genai"
            )
            raise ImportError(msg) from exc

        self._client = genai.Client(api_key=self.api_key)
        self._types = genai_types

    # ------------------------------------------------------------------ #
    # ImageProvider                                                      #
    # ------------------------------------------------------------------ #

    def generate(self, request: ImageRequest) -> ImageGenerationResult:
        model_id = request.config.provider_model_ids.get("google", request.config.name)
        contents = self._build_contents(request)
        cfg = self._build_config(request)
        logger.debug(
            "GoogleGeminiProvider generate model=%s image_config=%s",
            model_id,
            {
                "aspect_ratio": request.aspect_ratio,
                "image_size": request.image_size,
            },
        )
        try:
            response = self._client.models.generate_content(
                model=model_id,
                contents=contents,
                config=cfg,
            )
        except Exception as exc:
            return ImageGenerationResult(
                success=False,
                model=request.config.name,
                error=f"Google Gemini API error: {exc}",
            )
        return self._parse_response(response, request.config)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _build_contents(self, request: ImageRequest) -> list[Any]:
        """Build the ``contents`` list: text prompt + optional reference images."""
        parts: list[Any] = [request.prompt]
        parts.extend(self._img_to_part(img) for img in request.reference_images or [])
        return parts

    def _img_to_part(self, img: str) -> Any:
        """Convert a URL / data URI / raw base64 string to a genai Part."""
        types = self._types
        if img.startswith(("http://", "https://")):
            return types.Part.from_uri(uri=img, mime_type="image/jpeg")
        if img.startswith("data:"):
            try:
                header, payload = img.split(",", 1)
                mime = header.split(";")[0][len("data:"):]
                return types.Part.from_bytes(
                    data=base64.b64decode(payload),
                    mime_type=mime or "image/jpeg",
                )
            except Exception as exc:
                logger.warning("Could not parse data URI reference image: %s", exc)
                return img  # fallback: pass raw, SDK may handle it
        # Raw base64
        try:
            return types.Part.from_bytes(
                data=base64.b64decode(img),
                mime_type="image/jpeg",
            )
        except Exception as exc:
            logger.warning("Could not decode base64 reference image: %s", exc)
            return img

    def _build_config(self, request: ImageRequest) -> Any:
        """Build GenerateContentConfig with modalities and optional image_config."""
        types = self._types
        image_cfg = None
        if request.config.supports_image_config:
            kwargs: dict[str, str] = {}
            if request.aspect_ratio:
                kwargs["aspect_ratio"] = request.aspect_ratio
            if request.image_size:
                kwargs["image_size"] = request.image_size
            image_cfg = types.ImageConfig(**kwargs) if kwargs else None

        return types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=image_cfg,
        )

    def _parse_response(self, response: Any, config: Any) -> ImageGenerationResult:
        """Extract the first non-thought inline image from the response."""
        model_name: str = config.name

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens: int | None = getattr(usage, "prompt_token_count", None)
        completion_tokens: int | None = getattr(usage, "candidates_token_count", None)
        total_tokens: int | None = getattr(usage, "total_token_count", None)

        candidates = getattr(response, "candidates", None) or []
        parts: list[Any] = []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = list(getattr(content, "parts", None) or [])
        # Also check response.parts (flat form returned by some SDK versions)
        if not parts:
            parts = list(getattr(response, "parts", None) or [])

        for part in parts:
            if getattr(part, "thought", False):
                continue
            inline = getattr(part, "inline_data", None)
            if inline is not None:
                image_bytes = getattr(inline, "data", None)
                mime_type = getattr(inline, "mime_type", None) or "image/png"
                if isinstance(image_bytes, bytes) and image_bytes:
                    return ImageGenerationResult(
                        success=True,
                        image_bytes=image_bytes,
                        mime_type=mime_type,
                        model=model_name,
                        cost=config.price_per_generation_usd,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )

        # No image found — surface any text as error context
        text_parts = [
            getattr(p, "text", None)
            for p in parts
            if not getattr(p, "thought", False) and getattr(p, "text", None)
        ]
        error_ctx = (
            " Text response: " + repr(" ".join(text_parts)[:200])
        ) if text_parts else ""
        return ImageGenerationResult(
            success=False,
            model=model_name,
            error=f"No image in Google Gemini response.{error_ctx}",
        )


__all__ = ["GoogleGeminiProvider"]
