"""
OpenRouter image-generation adapter.

Implements :class:`agent_ng.image_providers.base.ImageProvider` against
OpenRouter's ``/chat/completions`` endpoint for image-output models.

Why a direct HTTP client instead of ``ChatOpenAI``: LangChain's chat client
silently discards the ``message.images`` field where generated images live.

Per-call cost is returned by OpenRouter in ``response.usage.cost`` (see
https://openrouter.ai/docs/guides/administration/usage-accounting).
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

from dotenv import load_dotenv
import requests

from .base import ImageGenerationResult, ImageProvider, ImageRequest

load_dotenv()
logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_TIMEOUT = 120.0  # Slower models (e.g. Riverflow) can take ~60s.


class OpenRouterProvider(ImageProvider):
    """Image provider backed by the OpenRouter chat-completions API."""

    name = "openrouter"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY")
        if not key:
            msg = (
                "OPENROUTER_API_KEY is required (pass api_key= or set the "
                "environment variable)."
            )
            raise ValueError(msg)
        self.api_key: str = key
        base = base_url or os.getenv("OPENROUTER_BASE_URL") or _DEFAULT_BASE_URL
        self.base_url: str = base.rstrip("/") + "/chat/completions"
        self.timeout: float = timeout

    # --------------------------------------------------------------- #
    # ImageProvider                                                   #
    # --------------------------------------------------------------- #

    def generate(self, request: ImageRequest) -> ImageGenerationResult:
        try:
            response = self._post(request)
        except requests.exceptions.Timeout as exc:
            return ImageGenerationResult(
                success=False,
                model=request.config.name,
                error=f"Request timed out after {self.timeout}s: {exc}",
            )
        except requests.exceptions.RequestException as exc:
            return ImageGenerationResult(
                success=False,
                model=request.config.name,
                error=f"Request failed: {exc}",
            )

        if not response.ok:
            body = (response.text or "")[:500]
            return ImageGenerationResult(
                success=False,
                model=request.config.name,
                error=f"HTTP {response.status_code}: {body}",
            )

        try:
            payload = response.json()
        except ValueError as exc:
            return ImageGenerationResult(
                success=False,
                model=request.config.name,
                error=f"Non-JSON response from OpenRouter: {exc}",
            )

        return self._parse_response(payload, request.config.name)

    # --------------------------------------------------------------- #
    # Internals                                                       #
    # --------------------------------------------------------------- #

    def _post(self, request: ImageRequest) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://cmw-platform-agent.local",
            "X-Title": "CMW Platform Agent",
        }
        payload: dict[str, Any] = {
            "model": request.config.name,
            "modalities": list(request.config.modalities),
            "messages": [{"role": "user", "content": request.prompt}],
        }
        image_config = self._build_image_config(request)
        if image_config is not None:
            payload["image_config"] = image_config

        logger.debug(
            "OpenRouterProvider POST model=%s modalities=%s image_config=%s",
            request.config.name,
            payload["modalities"],
            image_config,
        )
        return requests.post(
            self.base_url, headers=headers, json=payload, timeout=self.timeout
        )

    @staticmethod
    def _build_image_config(request: ImageRequest) -> dict[str, str] | None:
        if not (request.aspect_ratio or request.image_size):
            return None
        if not request.config.supports_image_config:
            logger.warning(
                "Model %s does not support image_config; dropping "
                "aspect_ratio=%r image_size=%r.",
                request.config.name,
                request.aspect_ratio,
                request.image_size,
            )
            return None
        ic: dict[str, str] = {}
        if request.aspect_ratio:
            ic["aspect_ratio"] = request.aspect_ratio
        if request.image_size:
            ic["image_size"] = request.image_size
        return ic

    @staticmethod
    def _parse_response(
        payload: dict[str, Any], model: str
    ) -> ImageGenerationResult:
        choices = payload.get("choices") or []
        if not choices:
            return ImageGenerationResult(
                success=False, model=model, error="Response had no choices."
            )
        message = choices[0].get("message") or {}
        images = message.get("images") or []
        if not images:
            return ImageGenerationResult(
                success=False,
                model=model,
                error=(
                    "Model returned no images. Text content: "
                    f"{(message.get('content') or '')[:200]!r}"
                ),
            )

        url = (images[0].get("image_url") or {}).get("url", "")
        if not isinstance(url, str) or not url.startswith("data:"):
            return ImageGenerationResult(
                success=False,
                model=model,
                error=f"Expected data URL, got: {str(url)[:120]!r}",
            )

        try:
            mime, image_bytes = _decode_data_url(url)
        except ValueError as exc:
            return ImageGenerationResult(
                success=False, model=model, error=f"Malformed data URL: {exc}"
            )

        usage = payload.get("usage") or {}

        def _num(key: str) -> float | int | None:
            v = usage.get(key)
            return v if isinstance(v, (int, float)) else None

        return ImageGenerationResult(
            success=True,
            image_bytes=image_bytes,
            mime_type=mime,
            model=model,
            cost=_num("cost"),
            prompt_tokens=(
                int(_num("prompt_tokens"))  # type: ignore[arg-type]
                if _num("prompt_tokens") is not None
                else None
            ),
            completion_tokens=(
                int(_num("completion_tokens"))  # type: ignore[arg-type]
                if _num("completion_tokens") is not None
                else None
            ),
            total_tokens=(
                int(_num("total_tokens"))  # type: ignore[arg-type]
                if _num("total_tokens") is not None
                else None
            ),
            generation_id=(
                payload.get("id") if isinstance(payload.get("id"), str) else None
            ),
        )


def _decode_data_url(url: str) -> tuple[str, bytes]:
    """Decode a ``data:<mime>;base64,<payload>`` URL into (mime, bytes)."""
    if not url.startswith("data:"):
        msg = "not a data URL"
        raise ValueError(msg)
    try:
        header, payload = url.split(",", 1)
    except ValueError as exc:
        msg = f"missing comma separator: {exc}"
        raise ValueError(msg) from exc
    meta = header[len("data:"):]
    parts = meta.split(";")
    mime = parts[0] or "application/octet-stream"
    if "base64" not in parts[1:]:
        msg = f"only base64 data URLs supported, got header: {header!r}"
        raise ValueError(msg)
    try:
        return mime, base64.b64decode(payload, validate=False)
    except (ValueError, TypeError) as exc:
        msg = f"invalid base64 payload: {exc}"
        raise ValueError(msg) from exc


__all__ = ["OpenRouterProvider"]
