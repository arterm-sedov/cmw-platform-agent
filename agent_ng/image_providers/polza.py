"""
Polza.ai image-generation adapter.

Implements :class:`agent_ng.image_providers.base.ImageProvider` against
Polza.ai's ``/api/v1/media`` endpoint.

Flow:
1. POST to ``/api/v1/media`` — returns ``{id, status: "pending", ...}``.
2. Poll ``GET /api/v1/media/{id}`` every 3 s until ``completed`` or ``failed``
   (or until the configured timeout).
3. On ``completed``, download the image from the CDN URL in ``data.url``.
4. Convert ``cost_rub`` → USD using ``POLZA_RUB_TO_USD_RATE`` (default 90).

Supports ``aspect_ratio`` and ``image_resolution`` (1K / 2K / 4K) for models
that document these parameters (controlled by ``supports_image_config``).
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
import requests

from .base import ImageGenerationResult, ImageProvider, ImageRequest

load_dotenv()
logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://polza.ai/api"
_POLL_INTERVAL = 3.0  # seconds between status checks
_DEFAULT_TIMEOUT = 180.0  # total max wait for generation
_DEFAULT_RUB_RATE = 90.0  # RUB per 1 USD fallback


def _get_rub_rate() -> float:
    env_val = os.getenv("POLZA_RUB_TO_USD_RATE", "").strip()
    if env_val:
        with contextlib.suppress(ValueError):
            return float(env_val)
        logger.warning(
            "POLZA_RUB_TO_USD_RATE='%s' is not valid; using %.0f",
            env_val,
            _DEFAULT_RUB_RATE,
        )
    return _DEFAULT_RUB_RATE


class PolzaProvider(ImageProvider):
    """Image provider backed by the Polza.ai media generation API."""

    name = "polza"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        key = api_key if api_key is not None else os.getenv("POLZA_API_KEY")
        if not key:
            msg = (
                "POLZA_API_KEY is required (pass api_key= or set the "
                "environment variable)."
            )
            raise ValueError(msg)
        self.api_key: str = key
        raw_base = (
            base_url or os.getenv("POLZA_BASE_URL") or _DEFAULT_BASE_URL
        ).rstrip("/")
        # Normalize: POLZA_BASE_URL may be set to ".../api/v1" (chat endpoint
        # style), but media endpoints live at ".../api/v1/media". Strip any
        # trailing "/v1" so we always append the full path ourselves.
        self.base_url: str = raw_base.removesuffix("/v1")
        self.timeout: float = timeout

    # --------------------------------------------------------------- #
    # ImageProvider                                                   #
    # --------------------------------------------------------------- #

    def generate(self, request: ImageRequest) -> ImageGenerationResult:
        try:
            resp = requests.post(
                f"{self.base_url}/v1/media",
                headers=self._headers(),
                json=self._build_payload(request),
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as exc:
            return ImageGenerationResult(
                success=False,
                model=request.config.name,
                error=f"Submit timed out: {exc}",
            )
        except requests.exceptions.RequestException as exc:
            return ImageGenerationResult(
                success=False,
                model=request.config.name,
                error=f"Submit failed: {exc}",
            )

        if not resp.ok:
            body = (resp.text or "")[:500]
            return ImageGenerationResult(
                success=False,
                model=request.config.name,
                error=f"HTTP {resp.status_code}: {body}",
            )

        try:
            data = resp.json()
        except ValueError as exc:
            return ImageGenerationResult(
                success=False,
                model=request.config.name,
                error=f"Non-JSON response: {exc}",
            )

        status = data.get("status", "")
        logger.debug("Polza media submit status=%s id=%s", status, data.get("id"))

        # Synchronous mode (async=false, the default): API blocks until done
        # and returns the completed result in the POST response.
        if status == "completed":
            return self._extract_result(data, request.config.name)

        if status == "failed":
            err_obj = data.get("error") or {}
            msg = (
                err_obj.get("message") or str(err_obj)
                if isinstance(err_obj, dict)
                else str(err_obj)
            )
            return ImageGenerationResult(
                success=False,
                model=request.config.name,
                error=f"Generation failed: {msg}",
            )

        # Async mode: poll until completed.
        gen_id = data.get("id")
        if not gen_id or not isinstance(gen_id, str):
            return ImageGenerationResult(
                success=False,
                model=request.config.name,
                error=f"No generation id in response: {data}",
            )
        return self._poll_until_done(gen_id, request.config.name)

    # --------------------------------------------------------------- #
    # Internals                                                       #
    # --------------------------------------------------------------- #

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, request: ImageRequest) -> dict[str, Any]:
        model_id = request.config.provider_model_ids.get("polza", request.config.name)
        input_body: dict[str, Any] = {"prompt": request.prompt}
        if request.config.supports_image_config:
            # aspect_ratio is required by Polza even when docs say it has a default.
            input_body["aspect_ratio"] = request.aspect_ratio or "1:1"
            if request.image_size:
                # Polza uses quality=basic(2K)|high(4K) instead of image_resolution.
                input_body["quality"] = (
                    "high" if request.image_size.lower() in ("4k", "high") else "basic"
                )
        logger.debug(
            "PolzaProvider POST model=%s input=%s", model_id, input_body
        )
        return {"model": model_id, "input": input_body}

    def _poll_until_done(self, gen_id: str, model: str) -> ImageGenerationResult:
        """Poll until completed/failed, then download the image."""
        deadline = time.monotonic() + self.timeout
        url = f"{self.base_url}/v1/media/{gen_id}"

        while time.monotonic() < deadline:
            try:
                resp = requests.get(url, headers=self._headers(), timeout=15.0)
            except requests.exceptions.RequestException as exc:
                logger.warning("Poll error for %s: %s", gen_id, exc)
                time.sleep(_POLL_INTERVAL)
                continue

            if not resp.ok:
                return ImageGenerationResult(
                    success=False,
                    model=model,
                    error=f"Poll HTTP {resp.status_code}: {(resp.text or '')[:300]}",
                )
            try:
                data = resp.json()
            except ValueError as exc:
                return ImageGenerationResult(
                    success=False, model=model, error=f"Non-JSON poll response: {exc}"
                )

            status = data.get("status", "")
            if status == "completed":
                return self._extract_result(data, model)
            if status == "failed":
                err_obj = data.get("error") or {}
                msg = (
                    err_obj.get("message") or str(err_obj) or "generation failed"
                    if isinstance(err_obj, dict)
                    else str(err_obj)
                )
                return ImageGenerationResult(
                    success=False, model=model, error=f"Generation failed: {msg}"
                )

            logger.debug("Polza gen %s status=%s — polling…", gen_id, status)
            time.sleep(_POLL_INTERVAL)

        return ImageGenerationResult(
            success=False,
            model=model,
            error=f"Generation timed out after {self.timeout}s (id={gen_id})",
        )

    def _extract_result(
        self, data: dict[str, Any], model: str
    ) -> ImageGenerationResult:
        """Download the image from the CDN URL and build the result."""
        raw_data = data.get("data")
        # API returns either a dict {"url": ...} or a list [{"url": ...}]
        if isinstance(raw_data, list):
            image_data = raw_data[0] if raw_data else {}
        elif isinstance(raw_data, dict):
            image_data = raw_data
        else:
            image_data = {}
        cdn_url = image_data.get("url") if isinstance(image_data, dict) else None
        if not cdn_url or not isinstance(cdn_url, str):
            return ImageGenerationResult(
                success=False,
                model=model,
                error=f"No image URL in completed response: {data}",
            )

        try:
            img_resp = requests.get(cdn_url, timeout=60.0)
        except requests.exceptions.RequestException as exc:
            return ImageGenerationResult(
                success=False, model=model, error=f"Image download failed: {exc}"
            )

        if not img_resp.ok:
            return ImageGenerationResult(
                success=False,
                model=model,
                error=f"Image download HTTP {img_resp.status_code}",
            )

        image_bytes = img_resp.content
        content_type = img_resp.headers.get("Content-Type", "image/jpeg").split(";")[0]

        usage = data.get("usage") or {}
        cost_rub = usage.get("cost_rub") or usage.get("cost")
        cost_usd: float | None = None
        if isinstance(cost_rub, (int, float)):
            cost_usd = cost_rub / _get_rub_rate()

        def _int(key: str) -> int | None:
            v = usage.get(key)
            return int(v) if isinstance(v, (int, float)) else None

        return ImageGenerationResult(
            success=True,
            image_bytes=image_bytes,
            mime_type=content_type,
            model=model,
            cost=cost_usd,
            prompt_tokens=_int("input_tokens"),
            completion_tokens=_int("output_tokens"),
            total_tokens=_int("total_tokens"),
            generation_id=data.get("id") if isinstance(data.get("id"), str) else None,
        )


__all__ = ["PolzaProvider"]
