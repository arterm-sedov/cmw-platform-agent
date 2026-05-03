"""
Tests for ImageEngine — OpenRouter image-generation adapter.

Behavior contracts:
- Successful calls return bytes + mime_type + cost + token counts from
  ``response.usage.cost`` (confirmed by OpenRouter Usage Accounting docs).
- HTTP errors surface with a helpful message and ``success=False``.
- Malformed / empty responses are handled without crashing.
- ``image_config`` is sent only for models whose config declares
  ``supports_image_config=True``.
- Image-only models (Flux, Seedream) receive ``modalities=["image"]``;
  multimodal models (Gemini) receive ``modalities=["image","text"]``.
- Missing API key raises a clear ValueError at construction time.

Unit tests mock ``requests.post``. One gated integration test runs against
the real API when ``OPENROUTER_API_KEY`` is present.

Run:  pytest agent_ng/_tests/test_image_engine.py -v
"""

from __future__ import annotations

import base64
import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from agent_ng.image_engine import ImageEngine, ImageGenerationResult
from agent_ng.image_models import get_default_model

# Target for mocking the OpenRouter HTTP call made by the adapter.
_POST_TARGET = "agent_ng.image_providers.openrouter.requests.post"

# A single-pixel PNG, base64-encoded, for realistic mocked responses.
_PNG_1PX_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAj"
    "CB0C8AAAAASUVORK5CYII="
)
_PNG_1PX_BYTES = base64.b64decode(_PNG_1PX_B64)


def _ok_response(
    *,
    b64: str = _PNG_1PX_B64,
    mime: str = "image/png",
    cost: float = 0.039,
    prompt_tokens: int = 12,
    completion_tokens: int = 1296,
    total_tokens: int = 1308,
    generation_id: str = "gen-test-1",
) -> MagicMock:
    mock = MagicMock(spec=requests.Response)
    mock.status_code = 200
    mock.ok = True
    mock.json.return_value = {
        "id": generation_id,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Here is your image.",
                    "images": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        }
                    ],
                }
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
        },
    }
    return mock


def _err_response(status: int, body: str = "forbidden") -> MagicMock:
    mock = MagicMock(spec=requests.Response)
    mock.status_code = status
    mock.ok = False
    mock.text = body
    mock.json.side_effect = ValueError("no json")
    return mock


class TestEngineConstruction:
    def test_no_key_still_constructs(self) -> None:
        # Engine no longer validates keys at construction — each provider
        # validates its own credentials at generate-time so that Polza-only
        # (or other non-OpenRouter) deployments can construct without an
        # OPENROUTER_API_KEY.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENROUTER_API_KEY", None)
            engine = ImageEngine(api_key=None)
            assert engine.api_key is None  # No OpenRouter key resolved.

    def test_accepts_explicit_api_key(self) -> None:
        engine = ImageEngine(api_key="sk-test")
        assert engine.api_key == "sk-test"

    def test_reads_env_api_key_when_none_passed(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-env"}, clear=False):
            engine = ImageEngine()
            assert engine.api_key == "sk-env"

    def test_default_base_url(self) -> None:
        engine = ImageEngine(api_key="sk-test")
        assert engine.base_url.endswith("/chat/completions")


class TestSuccessfulGeneration:
    @pytest.fixture(autouse=True)
    def _use_openrouter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # These tests mock only the OpenRouter HTTP layer; force that provider.
        monkeypatch.setenv("IMAGE_GEN_PROVIDER", "openrouter")

    def test_gemini_returns_bytes_cost_and_tokens(self) -> None:
        engine = ImageEngine(api_key="sk-test")
        with patch(_POST_TARGET, return_value=_ok_response()):
            result = engine.generate("a cat", model="google/gemini-2.5-flash-image")

        assert isinstance(result, ImageGenerationResult)
        assert result.success is True
        assert result.image_bytes == _PNG_1PX_BYTES
        assert result.mime_type == "image/png"
        assert result.model == "google/gemini-2.5-flash-image"
        assert result.cost == 0.039
        assert result.prompt_tokens == 12
        assert result.completion_tokens == 1296
        assert result.total_tokens == 1308
        assert result.generation_id == "gen-test-1"
        assert result.error is None

    def test_flux_uses_image_only_modalities(self) -> None:
        engine = ImageEngine(api_key="sk-test")
        with patch(
            _POST_TARGET, return_value=_ok_response()
        ) as post:
            engine.generate("a cat", model="black-forest-labs/flux.2-pro")

        payload = post.call_args.kwargs["json"]
        assert payload["modalities"] == ["image"]
        assert payload["model"] == "black-forest-labs/flux.2-pro"

    def test_gemini_uses_text_and_image_modalities(self) -> None:
        engine = ImageEngine(api_key="sk-test")
        with patch(
            _POST_TARGET, return_value=_ok_response()
        ) as post:
            engine.generate("a cat", model="google/gemini-2.5-flash-image")

        payload = post.call_args.kwargs["json"]
        assert set(payload["modalities"]) == {"image", "text"}

    def test_default_model_used_when_none_passed(self) -> None:
        """Engine picks the registry default when no ``model`` is given."""
        _polza_post = "agent_ng.image_providers.polza.requests.post"
        _polza_get = "agent_ng.image_providers.polza.requests.get"
        # Default model is now Polza — mock submit + poll responses.
        submit_resp = MagicMock(spec=requests.Response)
        submit_resp.ok = True
        submit_resp.json.return_value = {
            "id": "aig_test123",
            "status": "pending",
            "model": "google/gemini-3.1-flash-image-preview",
        }
        poll_resp = MagicMock(spec=requests.Response)
        poll_resp.ok = True
        poll_resp.json.return_value = {
            "id": "aig_test123",
            "status": "completed",
            "model": "google/gemini-3.1-flash-image-preview",
            "data": {"url": "https://s3.polza.ai/test.png"},
            "usage": {"cost_rub": 5.0},
        }
        img_resp = MagicMock(spec=requests.Response)
        img_resp.ok = True
        img_resp.content = _PNG_1PX_BYTES
        img_resp.headers = {"Content-Type": "image/png"}

        engine = ImageEngine()
        with (
            patch(_polza_post, return_value=submit_resp) as polza_post,
            patch(_polza_get, side_effect=[poll_resp, img_resp]),
        ):
            engine.generate("a cat")

        # Verify the default model slug (the API slug, not the registry key).
        submitted = polza_post.call_args.kwargs["json"]
        assert submitted["model"] == "google/gemini-3.1-flash-image-preview"
        assert submitted["model"] == get_default_model()

    def test_authorization_header_set(self) -> None:
        engine = ImageEngine(api_key="sk-test")
        with patch(
            _POST_TARGET, return_value=_ok_response()
        ) as post:
            engine.generate("a cat", model="google/gemini-2.5-flash-image")

        headers = post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["Content-Type"] == "application/json"


class TestImageConfigForwarding:
    """image_config is forwarded only for Gemini, ignored for Flux/Seedream."""

    @pytest.fixture(autouse=True)
    def _use_openrouter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IMAGE_GEN_PROVIDER", "openrouter")

    def test_image_config_forwarded_for_gemini(self) -> None:
        engine = ImageEngine(api_key="sk-test")
        with patch(
            _POST_TARGET, return_value=_ok_response()
        ) as post:
            engine.generate(
                "a cat",
                model="google/gemini-2.5-flash-image",
                aspect_ratio="16:9",
                image_size="2K",
            )

        payload = post.call_args.kwargs["json"]
        expected = {"aspect_ratio": "16:9", "image_size": "2K"}
        assert payload.get("image_config") == expected

    def test_image_config_omitted_for_flux(self) -> None:
        engine = ImageEngine(api_key="sk-test")
        with patch(
            _POST_TARGET, return_value=_ok_response()
        ) as post:
            engine.generate(
                "a cat",
                model="black-forest-labs/flux.2-pro",
                aspect_ratio="16:9",
                image_size="2K",
            )

        payload = post.call_args.kwargs["json"]
        assert "image_config" not in payload

    def test_image_config_omitted_when_no_params_passed(self) -> None:
        engine = ImageEngine(api_key="sk-test")
        with patch(
            _POST_TARGET, return_value=_ok_response()
        ) as post:
            engine.generate("a cat", model="google/gemini-2.5-flash-image")

        payload = post.call_args.kwargs["json"]
        assert "image_config" not in payload

    def test_partial_image_config_accepted(self) -> None:
        """Only aspect_ratio provided → only aspect_ratio sent."""
        engine = ImageEngine(api_key="sk-test")
        with patch(
            _POST_TARGET, return_value=_ok_response()
        ) as post:
            engine.generate(
                "a cat",
                model="google/gemini-2.5-flash-image",
                aspect_ratio="1:1",
            )

        payload = post.call_args.kwargs["json"]
        assert payload["image_config"] == {"aspect_ratio": "1:1"}


class TestErrorHandling:
    @pytest.fixture(autouse=True)
    def _use_openrouter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IMAGE_GEN_PROVIDER", "openrouter")

    def test_http_error_returns_failure_with_status(self) -> None:
        engine = ImageEngine(api_key="sk-test")
        with patch(
            _POST_TARGET,
            return_value=_err_response(403, "region blocked"),
        ):
            result = engine.generate("a cat", model="google/gemini-2.5-flash-image")

        assert result.success is False
        assert result.image_bytes is None
        assert result.error is not None
        assert "403" in result.error
        assert "region blocked" in result.error

    def test_unknown_model_returns_failure_without_calling_api(self) -> None:
        engine = ImageEngine(api_key="sk-test")
        with patch(_POST_TARGET) as post:
            result = engine.generate("a cat", model="bogus/unknown-model")

        assert result.success is False
        assert result.error is not None
        assert "bogus/unknown-model" in result.error
        post.assert_not_called()

    def test_timeout_returns_failure(self) -> None:
        engine = ImageEngine(api_key="sk-test")
        with patch(
            _POST_TARGET,
            side_effect=requests.exceptions.Timeout("timed out"),
        ):
            result = engine.generate("a cat", model="google/gemini-2.5-flash-image")

        assert result.success is False
        assert "timed out" in (result.error or "").lower()

    def test_connection_error_returns_failure(self) -> None:
        engine = ImageEngine(api_key="sk-test")
        with patch(
            _POST_TARGET,
            side_effect=requests.exceptions.ConnectionError("dns fail"),
        ):
            result = engine.generate("a cat", model="google/gemini-2.5-flash-image")

        assert result.success is False
        assert result.error is not None

    def test_no_images_in_response_returns_failure(self) -> None:
        engine = ImageEngine(api_key="sk-test")
        mock = MagicMock(spec=requests.Response)
        mock.status_code = 200
        mock.ok = True
        mock.json.return_value = {
            "id": "gen-empty",
            "choices": [
                {"message": {"role": "assistant", "content": "refusal", "images": []}}
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "cost": 0.0,
            },
        }
        with patch(_POST_TARGET, return_value=mock):
            result = engine.generate("a cat", model="google/gemini-2.5-flash-image")

        assert result.success is False
        assert "no image" in (result.error or "").lower()

    def test_malformed_data_url_returns_failure(self) -> None:
        engine = ImageEngine(api_key="sk-test")
        bad = _ok_response()
        images = bad.json.return_value["choices"][0]["message"]["images"]
        images[0]["image_url"]["url"] = "not-a-data-url"
        with patch(_POST_TARGET, return_value=bad):
            result = engine.generate("a cat", model="google/gemini-2.5-flash-image")

        assert result.success is False
        assert result.error is not None

    def test_missing_usage_defaults_to_none_cost_but_still_succeeds(self) -> None:
        """Cost/tokens missing is not fatal — generation is still a success."""
        engine = ImageEngine(api_key="sk-test")
        mock = _ok_response()
        mock.json.return_value.pop("usage", None)
        with patch(_POST_TARGET, return_value=mock):
            result = engine.generate("a cat", model="google/gemini-2.5-flash-image")

        assert result.success is True
        assert result.image_bytes == _PNG_1PX_BYTES
        assert result.cost is None
        assert result.total_tokens is None


class TestGenerationIdFallback:
    """generation_id is a nice-to-have; missing id doesn't cause failure."""

    @pytest.fixture(autouse=True)
    def _use_openrouter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IMAGE_GEN_PROVIDER", "openrouter")

    def test_missing_id_is_none(self) -> None:
        engine = ImageEngine(api_key="sk-test")
        mock = _ok_response()
        mock.json.return_value.pop("id", None)
        with patch(_POST_TARGET, return_value=mock):
            result = engine.generate("a cat", model="google/gemini-2.5-flash-image")

        assert result.success is True
        assert result.generation_id is None


# ------------------------------------------------------------------------- #
# Live integration test — gated by OPENROUTER_API_KEY presence               #
# ------------------------------------------------------------------------- #


def _has_openrouter_key() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


def _has_polza_key() -> bool:
    return bool(os.getenv("POLZA_API_KEY"))


@pytest.mark.skipif(
    not _has_polza_key(),
    reason="POLZA_API_KEY not set; skipping live Polza call",
)
@pytest.mark.xfail(
    strict=False,
    reason="Live API: transient safety filters or network errors may cause flakes",
)
class TestImageEngineLivePolza:
    """Real API calls forced through Polza.ai (no regional restrictions)."""

    def test_gemini_via_polza(self) -> None:
        with patch.dict(os.environ, {"IMAGE_GEN_PROVIDER": "polza"}):
            engine = ImageEngine()
            result = engine.generate(
                "A minimalist geometric logo: simple blue square on white background",
                model="google/gemini-3.1-flash-image-preview",
            )
        assert result.success is True, f"generation failed: {result.error}"
        assert result.image_bytes is not None
        assert len(result.image_bytes) > 1024, "image suspiciously small"
        is_png = result.image_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        is_jpeg = result.image_bytes[:2] == b"\xff\xd8"
        assert is_png or is_jpeg, "expected PNG or JPEG output"
        assert result.cost is not None
        assert result.cost > 0

    def test_seedream_via_polza(self) -> None:
        with patch.dict(os.environ, {"IMAGE_GEN_PROVIDER": "polza"}):
            engine = ImageEngine()
            result = engine.generate(
                "Minimalist icon: blue circle on white background",
                model="bytedance-seed/seedream-4.5",
            )
        assert result.success is True, f"generation failed: {result.error}"
        assert result.image_bytes is not None
        assert len(result.image_bytes) > 1024, "image suspiciously small"

    def test_seedream5_lite_via_polza(self) -> None:
        with patch.dict(os.environ, {"IMAGE_GEN_PROVIDER": "polza"}):
            engine = ImageEngine()
            result = engine.generate(
                "Minimalist icon: blue circle on white background",
                model="bytedance-seed/seedream-5-lite",
            )
        assert result.success is True, f"generation failed: {result.error}"
        assert result.image_bytes is not None
        assert len(result.image_bytes) > 1024, "image suspiciously small"


@pytest.mark.skipif(
    not _has_openrouter_key(),
    reason="OPENROUTER_API_KEY not set; skipping live OpenRouter call",
)
@pytest.mark.xfail(
    strict=False,
    reason="Live API: regional restrictions or network errors may cause flakes",
)
class TestImageEngineLiveOpenRouter:
    """Real API calls forced through OpenRouter (may be geo-restricted)."""

    def test_seedream_via_openrouter(self) -> None:
        with patch.dict(os.environ, {"IMAGE_GEN_PROVIDER": "openrouter"}):
            engine = ImageEngine()
            result = engine.generate(
                "Minimalist icon: blue circle on white background",
                model="bytedance-seed/seedream-4.5",
            )
        assert result.success is True, f"generation failed: {result.error}"
        assert result.image_bytes is not None
        assert len(result.image_bytes) > 1024, "image suspiciously small"


def _has_gemini_key() -> bool:
    return bool(os.getenv("GEMINI_KEY"))


@pytest.mark.skipif(
    not _has_gemini_key(),
    reason="GEMINI_KEY not set; skipping live Google native call",
)
@pytest.mark.xfail(
    strict=False,
    reason="Live API: geo-restrictions (Russia) or quota errors may cause flakes",
)
class TestImageEngineLiveGoogle:
    """Real API calls forced through Google's native Gemini API.

    Requires ``GEMINI_KEY`` and unrestricted access to generativelanguage.googleapis.com
    (does not work from Russia/CIS without a VPN or proxy).
    """

    def test_gemini_flash_via_google(self) -> None:
        with patch.dict(os.environ, {"IMAGE_GEN_PROVIDER": "google"}):
            engine = ImageEngine()
            result = engine.generate(
                "A minimalist geometric logo: simple blue square on white background",
                model="google/gemini-3.1-flash-image-preview",
            )
        assert result.success is True, f"generation failed: {result.error}"
        assert result.image_bytes is not None
        assert len(result.image_bytes) > 1024, "image suspiciously small"
        is_png = result.image_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        is_jpeg = result.image_bytes[:2] == b"\xff\xd8"
        assert is_png or is_jpeg, "expected PNG or JPEG output"
        # Google provider uses static price from registry
        assert result.cost is not None
        assert result.cost > 0

    def test_gemini_pro_via_google(self) -> None:
        with patch.dict(os.environ, {"IMAGE_GEN_PROVIDER": "google"}):
            engine = ImageEngine()
            result = engine.generate(
                "A minimalist geometric logo: simple blue circle on white background",
                model="google/gemini-3-pro-image-preview",
            )
        assert result.success is True, f"generation failed: {result.error}"
        assert result.image_bytes is not None
        assert len(result.image_bytes) > 1024, "image suspiciously small"
