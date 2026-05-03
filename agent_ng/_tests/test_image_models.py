"""
Tests for image model registry.

Behavior contracts:
- Registry returns a non-empty dict of ImageModelConfig keyed by OpenRouter slug.
- The default model is resolvable and belongs to the registry.
- ``IMAGE_GEN_DEFAULT_MODEL`` env var overrides the default for dev/testing.
- Each config declares modalities and whether it accepts ``image_config``.

Run:  pytest agent_ng/_tests/test_image_models.py -v
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from agent_ng.image_models import (
    DEFAULT_IMAGE_MODEL,
    IMAGE_MODELS,
    ImageModelConfig,
    get_default_model,
    get_image_models,
    get_model_config,
)


class TestRegistryShape:
    """Registry exposes at least the verified working models with correct metadata."""

    def test_registry_non_empty(self) -> None:
        models = get_image_models()
        assert isinstance(models, dict)
        assert len(models) >= 3, (
            "Expected at least 3 verified models (Gemini, Flux, Seedream)"
        )

    def test_registry_keys_match_config_name(self) -> None:
        for slug, cfg in get_image_models().items():
            assert cfg.name == slug, (
                f"Registry key {slug} must equal config.name {cfg.name}"
            )

    def test_every_config_is_image_model_config(self) -> None:
        for cfg in get_image_models().values():
            assert isinstance(cfg, ImageModelConfig)

    def test_gemini_default_present(self) -> None:
        models = get_image_models()
        assert "google/gemini-2.5-flash-image" in models

    def test_flux_present_with_image_only_modalities(self) -> None:
        models = get_image_models()
        flux = models.get("black-forest-labs/flux.2-pro")
        assert flux is not None
        assert flux.modalities == ["image"], (
            "Flux is an image-only model per OpenRouter docs"
        )

    def test_seedream_present_with_image_only_modalities(self) -> None:
        models = get_image_models()
        seed = models.get("bytedance-seed/seedream-4.5")
        assert seed is not None
        assert seed.modalities == ["image"]

    def test_gemini_has_text_and_image_modalities(self) -> None:
        """Gemini models return both text and images."""
        gemini = get_image_models()["google/gemini-2.5-flash-image"]
        assert set(gemini.modalities) == {"image", "text"}

    def test_only_gemini_supports_image_config(self) -> None:
        """Google Gemini and Seedream 4.5 document image_config support."""
        _supports = {"google/", "bytedance-seed/seedream-4.5"}
        for slug, cfg in get_image_models().items():
            expected = any(cfg.name.startswith(p) or cfg.name == p for p in _supports)
            assert cfg.supports_image_config is expected, (
                f"{slug}: supports_image_config should be {expected}"
            )

    def test_every_config_has_known_providers(self) -> None:
        known = {"openrouter", "polza"}
        for slug, cfg in get_image_models().items():
            assert isinstance(cfg.providers, list), (
                f"{slug}: providers must be a list"
            )
            assert cfg.providers, f"{slug}: providers must not be empty"
            for p in cfg.providers:
                assert p in known, (
                    f"{slug}: unknown provider {p!r}; add adapter before using it."
                )

    def test_provider_property_returns_first(self) -> None:
        """Backward-compat .provider property returns providers[0]."""
        cfg = get_image_models()["google/gemini-3.1-flash-image-preview"]
        assert cfg.provider == cfg.providers[0]

    def test_gemini_default_has_polza_first(self) -> None:
        """Default Gemini model routes through Polza first (Russian CDN)."""
        cfg = get_image_models()["google/gemini-3.1-flash-image-preview"]
        assert cfg.providers[0] == "polza", (
            "Polza should be first provider for Gemini to avoid regional restrictions"
        )


class TestDefaultSelection:
    """``get_default_model`` honors env override and falls back to the default."""

    def test_compile_time_default_is_registered(self) -> None:
        """Whichever slug is chosen as default must live in the registry."""
        assert DEFAULT_IMAGE_MODEL in IMAGE_MODELS

    def test_default_without_env_returns_compile_time_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("IMAGE_GEN_DEFAULT_MODEL", None)
            assert get_default_model() == DEFAULT_IMAGE_MODEL

    def test_env_override_wins_when_model_registered(self) -> None:
        override = "black-forest-labs/flux.2-pro"
        with patch.dict(os.environ, {"IMAGE_GEN_DEFAULT_MODEL": override}, clear=False):
            assert get_default_model() == override

    def test_env_override_ignored_when_model_unknown(self) -> None:
        """Unknown models fall back to the compile-time default + warn (never crash)."""
        with patch.dict(
            os.environ,
            {"IMAGE_GEN_DEFAULT_MODEL": "bogus/model-does-not-exist"},
            clear=False,
        ):
            assert get_default_model() == DEFAULT_IMAGE_MODEL


class TestGetModelConfig:
    """Explicit lookup helper used by the engine to validate slug + fetch config."""

    def test_known_slug_returns_config(self) -> None:
        cfg = get_model_config("google/gemini-2.5-flash-image")
        assert cfg is not None
        assert isinstance(cfg, ImageModelConfig)

    def test_unknown_slug_returns_none(self) -> None:
        assert get_model_config("bogus/nope") is None

    def test_empty_string_returns_none(self) -> None:
        assert get_model_config("") is None

    @pytest.mark.parametrize("bad", [None, 123, [], {}])
    def test_non_string_returns_none(self, bad: object) -> None:
        assert get_model_config(bad) is None  # type: ignore[arg-type]
