"""
Tests for the ``generate_ai_image`` tool — LangChain wrapper around ImageEngine
that integrates with the session file registry.

Behavior contracts:
- Invoked with ``agent`` injection, saves image to session-isolated directory
  and calls ``agent.register_file(display_name, disk_path)``.
- Without an agent (e.g. unit-test harness), returns the raw absolute path as
  ``file_reference`` and skips registration.
- Pydantic schema validates required ``prompt``; exposes optional ``model``,
  ``aspect_ratio``, ``image_size``, ``reference_images``; hides ``agent``
  via ``InjectedToolArg`` (so args_schema must declare it).
- Returns a dict with ``success``, ``file_reference``, ``cost``,
  ``mime_type``, ``size_bytes``, ``reference_images_warning``.
- On engine failure, returns ``success=False`` with the engine's ``error``
  string and no file side effects.
- ``_resolve_reference_images`` converts filenames in the agent's file
  registry to base64 data URIs; URLs and data URIs pass through unchanged;
  unresolvable names are skipped with a warning.

All tests patch ``ImageEngine.generate`` — no network calls.

Run:  pytest tools/_tests/test_generate_ai_image.py -v
"""

from __future__ import annotations

import base64
import contextlib
import os
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from pydantic import ValidationError
import pytest

from agent_ng.image_engine import ImageGenerationResult
from tools.tools import _resolve_reference_images, generate_ai_image

if TYPE_CHECKING:
    from pathlib import Path

# A single-pixel PNG for deterministic tests.
_PNG_1PX_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAj"
    "CB0C8AAAAASUVORK5CYII="
)


def _make_registry_agent(session_id: str = "sess-img-1") -> Any:
    """Minimal agent stub matching CmwAgent.file_registry shape."""

    class _Agent:
        def __init__(self) -> None:
            self.session_id = session_id
            self.file_registry: dict[tuple[str, str], str] = {}

        def register_file(self, name: str, path: str) -> None:
            self.file_registry[(self.session_id, name)] = path

        def get_file_path(self, name: str) -> str | None:
            p = self.file_registry.get((self.session_id, name))
            return p if p and os.path.isfile(p) else None

    return _Agent()


def _ok_result(**overrides: Any) -> ImageGenerationResult:
    defaults = {
        "success": True,
        "image_bytes": _PNG_1PX_BYTES,
        "mime_type": "image/png",
        "model": "google/gemini-2.5-flash-image",
        "cost": 0.039,
        "prompt_tokens": 12,
        "completion_tokens": 1296,
        "total_tokens": 1308,
        "generation_id": "gen-mock-1",
        "error": None,
    }
    defaults.update(overrides)
    return ImageGenerationResult(**defaults)


class TestInvocationWithAgent:
    """Happy path: agent injected, file registered."""

    def test_success_registers_file_and_returns_display_name(
        self, tmp_path: Path
    ) -> None:
        agent = _make_registry_agent()
        # Redirect session dir to tmp_path so we don't pollute .gradio/
        with (
            patch(
                "agent_ng.image_engine.ImageEngine.generate",
                return_value=_ok_result(),
            ),
            patch("tools.tools._IMAGE_OUTPUT_ROOT", str(tmp_path)),
        ):
            out = generate_ai_image.invoke(
                {"prompt": "a blue circle", "agent": agent}
            )

        assert isinstance(out, dict)
        assert out["success"] is True
        assert out["error"] is None
        # file_reference is the logical display name
        ref = out["file_reference"]
        assert isinstance(ref, str)
        assert ref.endswith(".png")
        # Agent registry should contain that name
        stored_path = agent.get_file_path(ref)
        assert stored_path is not None, "file should be registered with the agent"
        assert os.path.isfile(stored_path)
        with open(stored_path, "rb") as f:
            assert f.read() == _PNG_1PX_BYTES

    def test_success_returns_cost_and_size(self, tmp_path: Path) -> None:
        """LLM-facing payload exposes cost and size, never vendor metadata."""
        agent = _make_registry_agent()
        with (
            patch(
                "agent_ng.image_engine.ImageEngine.generate",
                return_value=_ok_result(cost=0.042),
            ),
            patch("tools.tools._IMAGE_OUTPUT_ROOT", str(tmp_path)),
        ):
            out = generate_ai_image.invoke(
                {"prompt": "a blue circle", "agent": agent}
            )

        assert out["cost"] == 0.042
        assert out["mime_type"] == "image/png"
        assert out["size_bytes"] == len(_PNG_1PX_BYTES)
        # The tool must not surface vendor-specific fields to the LLM.
        assert "model" not in out
        assert "generation_id" not in out
        assert "prompt_tokens" not in out

    def test_image_config_parameters_forwarded_to_engine(
        self, tmp_path: Path
    ) -> None:
        agent = _make_registry_agent()
        with (
            patch(
                "agent_ng.image_engine.ImageEngine.generate",
                return_value=_ok_result(),
            ) as gen,
            patch("tools.tools._IMAGE_OUTPUT_ROOT", str(tmp_path)),
        ):
            generate_ai_image.invoke(
                {
                    "prompt": "a wide banner",
                    "aspect_ratio": "16:9",
                    "image_size": "2K",
                    "agent": agent,
                }
            )

        kwargs = gen.call_args.kwargs
        assert kwargs.get("aspect_ratio") == "16:9"
        assert kwargs.get("image_size") == "2K"


class TestInvocationWithoutAgent:
    """Fallback path: no agent → return absolute path, skip registration."""

    def test_no_agent_returns_absolute_path(self, tmp_path: Path) -> None:
        with (
            patch(
                "agent_ng.image_engine.ImageEngine.generate",
                return_value=_ok_result(),
            ),
            patch("tools.tools._IMAGE_OUTPUT_ROOT", str(tmp_path)),
        ):
            out = generate_ai_image.invoke({"prompt": "a blue circle"})

        assert out["success"] is True
        ref = out["file_reference"]
        # Without an agent, file_reference is an absolute on-disk path
        assert os.path.isabs(ref)
        assert os.path.isfile(ref)
        try:
            with open(ref, "rb") as f:
                assert f.read() == _PNG_1PX_BYTES
        finally:
            # Clean up so pytest doesn't leave orphans between runs
            with contextlib.suppress(OSError):
                os.unlink(ref)


class TestErrorPropagation:
    def test_engine_failure_surfaces_as_tool_failure(self) -> None:
        agent = _make_registry_agent()
        with patch(
            "agent_ng.image_engine.ImageEngine.generate",
            return_value=ImageGenerationResult(
                success=False, error="HTTP 403: region blocked"
            ),
        ):
            out = generate_ai_image.invoke(
                {"prompt": "a cat", "agent": agent}
            )

        assert out["success"] is False
        assert "403" in out["error"]
        # No file registered on failure
        assert agent.file_registry == {}

    def test_unknown_model_surfaces_engine_error(self) -> None:
        agent = _make_registry_agent()
        with patch(
            "agent_ng.image_engine.ImageEngine.generate",
            return_value=ImageGenerationResult(
                success=False, error="Unknown model: bogus/x"
            ),
        ):
            out = generate_ai_image.invoke(
                {"prompt": "a cat", "model": "bogus/x", "agent": agent}
            )

        assert out["success"] is False
        assert "bogus/x" in out["error"]

    def test_missing_prompt_raises_validation_error(self) -> None:
        """Pydantic schema enforces prompt presence."""
        with pytest.raises(ValidationError):
            generate_ai_image.invoke({})


class TestSchemaVisibility:
    """Confirm ``agent`` is declared on the schema (so LangChain preserves the injection).

    Regression guard: if ``agent`` is only on the function signature but omitted
    from the Pydantic schema, ``BaseTool._parse_input`` silently drops it — the
    tool runs without access to the session registry.
    """

    def test_agent_field_exists_on_args_schema(self) -> None:
        schema_cls = generate_ai_image.args_schema
        assert schema_cls is not None
        assert "agent" in schema_cls.model_fields, (
            "agent must be a schema field with InjectedToolArg to survive "
            "LangChain's input parsing"
        )

    def test_schema_exposes_only_intended_fields(self) -> None:
        """Schema surface is exactly the five intended fields.

        ``model`` is intentionally absent: the active image generator is an
        operations-level decision (IMAGE_GEN_DEFAULT_MODEL) that must not
        be controllable from the LLM.
        """
        fields = set(generate_ai_image.args_schema.model_fields.keys())
        assert fields == {
            "prompt",
            "aspect_ratio",
            "image_size",
            "reference_images",
            "agent",
        }, f"unexpected schema fields: {fields}"


class TestLLMFacingDescription:
    """Guard against leaking vendor / slug / infrastructure detail to the LLM."""

    _FORBIDDEN_LEAKS = (
        "OpenRouter",
        "file_registry",
        "file registry",
        "ImageEngine",
        # Vendor names the LLM should not know about:
        "Gemini",
        "Flux",
        "FLUX",
        "Nano Banana",
        "Seedream",
        "Riverflow",
        "GPT-5",
        # Slugs from the registry:
        "google/",
        "openai/",
        "black-forest-labs/",
        "bytedance-seed/",
        "sourceful/",
    )

    def test_tool_description_has_no_leaks(self) -> None:
        desc = generate_ai_image.description or ""
        leaks = [s for s in self._FORBIDDEN_LEAKS if s in desc]
        assert not leaks, f"tool description leaks infrastructure detail: {leaks}"

    def test_param_descriptions_have_no_leaks(self) -> None:
        for name, field in generate_ai_image.args_schema.model_fields.items():
            text = field.description or ""
            leaks = [s for s in self._FORBIDDEN_LEAKS if s in text]
            assert not leaks, (
                f"param {name!r} description leaks: {leaks} (text={text!r})"
            )


class TestResolveReferenceImages:
    """``_resolve_reference_images`` contracts.

    - HTTP/HTTPS URLs pass through unchanged (remote provider fetches them).
    - Existing data URIs (``data:...``) pass through unchanged.
    - Filenames present in the agent's file registry are resolved to bytes
      and returned as ``data:<mime>;base64,<b64>`` URIs.
    - MIME type is guessed from the file extension; unknown extensions fall
      back to ``image/jpeg``.
    - Unresolvable names (not in registry, not on disk) are silently skipped
      with a WARNING log — the remaining items are still returned.
    - An empty input list returns an empty list.
    - A ``None`` agent skips registry lookup (filenames unresolvable).
    """

    def test_http_url_passes_through(self) -> None:
        items = ["https://example.com/foo.png"]
        assert _resolve_reference_images(items, agent=None) == items

    def test_http_non_ssl_url_passes_through(self) -> None:
        items = ["http://cdn.local/img.jpg"]
        assert _resolve_reference_images(items, agent=None) == items

    def test_data_uri_passes_through(self) -> None:
        item = "data:image/png;base64,abc123=="
        assert _resolve_reference_images([item], agent=None) == [item]

    def test_empty_list_returns_empty(self) -> None:
        assert _resolve_reference_images([], agent=None) == []

    def test_filename_resolved_to_data_uri(self, tmp_path: Path) -> None:
        img_path = tmp_path / "ref.png"
        img_path.write_bytes(_PNG_1PX_BYTES)

        agent = _make_registry_agent()
        agent.register_file("ref.png", str(img_path))

        result = _resolve_reference_images(["ref.png"], agent=agent)

        assert len(result) == 1
        assert result[0].startswith("data:image/png;base64,")
        _, b64 = result[0].split(",", 1)
        assert base64.b64decode(b64) == _PNG_1PX_BYTES

    def test_jpeg_extension_gives_jpeg_mime(self, tmp_path: Path) -> None:
        img_path = tmp_path / "photo.jpg"
        img_path.write_bytes(_PNG_1PX_BYTES)

        agent = _make_registry_agent()
        agent.register_file("photo.jpg", str(img_path))

        result = _resolve_reference_images(["photo.jpg"], agent=agent)

        assert result[0].startswith("data:image/jpeg;base64,")

    def test_unknown_extension_falls_back_to_jpeg_mime(
        self, tmp_path: Path
    ) -> None:
        # .xyz is not in mimetypes — should fall back to image/jpeg.
        img_path = tmp_path / "mystery.xyz"
        img_path.write_bytes(_PNG_1PX_BYTES)

        agent = _make_registry_agent()
        agent.register_file("mystery.xyz", str(img_path))

        result = _resolve_reference_images(["mystery.xyz"], agent=agent)

        assert result[0].startswith("data:image/jpeg;base64,")

    def test_unresolvable_filename_skipped_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            result = _resolve_reference_images(["ghost.png"], agent=None)

        assert result == []
        assert any("ghost.png" in r.message for r in caplog.records)

    def test_mixed_list_resolves_correctly(self, tmp_path: Path) -> None:
        img_path = tmp_path / "ref.jpg"
        img_path.write_bytes(_PNG_1PX_BYTES)

        agent = _make_registry_agent()
        agent.register_file("ref.jpg", str(img_path))

        items = [
            "https://cdn.example.com/style.png",
            "ref.jpg",
            "data:image/jpeg;base64,/9j/abc",
        ]
        result = _resolve_reference_images(items, agent=agent)

        assert len(result) == 3
        assert result[0] == "https://cdn.example.com/style.png"
        assert result[1].startswith("data:image/jpeg;base64,")
        assert result[2] == "data:image/jpeg;base64,/9j/abc"

    def test_unresolvable_items_do_not_block_valid_ones(
        self, tmp_path: Path
    ) -> None:
        good_path = tmp_path / "good.png"
        good_path.write_bytes(_PNG_1PX_BYTES)

        agent = _make_registry_agent()
        agent.register_file("good.png", str(good_path))

        result = _resolve_reference_images(
            ["bad_filename.png", "good.png"], agent=agent
        )

        assert len(result) == 1
        assert result[0].startswith("data:image/png;base64,")

    def test_none_agent_skips_registry_resolution(self) -> None:
        result = _resolve_reference_images(["some_file.png"], agent=None)
        assert result == []
