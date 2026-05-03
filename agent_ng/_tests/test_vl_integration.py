"""
Integration tests for Vision-Language model support.

Uses real files and real API keys loaded from .env
Run:  pytest agent_ng/_tests/test_vl_integration.py -v -s

Test files (set VL_TEST_* to override; defaults under experiments/test_files/):
  VL_TEST_IMAGE  - image (default: test_image.jpg)
  VL_TEST_VIDEO  - MP4 (default: test_video.mp4, optional — see test_files/README.md)
  VL_TEST_AUDIO  - MP3 (default: test_audio.mp3, optional)
"""
import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch
from dotenv import load_dotenv

# Load .env so real API keys are available
load_dotenv()

# Repo root: agent_ng/_tests/ -> agent_ng/ -> project root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEST_FILES = _REPO_ROOT / "experiments" / "test_files"

# ── test file locations (override with VL_TEST_*; defaults under experiments/test_files) ──
TEST_IMAGE = Path(
    os.getenv("VL_TEST_IMAGE", str(_TEST_FILES / "test_image.jpg"))
)
TEST_VIDEO = Path(
    os.getenv("VL_TEST_VIDEO", str(_TEST_FILES / "test_video.mp4"))
)
TEST_AUDIO = Path(
    os.getenv("VL_TEST_AUDIO", str(_TEST_FILES / "test_audio.mp3"))
)


def _has_openrouter_key() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


def _result(raw: str) -> dict:
    """Parse tool response JSON."""
    data = json.loads(raw)
    assert data["type"] == "tool_response", f"Unexpected response type: {data}"
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  Config / capability flag tests  (fast, no API calls)
# ══════════════════════════════════════════════════════════════════════════════

class TestVLCapabilityFlags:
    """Verify vision_support / video_support / audio_support flags in llm_configs."""

    def setup_method(self):
        os.environ["OPENROUTER_FETCH_PRICING_AT_STARTUP"] = "false"
        from agent_ng.llm_configs import get_default_llm_configs
        self.configs = get_default_llm_configs()

    def _all_models(self):
        for _prov, cfg in self.configs.items():
            yield from cfg.models

    def test_qwen3_6_plus_has_image_video(self):
        model = next(m for m in self._all_models() if m["model"] == "qwen/qwen3.6-plus")
        assert model.get("vision_support") is True, "Qwen 3.6 Plus should support images"
        assert model.get("video_support") is True,  "Qwen 3.6 Plus should support video"
        # Audio not supported by Qwen via OpenRouter
        assert not model.get("audio_support"), "Qwen 3.6 Plus does NOT support audio"

    def test_gemini_2_5_flash_has_all_modalities(self):
        # Both Direct and OpenRouter paths support audio when the client sends
        # input_audio (base64 + format); the OpenRouter app previously used
        # audio_url data URLs, which do not work reliably for transcription.
        gemini_models = [m for m in self._all_models()
                         if "gemini-2.5-flash" in m["model"]]
        assert len(gemini_models) >= 1, "gemini-2.5-flash must be in configs"
        for m in gemini_models:
            assert m.get("vision_support") is True
            assert m.get("video_support") is True
            assert m.get("audio_support") is True, f"{m['model']} should support audio"

    def test_mimo_v2_5_is_text_only_via_openrouter(self):
        model = next((m for m in self._all_models() if m["model"] == "xiaomi/mimo-v2.5"), None)
        assert model is not None, "mimo-v2.5 should be in OpenRouter config"
        assert model.get("vision_support") is False
        assert model.get("video_support") is False
        assert model.get("audio_support") is False

    def test_mimo_v2_omni_is_omni_on_openrouter(self):
        model = next((m for m in self._all_models() if m["model"] == "xiaomi/mimo-v2-omni"), None)
        assert model is not None, "mimo-v2-omni should be in OpenRouter config"
        assert model.get("vision_support") is True
        assert model.get("video_support") is True
        assert model.get("audio_support") is True

    def test_claude_has_image_only(self):
        claude_models = [m for m in self._all_models() if "claude-sonnet-4" in m["model"]]
        assert len(claude_models) >= 1
        for m in claude_models:
            assert m.get("vision_support") is True,   f"{m['model']} should support images"
            assert not m.get("video_support"),         f"{m['model']} does NOT support video"
            assert not m.get("audio_support"),         f"{m['model']} does NOT support audio"

    def test_grok_is_text_only_on_openrouter(self):
        # Grok via OpenRouter has no vision support; Polza's grok-4.20 may differ.
        from agent_ng.llm_manager import LLMProvider
        or_cfg = self.configs.get(LLMProvider.OPENROUTER)
        if or_cfg is None:
            pytest.skip("OpenRouter config not found")
        grok_models = [m for m in or_cfg.models if "grok" in m["model"]]
        for m in grok_models:
            assert not m.get("vision_support"), f"{m['model']} should be text-only on OpenRouter"
            assert not m.get("video_support")
            assert not m.get("audio_support")


class TestVisionToolManagerRouting:
    """Verify that media-type routing selects correct models."""

    def setup_method(self):
        os.environ["OPENROUTER_FETCH_PRICING_AT_STARTUP"] = "false"

    def test_audio_routes_to_gemini(self):
        from agent_ng.vision_tool_manager import VisionToolManager
        from agent_ng.vision_input import VisionInput, MediaType
        with patch.dict(
            os.environ,
            {"VL_AUDIO_MODEL": "gemini-2.5-flash"},
            clear=False,
        ):
            mgr = VisionToolManager()
            vi = VisionInput(prompt="transcribe", audio_url="http://example.com/x.mp3")
            model = mgr.get_model_for_input(vi)
        assert "gemini" in model.lower(), f"Audio should route to Gemini, got: {model}"

    def test_image_routes_to_default_model(self):
        from agent_ng.vision_tool_manager import VisionToolManager
        from agent_ng.vision_input import VisionInput
        mgr = VisionToolManager()
        vi = VisionInput(prompt="describe", image_url="http://example.com/x.jpg")
        model = mgr.get_model_for_input(vi)
        assert model == os.getenv("VL_DEFAULT_MODEL", "qwen/qwen3.6-plus")

    def test_video_routes_to_default_model(self):
        from agent_ng.vision_tool_manager import VisionToolManager
        from agent_ng.vision_input import VisionInput
        mgr = VisionToolManager()
        vi = VisionInput(prompt="describe", video_url="http://example.com/x.mp4")
        model = mgr.get_model_for_input(vi)
        assert model == os.getenv("VL_DEFAULT_MODEL", "qwen/qwen3.6-plus")

    def test_youtube_VL_YOUTUBE_GEMINI_PROVIDER_google_bypasses_openrouter_id(self):
        from agent_ng.vision_tool_manager import VisionToolManager
        from agent_ng.vision_input import VisionInput
        with patch.dict(
            os.environ,
            {
                "VL_YOUTUBE_MODEL": "gemini-2.5-flash",
                "VL_GEMINI_PROVIDER": "openrouter",
                "VL_YOUTUBE_GEMINI_PROVIDER": "google",
            },
            clear=False,
        ):
            mgr = VisionToolManager()
            vi = VisionInput(
                prompt="x",
                video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            )
            assert mgr.get_model_for_input(vi) == "gemini-2.5-flash"

    def test_youtube_uses_VL_GEMINI_PROVIDER_when_youtube_override_unset(self):
        from agent_ng.vision_tool_manager import VisionToolManager
        from agent_ng.vision_input import VisionInput
        with patch.dict(
            os.environ,
            {
                "VL_YOUTUBE_MODEL": "gemini-2.5-flash",
                "VL_GEMINI_PROVIDER": "openrouter",
                "VL_YOUTUBE_GEMINI_PROVIDER": "",
            },
            clear=False,
        ):
            mgr = VisionToolManager()
            vi = VisionInput(
                prompt="x",
                video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            )
            assert mgr.get_model_for_input(vi) == "google/gemini-2.5-flash"

    def test_adapter_openrouter_for_qwen(self):
        from agent_ng.vision_tool_manager import VisionToolManager
        with patch.dict(os.environ, {"AGENT_PROVIDER": "openrouter"}, clear=False):
            mgr = VisionToolManager()
            adapter = mgr.get_adapter_for_model("qwen/qwen3.6-plus")
        assert adapter is mgr.adapters.get("openrouter"), "Qwen should use OpenRouter adapter"

    def test_adapter_gemini_direct_for_bare_gemini(self):
        from agent_ng.vision_tool_manager import VisionToolManager
        mgr = VisionToolManager()
        adapter = mgr.get_adapter_for_model("gemini-2.5-flash")
        assert adapter is mgr.adapters.get("gemini"), "gemini-2.5-flash should use Gemini Direct adapter"

    def test_adapter_openrouter_for_google_prefixed(self):
        from agent_ng.vision_tool_manager import VisionToolManager
        with patch.dict(os.environ, {"AGENT_PROVIDER": "openrouter"}, clear=False):
            mgr = VisionToolManager()
            adapter = mgr.get_adapter_for_model("google/gemini-2.5-flash")
        assert adapter is mgr.adapters.get("openrouter"), "google/ prefix → OpenRouter adapter"

    # ------------------------------------------------------------------
    # Polza routing tests (TDD — require implementation below)
    # ------------------------------------------------------------------

    def test_polza_adapter_registered(self):
        """Polza adapter slot is always registered in VisionToolManager."""
        from agent_ng.vision_tool_manager import VisionToolManager
        mgr = VisionToolManager()
        assert "polza" in mgr.adapters, "polza adapter must be registered"

    def test_polza_adapter_has_polza_provider(self):
        """Polza adapter carries LLMProvider.POLZA, not OPENROUTER."""
        from agent_ng.vision_tool_manager import VisionToolManager
        from agent_ng.llm_manager import LLMProvider
        mgr = VisionToolManager()
        adapter = mgr.adapters["polza"]
        assert adapter.provider == LLMProvider.POLZA

    def test_adapter_uses_polza_when_agent_provider_polza(self):
        """AGENT_PROVIDER=polza routes slash-model to polza adapter."""
        from agent_ng.vision_tool_manager import VisionToolManager
        with patch.dict(os.environ, {"AGENT_PROVIDER": "polza"}, clear=False):
            mgr = VisionToolManager()
            adapter = mgr.get_adapter_for_model("qwen/qwen3.6-plus")
        assert adapter is mgr.adapters.get("polza"), (
            "AGENT_PROVIDER=polza: qwen/* should use polza adapter"
        )

    def test_adapter_uses_polza_for_google_prefixed_when_agent_provider_polza(self):
        """AGENT_PROVIDER=polza routes google/* to polza adapter."""
        from agent_ng.vision_tool_manager import VisionToolManager
        with patch.dict(os.environ, {"AGENT_PROVIDER": "polza"}, clear=False):
            mgr = VisionToolManager()
            adapter = mgr.get_adapter_for_model("google/gemini-2.5-flash")
        assert adapter is mgr.adapters.get("polza"), (
            "AGENT_PROVIDER=polza: google/* should use polza adapter"
        )

    def test_vl_gemini_provider_polza_normalizes_to_google_prefix(self):
        """VL_GEMINI_PROVIDER=polza normalizes gemini-* to google/gemini-*."""
        from agent_ng.vision_tool_manager import VisionToolManager
        with patch.dict(
            os.environ,
            {"VL_GEMINI_PROVIDER": "polza", "VL_AUDIO_MODEL": "gemini-2.5-flash"},
            clear=False,
        ):
            mgr = VisionToolManager()
            from agent_ng.vision_input import VisionInput
            vi = VisionInput(prompt="x", audio_url="http://example.com/x.mp3")
            model = mgr.get_model_for_input(vi)
        assert model == "google/gemini-2.5-flash", (
            f"VL_GEMINI_PROVIDER=polza should produce google/ prefix, got: {model}"
        )

    def test_default_openrouter_adapter_unaffected(self):
        """Regression: default AGENT_PROVIDER still routes to openrouter."""
        from agent_ng.vision_tool_manager import VisionToolManager
        with patch.dict(os.environ, {"AGENT_PROVIDER": "openrouter"}, clear=False):
            mgr = VisionToolManager()
            adapter = mgr.get_adapter_for_model("qwen/qwen3.6-plus")
        assert adapter is mgr.adapters.get("openrouter")


# ══════════════════════════════════════════════════════════════════════════════
#  Real API integration tests  (require OPENROUTER_API_KEY)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _has_openrouter_key(), reason="OPENROUTER_API_KEY not set")
class TestImageAnalysisAPI:
    """Real API calls for image analysis."""

    def setup_method(self):
        os.environ["OPENROUTER_FETCH_PRICING_AT_STARTUP"] = "false"

    @pytest.mark.skipif(not TEST_IMAGE.exists(), reason=f"Test image not found: {TEST_IMAGE}")
    def test_analyze_image_with_qwen(self):
        """Qwen 3.6 Plus analyzes a local image."""
        from agent_ng.vision_tool_manager import VisionToolManager
        from agent_ng.vision_input import VisionInput
        mgr = VisionToolManager()
        vi = VisionInput(prompt="Describe the shapes and colors in this image.", image_path=str(TEST_IMAGE))
        result = mgr.analyze(vi)
        assert isinstance(result, str) and len(result) > 20, f"Expected description, got: {result!r}"
        print(f"\n[IMAGE] Qwen response:\n{result[:300]}")

    @pytest.mark.skipif(not TEST_IMAGE.exists(), reason=f"Test image not found: {TEST_IMAGE}")
    def test_analyze_image_ai_tool(self):
        """analyze_image_ai tool wrapper works end-to-end."""
        from tools.tools import analyze_image_ai
        raw = analyze_image_ai.func(
            file_reference=str(TEST_IMAGE),
            prompt="What shapes do you see?",
            agent=None
        )
        data = _result(raw)
        assert data.get("result"), f"Expected result, got: {data}"
        print(f"\n[IMAGE TOOL] Response:\n{data['result'][:300]}")


@pytest.mark.skipif(not _has_openrouter_key(), reason="OPENROUTER_API_KEY not set")
class TestVideoAnalysisAPI:
    """Real API calls for video analysis using video_url format."""

    def setup_method(self):
        os.environ["OPENROUTER_FETCH_PRICING_AT_STARTUP"] = "false"

    @pytest.mark.skipif(not TEST_VIDEO.exists(), reason=f"Test video not found: {TEST_VIDEO}")
    def test_analyze_video_with_qwen(self):
        """Qwen 3.6 Plus analyzes a local MP4 video."""
        from agent_ng.vision_tool_manager import VisionToolManager
        mgr = VisionToolManager()
        result = mgr.analyze_video(video_path=str(TEST_VIDEO), prompt="Briefly describe what happens in this video.")
        assert isinstance(result, str) and len(result) > 20, f"Expected description, got: {result!r}"
        print(f"\n[VIDEO] Qwen response:\n{result[:300]}")

    @pytest.mark.skipif(not TEST_VIDEO.exists(), reason=f"Test video not found: {TEST_VIDEO}")
    def test_understand_video_tool(self):
        """understand_video tool wrapper works end-to-end."""
        from tools.tools import understand_video
        raw = understand_video.func(
            file_reference=str(TEST_VIDEO),
            prompt="What is happening in this video?",
            agent=None
        )
        data = _result(raw)
        assert data.get("result"), f"Expected result, got: {data}"
        print(f"\n[VIDEO TOOL] Response:\n{data['result'][:300]}")


@pytest.mark.skipif(not _has_openrouter_key(), reason="OPENROUTER_API_KEY not set")
class TestAudioAnalysisAPI:
    """Real API calls for audio analysis (Gemini 2.5 Flash via OpenRouter)."""

    def setup_method(self):
        os.environ["OPENROUTER_FETCH_PRICING_AT_STARTUP"] = "false"

    @pytest.mark.skipif(not TEST_AUDIO.exists(), reason=f"Test audio not found: {TEST_AUDIO}")
    def test_analyze_audio_with_gemini(self):
        """Gemini 2.5 Flash analyzes a local MP3 file."""
        from agent_ng.vision_tool_manager import VisionToolManager
        mgr = VisionToolManager()
        result = mgr.analyze_audio(audio_path=str(TEST_AUDIO), prompt="Transcribe or describe what you hear in this audio.")
        assert isinstance(result, str) and len(result) > 10, f"Expected transcription, got: {result!r}"
        print(f"\n[AUDIO] Gemini response:\n{result[:300]}")

    @pytest.mark.skipif(not TEST_AUDIO.exists(), reason=f"Test audio not found: {TEST_AUDIO}")
    def test_understand_audio_tool(self):
        """understand_audio tool wrapper works end-to-end."""
        from tools.tools import understand_audio
        raw = understand_audio.func(
            file_reference=str(TEST_AUDIO),
            prompt="What do you hear?",
            agent=None
        )
        data = _result(raw)
        assert data.get("result"), f"Expected result, got: {data}"
        print(f"\n[AUDIO TOOL] Response:\n{data['result'][:300]}")
