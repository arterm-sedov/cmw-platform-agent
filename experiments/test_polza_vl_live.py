"""Live Polza VL tests — image, video, audio, YouTube.

Run:  python experiments/test_polza_vl_live.py

Requires POLZA_API_KEY in .env.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Force all VL through Polza for this test run
os.environ["AGENT_PROVIDER"] = "polza"
os.environ["OPENROUTER_FETCH_PRICING_AT_STARTUP"] = "false"

_ROOT = Path(__file__).resolve().parents[1]
_FILES = _ROOT / "experiments" / "test_files"

sys.path.insert(0, str(_ROOT))


def _header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def _ok(label: str, text: str) -> None:
    print(f"  ✓ {label}: {text[:200]}")


def _fail(label: str, err: Exception) -> None:
    print(f"  ✗ {label}: {err}")


def test_image() -> None:
    _header("IMAGE — qwen/qwen3.6-plus via Polza")
    from agent_ng.vision_tool_manager import VisionToolManager
    from agent_ng.vision_input import VisionInput

    mgr = VisionToolManager()
    adapter = mgr.get_adapter_for_model(mgr.vl_model)
    print(f"  model  : {mgr.vl_model}")
    print(f"  adapter: {adapter.provider.value if adapter else 'none'}")

    vi = VisionInput(prompt="Describe this image in one sentence.", image_path=str(_FILES / "test_image.jpg"))
    try:
        result = mgr.analyze(vi)
        _ok("response", result)
    except Exception as e:
        _fail("error", e)


def test_video() -> None:
    _header("VIDEO — qwen/qwen3.6-plus via Polza")
    from agent_ng.vision_tool_manager import VisionToolManager

    mgr = VisionToolManager()
    try:
        result = mgr.analyze_video(
            video_path=str(_FILES / "test_video.mp4"),
            prompt="Briefly describe what happens in this video.",
        )
        _ok("response", result)
    except Exception as e:
        _fail("error", e)


def test_audio() -> None:
    _header("AUDIO — xiaomi/mimo-v2-omni via Polza")
    from agent_ng.vision_tool_manager import VisionToolManager

    mgr = VisionToolManager()
    audio_model = mgr.get_model_for_input(
        __import__("agent_ng.vision_input", fromlist=["VisionInput"]).VisionInput(
            prompt="x", audio_url="http://example.com/x.mp3"
        )
    )
    adapter = mgr.get_adapter_for_model(audio_model)
    print(f"  model  : {audio_model}")
    print(f"  adapter: {adapter.provider.value if adapter else 'none'}")
    try:
        result = mgr.analyze_audio(
            audio_path=str(_FILES / "test_audio.mp3"),
            prompt="What do you hear? Give a one-sentence description.",
        )
        _ok("response", result)
    except Exception as e:
        _fail("error", e)


def test_youtube_gemini_direct() -> None:
    """YouTube should still route through Gemini Direct (VL_YOUTUBE_GEMINI_PROVIDER=google)."""
    _header("YOUTUBE — Gemini Direct (VL_YOUTUBE_GEMINI_PROVIDER=google)")
    os.environ["VL_YOUTUBE_MODEL"] = "gemini-2.5-flash"
    os.environ["VL_YOUTUBE_GEMINI_PROVIDER"] = "google"

    from agent_ng.vision_tool_manager import VisionToolManager
    from agent_ng.vision_input import VisionInput

    mgr = VisionToolManager()
    yt_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    vi = VisionInput(prompt="What song is this? One sentence.", video_url=yt_url)
    model = mgr.get_model_for_input(vi)
    adapter = mgr.get_adapter_for_model(model)
    print(f"  model  : {model}")
    print(f"  adapter: {adapter.provider.value if adapter else 'none'}")
    try:
        result = mgr.analyze(vi)
        _ok("response", result)
    except Exception as e:
        _fail("error", e)


def test_youtube_via_polza() -> None:
    """Test YouTube routed through Polza (google/gemini-2.5-flash on Polza)."""
    _header("YOUTUBE — google/gemini-2.5-flash via Polza (experimental)")
    os.environ["VL_YOUTUBE_MODEL"] = "gemini-2.5-flash"
    os.environ["VL_YOUTUBE_GEMINI_PROVIDER"] = ""  # falls through to VL_GEMINI_PROVIDER=polza

    from agent_ng.vision_tool_manager import VisionToolManager
    from agent_ng.vision_input import VisionInput

    # Re-instantiate to pick up the env change
    mgr = VisionToolManager()
    yt_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    vi = VisionInput(prompt="What song is this? One sentence.", video_url=yt_url)
    model = mgr.get_model_for_input(vi)
    adapter = mgr.get_adapter_for_model(model)
    print(f"  model  : {model}")
    print(f"  adapter: {adapter.provider.value if adapter else 'none'}")
    try:
        result = mgr.analyze(vi)
        _ok("response", result)
    except Exception as e:
        _fail("error", e)


if __name__ == "__main__":
    polza_key = os.getenv("POLZA_API_KEY", "")
    if not polza_key:
        print("POLZA_API_KEY not set — aborting")
        sys.exit(1)

    print(f"POLZA_API_KEY: {polza_key[:8]}...")

    test_image()
    test_video()
    test_audio()
    test_youtube_gemini_direct()
    test_youtube_via_polza()

    print("\n\nDone.")
