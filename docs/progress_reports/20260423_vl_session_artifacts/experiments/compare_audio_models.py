"""
Compare Gemini vs Mimo for audio analysis via OpenRouter.
Tests both models with the same audio file and prompt.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

# Test file
TEST_AUDIO = Path(r"E:\Downloads\1mb.mp3")

def test_audio_model(model_name: str, audio_path: Path, prompt: str):
    """Test a specific model for audio analysis."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")

    try:
        from agent_ng.vision_tool_manager import VisionToolManager
        from agent_ng.vision_input import VisionInput

        # Disable pricing fetch
        os.environ['OPENROUTER_FETCH_PRICING_AT_STARTUP'] = 'false'

        # Create manager
        mgr = VisionToolManager()

        # Create input
        vi = VisionInput(prompt=prompt, audio_path=str(audio_path))

        # Analyze with specific model
        print(f"Sending request to {model_name}...")
        result = mgr.analyze(vi, model=model_name)

        print(f"\n✅ SUCCESS")
        print(f"Response length: {len(result)} chars")
        print(f"\nFirst 500 chars:")
        print("-" * 70)
        print(result[:500])
        print("-" * 70)

        return True, result

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return False, str(e)


def main():
    if not TEST_AUDIO.exists():
        print(f"❌ Test audio not found: {TEST_AUDIO}")
        return

    print(f"Audio file: {TEST_AUDIO}")
    print(f"Size: {TEST_AUDIO.stat().st_size / 1024:.1f} KB")

    prompt = "Transcribe or describe what you hear in this audio file."

    # Test both models
    models = [
        "google/gemini-2.5-flash",
        "xiaomi/mimo-v2.5"
    ]

    results = {}
    for model in models:
        success, output = test_audio_model(model, TEST_AUDIO, prompt)
        results[model] = {"success": success, "output": output}

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for model, data in results.items():
        status = "✅ PASS" if data["success"] else "❌ FAIL"
        print(f"{model:40} | {status}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
