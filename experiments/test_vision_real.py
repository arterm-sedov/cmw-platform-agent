"""
Real-world integration test for VisionToolManager with OpenRouter API
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_ng.vision_input import VisionInput
from agent_ng.vision_tool_manager import VisionToolManager

def test_real_image_analysis():
    """Test real image analysis with OpenRouter"""
    print("="*80)
    print("Real-World VisionToolManager Test")
    print("="*80)

    # Check test image exists
    test_image = Path("experiments/test_files/test_image.jpg")
    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        return False

    print(f"\n✅ Test image found: {test_image}")

    # Create VisionInput
    print("\n[Step 1] Creating VisionInput...")
    try:
        vision_input = VisionInput(
            prompt="Describe this image in detail. What shapes and colors do you see?",
            image_path=str(test_image)
        )
        print(f"✅ VisionInput created")
        print(f"   Media type: {vision_input.media_type}")
        print(f"   Has media: {vision_input.has_media()}")

        # Validate
        vision_input.validate()
        print(f"✅ Validation passed")
    except Exception as e:
        print(f"❌ Failed to create VisionInput: {e}")
        return False

    # Initialize VisionToolManager
    print("\n[Step 2] Initializing VisionToolManager...")
    try:
        # Disable OpenRouter pricing fetch to avoid timeout
        os.environ['OPENROUTER_FETCH_PRICING_AT_STARTUP'] = 'false'

        manager = VisionToolManager()
        print(f"✅ VisionToolManager initialized")
        print(f"   Adapters: {list(manager.adapters.keys())}")
        print(f"   Default model: {manager.default_model}")
        print(f"   Fast model: {manager.fast_model}")
    except Exception as e:
        print(f"❌ Failed to initialize VisionToolManager: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Analyze image (this will make real API call)
    print("\n[Step 3] Analyzing image with OpenRouter...")
    print("   Model: google/gemini-2.5-flash (fast)")
    print("   This will make a REAL API call...")

    try:
        result = manager.analyze(vision_input, prefer_fast=True)
        print(f"\n✅ SUCCESS! Got response from model")
        print(f"\nResponse:\n{result[:500]}...")
        return True
    except Exception as e:
        print(f"\n❌ Failed to analyze image: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_image_analysis()
    sys.exit(0 if success else 1)
