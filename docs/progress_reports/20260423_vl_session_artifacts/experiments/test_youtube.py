"""Quick test for YouTube URL with Gemini Direct"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ['LOG_LEVEL'] = 'ERROR'
os.environ['OPENROUTER_FETCH_PRICING_AT_STARTUP'] = 'false'

from dotenv import load_dotenv
load_dotenv()

from agent_ng.vision_input import VisionInput
from agent_ng.vision_tool_manager import VisionToolManager

youtube_url = 'https://www.youtube.com/watch?v=v3Fr2JR47KA'

print("Testing YouTube URL with Gemini Direct")
print("="*70)

# Create input
vi = VisionInput(prompt='Describe this video briefly.', video_url=youtube_url)
print(f"Video URL: {vi.video_url}")
print(f"Media type: {vi.media_type}")

# Get manager
mgr = VisionToolManager()
selected_model = mgr.get_model_for_input(vi)
print(f"Selected model: {selected_model}")

adapter = mgr.get_adapter_for_model(selected_model)
print(f"Adapter: {adapter.__class__.__name__}")
print(f"Adapter available: {adapter.available if hasattr(adapter, 'available') else 'N/A'}")

print("\nCalling analyze...")
try:
    result = mgr.analyze(vi)
    print(f"\n✅ SUCCESS")
    print(f"Length: {len(result)} chars")
    print(f"\nResponse:\n{result[:500]}")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
