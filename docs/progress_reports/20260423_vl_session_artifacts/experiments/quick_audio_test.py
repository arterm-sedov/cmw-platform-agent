import os
import sys
from pathlib import Path

# Suppress logs
os.environ['LOG_LEVEL'] = 'CRITICAL'
os.environ['OPENROUTER_FETCH_PRICING_AT_STARTUP'] = 'false'

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agent_ng.vision_tool_manager import VisionToolManager
from agent_ng.vision_input import VisionInput

TEST_AUDIO = Path(r'E:\Downloads\1mb.mp3')

print("Testing Audio Models...")
print("="*70)

# Test Gemini
print("\n1. GEMINI (google/gemini-2.5-flash):")
try:
    mgr = VisionToolManager()
    vi = VisionInput(prompt='Transcribe what you hear.', audio_path=str(TEST_AUDIO))
    result = mgr.analyze(vi, model='google/gemini-2.5-flash')
    print(f"   Length: {len(result)} chars")
    print(f"   Response: {result[:250]}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test Mimo
print("\n2. MIMO (xiaomi/mimo-v2.5):")
try:
    mgr = VisionToolManager()
    vi = VisionInput(prompt='Transcribe what you hear.', audio_path=str(TEST_AUDIO))
    result = mgr.analyze(vi, model='xiaomi/mimo-v2.5')
    print(f"   Length: {len(result)} chars")
    print(f"   Response: {result[:250]}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "="*70)
print("Done!")
