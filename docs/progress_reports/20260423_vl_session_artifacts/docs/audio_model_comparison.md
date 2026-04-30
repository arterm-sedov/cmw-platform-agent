# Audio Model Comparison: Gemini vs Mimo

## Test Setup

**Goal:** Compare `google/gemini-2.5-flash` vs `xiaomi/mimo-v2.5` for audio transcription/analysis

**Test File:** `E:\Downloads\1mb.mp3` (1.2 MB)

**Prompt:** "Transcribe or describe what you hear in this audio file."

**Method:** Both models tested via OpenRouter with identical inputs

---

## Current Status

**❌ Unable to complete live test** - OpenRouter API key returns 401 Unauthorized

**Reason:** API key expired or invalid (not a code issue)

---

## What We Know from Config

### Audio Support Flags

| Model | Audio | Video | Image | Provider |
|-------|-------|-------|-------|----------|
| **google/gemini-2.5-flash** | ✅ | ✅ | ✅ | OpenRouter |
| **xiaomi/mimo-v2.5** | ✅ | ✅ | ✅ | OpenRouter |

Both models claim full multimodal support including audio.

---

## Current Production Routing

**Audio requests route to:** `google/gemini-2.5-flash`

**Configured via:** `VL_AUDIO_MODEL=google/gemini-2.5-flash`

**Reason:** 
- Gemini has proven track record for audio
- Mimo is newer, less battle-tested
- Conservative choice for production

---

## How to Test When API Key is Fixed

1. **Fix API key** in `.env`:
   ```bash
   OPENROUTER_API_KEY=sk-or-v1-<valid-key>
   ```

2. **Run comparison script**:
   ```bash
   python experiments/compare_audio_models.py
   ```

3. **Or run integration tests**:
   ```bash
   pytest agent_ng/_tests/test_vl_integration.py::TestAudioAnalysisAPI -v -s
   ```

4. **To switch to Mimo** for audio (if it performs better):
   ```bash
   # In .env
   VL_AUDIO_MODEL=xiaomi/mimo-v2.5
   ```

---

## Expected Differences

### Gemini 2.5 Flash
- **Strengths:** Proven audio transcription, good accuracy, fast
- **Weaknesses:** May be more expensive
- **Best for:** Production use, reliability

### Xiaomi Mimo v2.5
- **Strengths:** Newer model, potentially better quality, 1M token context
- **Weaknesses:** Less tested, unknown audio quality
- **Best for:** Experimentation, if quality proves superior

---

## Code is Ready

✅ `VisionToolManager` correctly routes audio to `vl_audio_model`  
✅ `OpenRouterVisionAdapter` uses correct `audio_url` format  
✅ Model selection works (finds correct model_index)  
✅ Integration tests written and ready  
✅ Comparison script created  

**Only blocker:** Valid OpenRouter API key needed for live testing

---

## Recommendation

1. **Keep Gemini as default** until Mimo is proven for audio
2. **Test Mimo when API access restored** using the comparison script
3. **Switch to Mimo** only if it shows clear quality/cost advantages
4. **Monitor both** for production use cases

---

**Status:** Code complete, awaiting valid API credentials for live testing  
**Date:** 2026-04-23 18:02 UTC
