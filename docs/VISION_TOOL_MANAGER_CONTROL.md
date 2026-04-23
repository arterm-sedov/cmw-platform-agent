# VisionToolManager Control - Configuration Guide

## Environment Variable: USE_VISION_TOOL_MANAGER

Controls whether tools use the new VisionToolManager or fall back to legacy implementations.

---

## Configuration

### Enable VisionToolManager (Default)
```bash
# In .env file
USE_VISION_TOOL_MANAGER=true
```

**What happens:**
- ✅ `analyze_image_ai()` uses VisionToolManager (Qwen/Gemini via OpenRouter)
- ✅ `understand_video()` tries VisionToolManager first, falls back to legacy Gemini
- ✅ `understand_audio()` tries VisionToolManager first, falls back to legacy Gemini
- 🎁 Better models, lower cost (-72% to -92%), faster responses

---

### Disable VisionToolManager
```bash
# In .env file
USE_VISION_TOOL_MANAGER=false
```

**What happens:**
- ❌ `analyze_image_ai()` returns error (requires VisionToolManager)
- ✅ `understand_video()` uses legacy Gemini Direct API
- ✅ `understand_audio()` uses legacy Gemini Direct API
- 📊 Original behavior, direct Gemini API calls

---

## Why Would You Disable It?

### 1. **Troubleshooting**
If VisionToolManager has issues, disable it to use proven legacy path:
```bash
USE_VISION_TOOL_MANAGER=false
```

### 2. **Cost Control**
If you want to use only Gemini Direct API (with your own key):
```bash
USE_VISION_TOOL_MANAGER=false
GEMINI_KEY=your_gemini_key
```

### 3. **Testing**
Compare old vs new behavior:
```bash
# Test with new
USE_VISION_TOOL_MANAGER=true

# Test with old
USE_VISION_TOOL_MANAGER=false
```

### 4. **Gradual Rollout**
Enable for some environments, disable for others:
```bash
# Production: use new
USE_VISION_TOOL_MANAGER=true

# Staging: use old for comparison
USE_VISION_TOOL_MANAGER=false
```

---

## Behavior Matrix

| Tool | USE_VISION_TOOL_MANAGER=true | USE_VISION_TOOL_MANAGER=false |
|------|------------------------------|-------------------------------|
| **analyze_image_ai** | Uses VisionToolManager (Qwen/Gemini) | Returns error (not available) |
| **understand_video** | Tries VisionToolManager → Falls back to Gemini | Uses legacy Gemini directly |
| **understand_audio** | Tries VisionToolManager → Falls back to Gemini | Uses legacy Gemini directly |
| **analyze_image** (legacy) | Not affected (always uses PIL) | Not affected (always uses PIL) |

---

## Fallback Behavior

### understand_video & understand_audio

Even with `USE_VISION_TOOL_MANAGER=true`, these tools have **automatic fallback**:

```python
if use_vision_manager:
    try:
        # Try VisionToolManager
        return result_from_vision_manager
    except Exception:
        # Fall back to legacy Gemini
        pass

# Legacy Gemini implementation (always available)
return result_from_gemini
```

**Fallback triggers:**
- VisionToolManager import fails
- API call fails
- Invalid input
- Any runtime error

**Result:** Tools always work, even if VisionToolManager fails

---

## analyze_image_ai Special Case

This is a **NEW** tool that requires VisionToolManager:

```python
if not use_vision_manager:
    return error("VisionToolManager is disabled")
```

**Why no fallback?**
- It's a new tool specifically for AI-powered analysis
- Legacy `analyze_image()` exists for basic metadata
- Users should use `analyze_image()` if they want non-AI analysis

---

## Recommendations

### For Production
```bash
USE_VISION_TOOL_MANAGER=true
```
✅ Better models, lower cost, faster

### For Troubleshooting
```bash
USE_VISION_TOOL_MANAGER=false
```
✅ Proven legacy path, simpler debugging

### For Testing
```bash
# Test both paths
USE_VISION_TOOL_MANAGER=true   # Test new
USE_VISION_TOOL_MANAGER=false  # Test old
```

---

## Summary

**In this monolithic repo:**
- VisionToolManager is always available (same repo)
- `USE_VISION_TOOL_MANAGER` controls whether to use it
- Default: `true` (use VisionToolManager)
- Fallback: automatic for understand_video/audio
- No fallback: analyze_image_ai (new tool, requires VisionToolManager)

**Backward compatibility:**
- ✅ All existing code works
- ✅ Same function signatures
- ✅ Same return formats
- ✅ Can disable new behavior with env var
- ✅ Automatic fallback for video/audio tools
