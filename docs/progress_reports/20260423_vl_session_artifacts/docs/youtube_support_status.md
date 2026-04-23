# YouTube URL Support - Implementation & Findings

**Date:** 2026-04-23 18:26 UTC  
**Status:** ⚠️ Implemented but blocked by API timeout

---

## ✅ Implementation Complete

### 1. **Updated Routing Logic**
```python
# VisionToolManager.get_model_for_input()
if media_type == VIDEO and is_youtube_url(video_url):
    return "gemini-2.5-flash"  # Use Gemini Direct for YouTube
```

### 2. **Updated Gemini Adapter**
- Added YouTube URL detection in `format_input()`
- Added YouTube URL handling in `invoke()`
- YouTube URLs passed directly to Gemini (no download/upload)

**Files modified:**
- `agent_ng/vision_tool_manager.py` — YouTube routing
- `agent_ng/vision_adapters/gemini_adapter.py` — YouTube support

---

## 🔴 Current Blocker: API Timeout

### Test Results

**YouTube URL:** `https://www.youtube.com/watch?v=v3Fr2JR47KA`

**Symptoms:**
- Gemini API call hangs indefinitely
- No response after 90+ seconds
- Both via adapter and direct API call

**Possible causes:**
1. **Network/Firewall:** Blocking Gemini API endpoints
2. **API Format:** YouTube URL format incorrect for Gemini
3. **Rate Limiting:** API key throttled or quota exceeded
4. **Regional Restrictions:** YouTube content not accessible from API location

---

## 📊 Capability Matrix (Updated)

| Provider | Image | Video (File) | Video (YouTube) | Audio |
|----------|-------|--------------|-----------------|-------|
| **Qwen (OpenRouter)** | ✅ | ✅ | ❌ | ❌ |
| **Gemini (OpenRouter)** | ✅ | ✅ | ❌ | ❌ |
| **Gemini (Direct API)** | ✅ | ✅ | ✅* | ✅* |

*Implemented but blocked by timeout issue

---

## 🎯 Routing Logic (Final)

```python
def get_model_for_input(vision_input):
    # Audio → Gemini Direct
    if media_type == AUDIO:
        return "gemini-2.5-flash"
    
    # YouTube URLs → Gemini Direct
    if media_type == VIDEO and is_youtube_url(video_url):
        return "gemini-2.5-flash"
    
    # Everything else → Qwen via OpenRouter
    return "qwen/qwen3.6-plus"
```

---

## 🧪 Testing Recommendations

### 1. **Verify Gemini API Access**
```bash
curl -X POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: $GEMINI_KEY" \
  -d '{"contents":[{"parts":[{"text":"test"}]}]}'
```

### 2. **Test with Different YouTube URL**
Try a shorter, public video to rule out content restrictions.

### 3. **Check API Quotas**
Verify Gemini API quota hasn't been exceeded in Google Cloud Console.

### 4. **Test from Different Network**
Rule out firewall/proxy blocking Gemini endpoints.

---

## 📝 Code Changes Summary

### `agent_ng/vision_tool_manager.py`
```python
def get_model_for_input(self, vision_input: VisionInput) -> str:
    # Audio always uses Gemini Direct
    if vision_input.media_type == MediaType.AUDIO:
        return self.vl_audio_model
    
    # YouTube URLs use Gemini Direct (native support)
    if vision_input.media_type == MediaType.VIDEO:
        video_url = vision_input.video_url or vision_input.get_media_url()
        if video_url and ('youtube.com' in video_url or 'youtu.be' in video_url):
            return self.vl_audio_model  # Gemini Direct
    
    # Everything else uses default (Qwen via OpenRouter)
    return self.vl_model
```

### `agent_ng/vision_adapters/gemini_adapter.py`
```python
def format_input(self, vision_input: VisionInput) -> Dict[str, Any]:
    # Check for YouTube URL (Gemini supports direct YouTube URLs)
    if vision_input.video_url:
        video_url = vision_input.video_url
        if 'youtube.com' in video_url or 'youtu.be' in video_url:
            return {
                "youtube_url": video_url,
                "prompt": vision_input.prompt,
                "media_type": vision_input.media_type
            }
    # ... rest of file handling

def invoke(self, vision_input: VisionInput, model: Optional[str] = None) -> str:
    # Handle YouTube URLs directly (no upload needed)
    if "youtube_url" in formatted:
        youtube_url = formatted["youtube_url"]
        contents = self.types.Content(
            parts=[
                self.types.Part(file_data=self.types.FileData(file_uri=youtube_url)),
                self.types.Part(text=prompt)
            ]
        )
        response = client.models.generate_content(model=model, contents=contents)
        return response.text
    # ... rest of file upload handling
```

---

## 🔧 Workaround (Until API Issue Resolved)

**Option 1:** Download YouTube video first, then analyze
```python
# Use yt-dlp or similar to download
# Then pass file path instead of URL
```

**Option 2:** Use OpenRouter with downloaded video
```python
# Download video → Upload to OpenRouter → Analyze with Qwen
```

**Option 3:** Use different video source
```python
# Direct video URLs (not YouTube) work via OpenRouter
```

---

## ✅ What's Working

1. ✅ YouTube URL detection
2. ✅ Routing to Gemini Direct
3. ✅ Adapter selection logic
4. ✅ Code structure for YouTube support

## ⚠️ What's Blocked

1. ⚠️ Gemini API call (timeout)
2. ⚠️ Live YouTube URL test
3. ⚠️ End-to-end verification

---

## 📌 Next Steps

1. **Debug Gemini API timeout** (network/auth/quota)
2. **Test with alternative video sources**
3. **Implement download fallback** for YouTube URLs
4. **Add timeout handling** and error messages
5. **Document limitations** in user-facing docs

---

**Status:** Code ready, waiting for API access resolution  
**Estimated effort to resolve:** 30-60 minutes (once API access confirmed)
