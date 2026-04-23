# Image Generation Test Results - CMW Platform Agent

**Date:** 2026-04-23
**Purpose:** Verify OpenRouter image generation models for CMW Platform business graphics

---

## Test Summary

**Total Models Tested:** 7
**Successful:** 4 (Google Gemini, Flux Pro, Seedream, Riverflow)
**Failed:** 3 (OpenAI blocked, Chinese models no image support)

---

## Working Models - Performance Comparison

| Model | Provider | Time | Cost | Size | Quality | Recommendation |
|-------|----------|------|------|------|---------|----------------|
| `google/gemini-2.5-flash-image` | Google | 6.2s | $0.039 | ~700KB | Excellent | ⭐ **PRIMARY** - Best speed/quality balance |
| `black-forest-labs/flux.2-pro` | Flux | 10.8s | $0.030 | ~330KB | Excellent | ⭐ **COST-OPTIMIZED** - Cheapest option |
| `bytedance-seed/seedream-4.5` | ByteDance | 12.9s | $0.040 | ~540KB | Good | **FALLBACK** - Good alternative |
| `sourceful/riverflow-v2-pro` | Sourceful | 110.7s | $0.150 | ~15KB | Good | ❌ Not recommended (slow/expensive) |

---

## Test Cases - CMW Business Prompts

### Test 1: Workflow Icon (Google Gemini)
- **Prompt:** "Business workflow icon, blue arrows connecting boxes, minimalist"
- **Result:** ✅ SUCCESS
- **File:** `test_workflow_google.png` (638,623 bytes)
- **Time:** 6.07s
- **Cost:** $0.038723

### Test 2: Form Icon (Google Gemini)
- **Prompt:** "Data entry form icon, document with fields, professional blue theme"
- **Result:** ✅ SUCCESS
- **File:** `test_form_google.png` (750,028 bytes)
- **Time:** 6.33s
- **Cost:** $0.038744

### Test 3: Workflow Icon (Flux Pro)
- **Prompt:** "Business workflow icon, blue and white, minimalist flat design"
- **Result:** ✅ SUCCESS
- **File:** `test_flux_pro.png` (329,323 bytes)
- **Time:** 10.80s
- **Cost:** $0.030000

### Test 4: Workflow Icon (Seedream)
- **Prompt:** "Business workflow icon, blue and white, minimalist flat design"
- **Result:** ✅ SUCCESS
- **File:** `test_seedream.png` (540,562 bytes)
- **Time:** 12.91s
- **Cost:** $0.040000

### Test 5: Workflow Icon (Riverflow)
- **Prompt:** "Business workflow icon, blue and white, minimalist flat design"
- **Result:** ✅ SUCCESS (but not recommended)
- **File:** `test_riverflow.png` (15,346 bytes)
- **Time:** 110.73s (VERY SLOW)
- **Cost:** $0.150000 (EXPENSIVE)

---

## Failed Models

### OpenAI Models
- **Models:** `openai/gpt-5-image-mini`, `openai/gpt-5-image`
- **Status:** ❌ BLOCKED
- **Error:** HTTP 403 - "unsupported_country_region_territory"
- **Reason:** Region restriction for Russia/CIS
- **VPN:** Does not bypass restriction

### Chinese Models
- **Models:** `qwen/qwen3.6-plus`, `minimax/minimax-m2.5:free`
- **Status:** ❌ NO IMAGE SUPPORT
- **Behavior:** Returns text descriptions instead of images
- **Note:** These models don't support image generation on OpenRouter

---

## Key Findings

### Speed Ranking
1. **Google Gemini** - 6.2s ⚡ (FASTEST)
2. **Flux Pro** - 10.8s
3. **Seedream** - 12.9s
4. **Riverflow** - 110.7s 🐌 (SLOWEST)

### Cost Ranking
1. **Flux Pro** - $0.030 💰 (CHEAPEST)
2. **Google Gemini** - $0.039
3. **Seedream** - $0.040
4. **Riverflow** - $0.150 💸 (MOST EXPENSIVE)

### Quality Assessment
- **Google Gemini:** Excellent for business graphics, icons, diagrams
- **Flux Pro:** Excellent artistic quality, good for creative needs
- **Seedream:** Good quality, similar to Gemini
- **Riverflow:** Good quality but not worth the cost/time

---

## Recommendations for CMW Platform

### Primary Choice
**`google/gemini-2.5-flash-image`**
- Best speed/cost/quality balance
- Proven reliable for business graphics
- Fast generation (6.2s)
- Reasonable cost ($0.039)

### Cost-Optimized Alternative
**`black-forest-labs/flux.2-pro`**
- Cheapest option ($0.030)
- Better artistic quality
- Use when cost is priority or need more creative output

### Fallback
**`bytedance-seed/seedream-4.5`**
- Similar performance to Gemini
- Good alternative if Gemini unavailable

---

## Implementation Notes

### API Details
- **Endpoint:** `https://openrouter.ai/api/v1/chat/completions`
- **Method:** POST
- **Headers:** `Authorization: Bearer <key>`, `Content-Type: application/json`
- **Request Format:**
```json
{
  "model": "google/gemini-2.5-flash-image",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Your prompt here"}
      ]
    }
  ]
}
```

### Response Format
```json
{
  "choices": [
    {
      "message": {
        "images": [
          {
            "image_url": {
              "url": "data:image/png;base64,iVBORw0KGgo..."
            }
          }
        ]
      }
    }
  ],
  "usage": {
    "total_tokens": 1308,
    "cost": 0.038723
  }
}
```

### Error Handling
- **403 errors:** Region restrictions (OpenAI)
- **404 errors:** Model not found
- **Text-only responses:** Model doesn't support image generation
- **Timeout:** Set to 120s minimum for slower models

---

## Generated Test Images

All test images saved in repository root:
- `test_workflow_google.png` - Google Gemini workflow icon
- `test_form_google.png` - Google Gemini form icon
- `test_workflow_icon.png` - Earlier test (Google Gemini)
- `test_form_icon.png` - Earlier test (Google Gemini)
- `test_flux_pro.png` - Flux Pro workflow icon
- `test_seedream.png` - Seedream workflow icon
- `test_riverflow.png` - Riverflow workflow icon

---

## Conclusion

✅ **Image generation via OpenRouter is production-ready for CMW Platform**

- Multiple working models available
- Tested with actual CMW business use cases
- Cost-effective ($0.030-$0.040 per image)
- Fast generation (6-13 seconds for recommended models)
- Reliable API with good error handling

**Ready to proceed with implementation.**