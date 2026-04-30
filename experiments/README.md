# Vision-Language Model Experiments

This directory contains experiments to test VL model capabilities before implementation.

## Directory Structure

```
experiments/
├── test_vl_models.py          # Main testing script
├── test_files/                # Test files for VL models
│   ├── test_image.jpg         # General image for description
│   ├── test_document.jpg      # Image with text for OCR testing
│   ├── test_chart.jpg         # Chart/graph for data extraction
│   ├── test_video.mp4         # (Optional) Short video clip
│   └── test_audio.mp3         # (Optional) Audio file
├── vl_test_results.json       # Test results (generated)
└── README.md                  # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install requests
```

### 2. Set Environment Variable

```bash
# Windows PowerShell
$env:OPENROUTER_API_KEY="your_key_here"

# Windows CMD
set OPENROUTER_API_KEY=your_key_here

# Linux/Mac
export OPENROUTER_API_KEY=your_key_here
```

### 3. Prepare Test Files

Add test files to `experiments/test_files/`:

**Required:**
- `test_image.jpg` - Any image (photo, screenshot, etc.)
- `test_document.jpg` - Image containing text (document scan, screenshot with text)
- `test_chart.jpg` - Image with chart, graph, or data visualization

**Optional:**
- `test_video.mp4` - Short video clip (10-30 seconds)
- `test_audio.mp3` - Audio file with speech

**Suggestions for test files:**
- Use real-world examples from your use case
- Include challenging cases (low quality, rotated text, complex charts)
- Keep files small (<5MB) for faster testing

## Running Tests

### Basic Usage

```bash
# From repository root
python experiments/test_vl_models.py
```

### Expected Output

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                   Vision-Language Model Testing Script                      ║
║                                                                              ║
║  This script tests VL capabilities of models via OpenRouter API             ║
╚══════════════════════════════════════════════════════════════════════════════╝

🚀 Starting tests...
Total test cases: 8

[Test 1/8]
================================================================================
Testing: qwen/qwen3.6-plus
File: experiments/test_files/test_image.jpg
Type: image
Prompt: Describe this image in detail.
================================================================================
Encoding file to base64...
Sending request to OpenRouter...

✅ SUCCESS (latency: 2341ms)

Response:
The image shows a modern office workspace with...

[Test 2/8]
...
```

## Test Results

Results are saved to `experiments/vl_test_results.json` with the following structure:

```json
[
  {
    "model": "qwen/qwen3.6-plus",
    "file_path": "experiments/test_files/test_image.jpg",
    "file_type": "image",
    "prompt": "Describe this image in detail.",
    "timestamp": 1713862167.123,
    "success": true,
    "response": "The image shows...",
    "error": null,
    "latency_ms": 2341,
    "usage": {
      "prompt_tokens": 1234,
      "completion_tokens": 567,
      "total_tokens": 1801
    }
  }
]
```

## Models Being Tested

### Priority 1: Qwen 3.6 Plus
- **Model ID:** `qwen/qwen3.6-plus`
- **Claims:** Text, image, video support
- **Cost:** $0.325/$1.95 per 1M tokens
- **Context:** 1M tokens

### Priority 2: Gemini 3.1 Flash
- **Model ID:** `google/gemini-3.1-flash`
- **Claims:** Text, image, video, audio, PDF support
- **Cost:** TBD
- **Context:** 1M+ tokens

### Priority 3: Gemini 2.0 Flash (Fallback)
- **Model ID:** `google/gemini-2.0-flash-exp`
- **Claims:** Text, image, video, audio support
- **Cost:** TBD
- **Context:** 1M tokens

### Priority 4: Xiaomi Mimo v2.5
- **Model ID:** `xiaomi/mimo-v2.5`
- **Claims:** Multimodal support
- **Cost:** TBD
- **Context:** TBD

## Evaluation Criteria

### Image Description
- **Quality:** How detailed and accurate is the description?
- **Latency:** Response time in milliseconds
- **Cost:** Tokens used per request

### OCR (Text Extraction)
- **Accuracy:** Compare with Tesseract OCR
- **Handling:** Rotated text, low quality, multiple languages
- **Format:** Structured output vs raw text

### Chart Analysis
- **Data Extraction:** Can it extract numerical data?
- **Understanding:** Does it understand chart type and context?
- **Structured Output:** Can it return data in JSON/table format?

### Video Analysis (Optional)
- **Temporal Understanding:** Can it describe events over time?
- **Scene Detection:** Can it identify scene changes?
- **Object Tracking:** Can it track objects across frames?

### Audio Analysis (Optional)
- **Transcription:** Accuracy of speech-to-text
- **Speaker Detection:** Can it identify multiple speakers?
- **Language Support:** Multiple languages

## Comparison Matrix Template

After running tests, fill in this matrix:

| Model | Image | Video | Audio | PDF | OCR Quality | Speed | Cost/Request | Notes |
|-------|-------|-------|-------|-----|-------------|-------|--------------|-------|
| Qwen 3.6 Plus | ✅ | ? | ? | ? | ?/10 | ?ms | $? | |
| Gemini 3.1 Flash | ✅ | ? | ? | ? | ?/10 | ?ms | $? | |
| Gemini 2.0 Flash | ✅ | ? | ? | ? | ?/10 | ?ms | $? | |
| Mimo v2.5 | ✅ | ? | ? | ? | ?/10 | ?ms | $? | |

**Legend:**
- ✅ Works
- ❌ Doesn't work
- ⚠️ Partial support
- ? Not tested yet

## Next Steps

1. **Review Results:** Analyze `vl_test_results.json`
2. **Compare Quality:** Test OCR against Tesseract
3. **Update Plan:** Update `C:/Users/ased/.opencode/plans/VL_MODEL_SUPPORT_IMPLEMENTATION.md` with findings
4. **Make Recommendations:** Choose default models based on results
5. **Proceed to Phase 1:** Begin implementation

## Troubleshooting

### Error: "OPENROUTER_API_KEY environment variable not set"
**Solution:** Set the environment variable before running the script

### Error: "File not found"
**Solution:** Add test files to `experiments/test_files/` directory

### Error: "HTTP 401: Unauthorized"
**Solution:** Check your OpenRouter API key is valid

### Error: "HTTP 429: Too Many Requests"
**Solution:** Wait a few minutes and try again, or reduce test frequency

### Error: "HTTP 400: Bad Request"
**Solution:** Model may not support the requested modality (e.g., audio)

## Cost Estimation

**Approximate costs per test:**
- Image description: ~1K tokens = $0.001-0.003
- OCR extraction: ~2K tokens = $0.002-0.006
- Chart analysis: ~2K tokens = $0.002-0.006

**Total for full test suite (8 tests):** ~$0.02-0.05

## Notes

- Test results may vary based on file content and complexity
- Latency depends on file size and network conditions
- Some models may not support all modalities
- OpenRouter may route to different providers for the same model ID

## References

- OpenRouter Models: https://openrouter.ai/models
- OpenRouter Docs: https://openrouter.ai/docs
- VL Support Plan: `C:/Users/ased/.opencode/plans/VL_MODEL_SUPPORT_IMPLEMENTATION.md`
