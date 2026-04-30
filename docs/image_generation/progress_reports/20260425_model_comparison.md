# Image model comparison

Run date: 2026-04-25T15:33:28.104634

| model | prompt | ok | time(s) | cost($) | size(KB) | dims | file |
|-------|--------|----|---------|---------|----------|------|------|
| `google/gemini-2.5-flash-image` | business | ✓ | 5.7 | 0.038350026 | 224.2 | 1024x1024 | docs\image_generation\progress_reports\20260425_google__gemini-2.5-flash-image_business.png |
| `google/gemini-2.5-flash-image` | russian | ✓ | 7.83 | 0.038351412 | 395.9 | 1024x1024 | docs\image_generation\progress_reports\20260425_google__gemini-2.5-flash-image_russian.png |
| `google/gemini-3.1-flash-image-preview` | business | ✓ | 15.41 | 0.066544335 | 553.0 | 1408x768 | docs\image_generation\progress_reports\20260425_google__gemini-3.1-flash-image-preview_business.png |
| `google/gemini-3.1-flash-image-preview` | russian | ✓ | 15.37 | 0.06655077 | 1294.0 | 1408x768 | docs\image_generation\progress_reports\20260425_google__gemini-3.1-flash-image-preview_russian.png |
| `google/gemini-3-pro-image-preview` | business | ✓ | 45.21 | 0.13464198 | 743.2 | 1408x768 | docs\image_generation\progress_reports\20260425_google__gemini-3-pro-image-preview_business.png |
| `google/gemini-3-pro-image-preview` | russian | ✓ | 40.41 | 0.13643784 | 2115.6 | 1408x768 | docs\image_generation\progress_reports\20260425_google__gemini-3-pro-image-preview_russian.png |
| `openai/gpt-5-image-mini` | business | ✗ | 1.0 | None | - | - | HTTP 403: {"error":{"message":"Provider returned error","code":403,"metadata":{" |
| `openai/gpt-5-image-mini` | russian | ✗ | 0.92 | None | - | - | HTTP 403: {"error":{"message":"Provider returned error","code":403,"metadata":{" |
| `openai/gpt-5-image` | business | ✗ | 0.88 | None | - | - | HTTP 403: {"error":{"message":"Provider returned error","code":403,"metadata":{" |
| `openai/gpt-5-image` | russian | ✗ | 0.86 | None | - | - | HTTP 403: {"error":{"message":"Provider returned error","code":403,"metadata":{" |
| `openai/gpt-5.4-image-2` | business | ✗ | 0.9 | None | - | - | HTTP 402: {"error":{"message":"This request requires more credits, or fewer max_ |
| `openai/gpt-5.4-image-2` | russian | ✗ | 0.82 | None | - | - | HTTP 402: {"error":{"message":"This request requires more credits, or fewer max_ |
| `bytedance-seed/seedream-4.5` | business | ✓ | 9.25 | 0.0396 | 388.2 | 2048x2048 | docs\image_generation\progress_reports\20260425_bytedance-seed__seedream-4.5_business.jpg |
| `bytedance-seed/seedream-4.5` | russian | ✓ | 7.91 | 0.0396 | 416.4 | 2048x2048 | docs\image_generation\progress_reports\20260425_bytedance-seed__seedream-4.5_russian.jpg |
| `black-forest-labs/flux.2-flex` | business | ✓ | 10.78 | 0.0495 | 294.0 | 1024x768 | docs\image_generation\progress_reports\20260425_black-forest-labs__flux.2-flex_business.png |
| `black-forest-labs/flux.2-flex` | russian | ✓ | 10.62 | 0.0495 | 332.5 | 1024x768 | docs\image_generation\progress_reports\20260425_black-forest-labs__flux.2-flex_russian.png |
| `black-forest-labs/flux.2-pro` | business | ✓ | 8.23 | 0.0297 | 233.9 | 1024x768 | docs\image_generation\progress_reports\20260425_black-forest-labs__flux.2-pro_business.png |
| `black-forest-labs/flux.2-pro` | russian | ✓ | 8.2 | 0.0297 | 280.4 | 1024x768 | docs\image_generation\progress_reports\20260425_black-forest-labs__flux.2-pro_russian.png |
| `black-forest-labs/flux.2-max` | business | ✓ | 19.28 | 0.0693 | 262.3 | 1024x1024 | docs\image_generation\progress_reports\20260425_black-forest-labs__flux.2-max_business.png |
| `black-forest-labs/flux.2-max` | russian | ✓ | 14.18 | 0.0693 | 215.1 | 1024x1024 | docs\image_generation\progress_reports\20260425_black-forest-labs__flux.2-max_russian.png |
| `sourceful/riverflow-v2-fast` | business | ✓ | 30.18 | 0.0198 | 13.1 | ? | docs\image_generation\progress_reports\20260425_sourceful__riverflow-v2-fast_business.png |
| `sourceful/riverflow-v2-fast` | russian | ✓ | 60.96 | 0.0198 | 8.7 | ? | docs\image_generation\progress_reports\20260425_sourceful__riverflow-v2-fast_russian.png |
