# OpenRouter API Compatibility Report
Generated on: 2025-09-18 01:23:29

## Summary
- Total models checked: 20
- Available on OpenRouter: 6
- High compatibility (≥80%): 4
- Medium compatibility (50-79%): 2
- Low compatibility (<50%): 14

## Detailed Results

### qwen/qwen3-coder:free
**Name:** Qwen: Qwen3 Coder 480B A35B (free)
**Available:** ✅ Yes
**Compatibility Score:** 100.00%
**Context Length:** 262,144 tokens
**Supported Parameters:** frequency_penalty, logit_bias, logprobs, max_tokens, min_p, presence_penalty, repetition_penalty, seed, stop, temperature, tool_choice, tools, top_k, top_logprobs, top_p
**Extra Features:** logit_bias, logprobs, min_p, repetition_penalty, top_k, top_logprobs
**Pricing:**

### deepseek/deepseek-chat-v3.1:free
**Name:** DeepSeek: DeepSeek V3.1 (free)
**Available:** ✅ Yes
**Compatibility Score:** 100.00%
**Context Length:** 163,840 tokens
**Supported Parameters:** frequency_penalty, include_reasoning, max_tokens, min_p, presence_penalty, reasoning, repetition_penalty, response_format, seed, stop, temperature, tool_choice, tools, top_k, top_p
**Extra Features:** include_reasoning, min_p, reasoning, repetition_penalty, response_format, top_k
**Pricing:**

### qwen/qwen3-coder:free
**Name:** Qwen: Qwen3 Coder 480B A35B (free)
**Available:** ✅ Yes
**Compatibility Score:** 100.00%
**Context Length:** 262,144 tokens
**Supported Parameters:** frequency_penalty, logit_bias, logprobs, max_tokens, min_p, presence_penalty, repetition_penalty, seed, stop, temperature, tool_choice, tools, top_k, top_logprobs, top_p
**Extra Features:** logit_bias, logprobs, min_p, repetition_penalty, top_k, top_logprobs
**Pricing:**

### mistralai/mistral-small-3.2-24b-instruct:free
**Name:** Mistral: Mistral Small 3.2 24B (free)
**Available:** ✅ Yes
**Compatibility Score:** 100.00%
**Context Length:** 131,072 tokens
**Supported Parameters:** frequency_penalty, logit_bias, logprobs, max_tokens, min_p, presence_penalty, repetition_penalty, seed, stop, structured_outputs, temperature, tool_choice, tools, top_k, top_logprobs, top_p
**Extra Features:** logit_bias, logprobs, min_p, repetition_penalty, structured_outputs, top_k, top_logprobs
**Pricing:**

### openrouter/sonoma-dusk-alpha
**Name:** Sonoma Dusk Alpha
**Available:** ✅ Yes
**Compatibility Score:** 77.78%
**Context Length:** 2,000,000 tokens
**Supported Parameters:** max_tokens, response_format, structured_outputs, tool_choice, tools
**Missing Features:** temperature, top_p, stop, frequency_penalty, presence_penalty, seed
**Extra Features:** response_format, structured_outputs
**Pricing:**

### deepseek/deepseek-r1-0528:free
**Name:** DeepSeek: R1 0528 (free)
**Available:** ✅ Yes
**Compatibility Score:** 59.26%
**Context Length:** 163,840 tokens
**Supported Parameters:** frequency_penalty, include_reasoning, logit_bias, logprobs, max_tokens, min_p, presence_penalty, reasoning, repetition_penalty, seed, stop, temperature, top_k, top_logprobs, top_p
**Missing Features:** tools, tool_choice
**Extra Features:** include_reasoning, logit_bias, logprobs, min_p, reasoning, repetition_penalty, top_k, top_logprobs
**Pricing:**

### gemini-2.5-pro
**Name:** gemini-2.5-pro
**Available:** ❌ No
**Compatibility Score:** 0.00%
**Context Length:** 2,000,000 tokens

### groq/compound
**Name:** groq/compound
**Available:** ❌ No
**Compatibility Score:** 0.00%
**Context Length:** 131,072 tokens

### llama-3.3-70b-versatile
**Name:** llama-3.3-70b-versatile
**Available:** ❌ No
**Compatibility Score:** 0.00%
**Context Length:** 131,072 tokens

### llama-3.3-70b-8192
**Name:** llama-3.3-70b-8192
**Available:** ❌ No
**Compatibility Score:** 0.00%
**Context Length:** 16,000 tokens

### Qwen/Qwen2.5-Coder-32B-Instruct
**Name:** Qwen/Qwen2.5-Coder-32B-Instruct
**Available:** ❌ No
**Compatibility Score:** 0.00%
**Context Length:** 3,000 tokens

### microsoft/DialoGPT-medium
**Name:** microsoft/DialoGPT-medium
**Available:** ❌ No
**Compatibility Score:** 0.00%
**Context Length:** 1,000 tokens

### gpt2
**Name:** gpt2
**Available:** ❌ No
**Compatibility Score:** 0.00%
**Context Length:** 1,000 tokens

### openrouter/cypher-alpha:free
**Name:** openrouter/cypher-alpha:free
**Available:** ❌ No
**Compatibility Score:** 0.00%
**Context Length:** 1,000,000 tokens

### mistral-large-latest
**Name:** mistral-large-latest
**Available:** ❌ No
**Compatibility Score:** 0.00%
**Context Length:** 32,000 tokens

### mistral-small-latest
**Name:** mistral-small-latest
**Available:** ❌ No
**Compatibility Score:** 0.00%
**Context Length:** 32,000 tokens

### mistral-medium-latest
**Name:** mistral-medium-latest
**Available:** ❌ No
**Compatibility Score:** 0.00%
**Context Length:** 32,000 tokens

### GigaChat-2
**Name:** GigaChat-2
**Available:** ❌ No
**Compatibility Score:** 0.00%
**Context Length:** 128,000 tokens

### GigaChat-2-Pro
**Name:** GigaChat-2-Pro
**Available:** ❌ No
**Compatibility Score:** 0.00%
**Context Length:** 128,000 tokens

### GigaChat-2-Max
**Name:** GigaChat-2-Max
**Available:** ❌ No
**Compatibility Score:** 0.00%
**Context Length:** 128,000 tokens

## Recommendations

### High Compatibility Models (Recommended)
- **qwen/qwen3-coder:free**: 100.00% compatibility
- **deepseek/deepseek-chat-v3.1:free**: 100.00% compatibility
- **qwen/qwen3-coder:free**: 100.00% compatibility
- **mistralai/mistral-small-3.2-24b-instruct:free**: 100.00% compatibility

### Models Missing Tool Support
- **deepseek/deepseek-r1-0528:free**: Consider disabling tool support or finding alternative

### Unavailable Models
- **gemini-2.5-pro**: Not found on OpenRouter
- **groq/compound**: Not found on OpenRouter
- **llama-3.3-70b-versatile**: Not found on OpenRouter
- **llama-3.3-70b-8192**: Not found on OpenRouter
- **Qwen/Qwen2.5-Coder-32B-Instruct**: Not found on OpenRouter
- **microsoft/DialoGPT-medium**: Not found on OpenRouter
- **gpt2**: Not found on OpenRouter
- **openrouter/cypher-alpha:free**: Not found on OpenRouter
- **mistral-large-latest**: Not found on OpenRouter
- **mistral-small-latest**: Not found on OpenRouter
- **mistral-medium-latest**: Not found on OpenRouter
- **GigaChat-2**: Not found on OpenRouter
- **GigaChat-2-Pro**: Not found on OpenRouter
- **GigaChat-2-Max**: Not found on OpenRouter
