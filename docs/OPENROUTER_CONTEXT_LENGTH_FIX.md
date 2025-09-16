# OpenRouter DeepSeek Context Length Fix

## Issue Summary

OpenRouter DeepSeek is returning a 400 error when the context length exceeds 163,840 tokens:

```
Error code: 400 - {'error': {'message': 'This endpoint's maximum context length is 163840 tokens. However, you requested about 270231 tokens (249066 of text input, 19117 of tool input, 2048 in the output). Please reduce the length of either one, or use the "middle-out" transform to compress your prompt automatically.', 'code': 400, 'metadata': {'provider_name': None}}}
```

## Current Configuration Analysis

### OpenRouter Configuration (agent_ng/llm_manager.py)
- **Current token_limit**: 100,000 tokens
- **Actual limit**: 163,840 tokens (as per error)
- **enable_chunking**: False
- **Issue**: Configuration doesn't match actual API limits

### Existing Context Management
- ✅ Token counting via tiktoken in `agent_old/token_manager.py`
- ✅ Error detection for context length in `agent_ng/error_handler.py`
- ✅ Message truncation in `agent_old/agent.py`
- ❌ No middle-out transform support
- ❌ OpenRouter-specific context handling

## Proposed Solution

### 1. Update OpenRouter Configuration
```python
LLMProvider.OPENROUTER: LLMConfig(
    name="OpenRouter",
    type_str="openrouter",
    api_key_env="OPENROUTER_API_KEY",
    api_base_env="OPENROUTER_BASE_URL",
    max_history=15,  # Reduced from 20
    tool_support=True,
    force_tools=False,
    models=[
        {
            "model": "deepseek/deepseek-chat-v3.1:free",
            "token_limit": 150000,  # Updated to 150k (safe margin)
            "max_tokens": 2048,
            "temperature": 0,
            "force_tools": True
        },
        # ... other models
    ],
    enable_chunking=True  # Enable chunking for OpenRouter
)
```

### 2. Add Middle-Out Transform Support
- Implement OpenRouter's "middle-out" transform parameter
- Add context compression for large prompts
- Integrate with existing error handling

### 3. Enhanced Context Management
- Add OpenRouter-specific context length validation
- Implement smart message truncation for OpenRouter
- Add tool input size management

## Implementation Priority

1. **High**: Update token limits to match actual API limits
2. **High**: Enable chunking for OpenRouter
3. **Medium**: Add middle-out transform support
4. **Low**: Enhanced context compression

## Files to Modify

1. `agent_ng/llm_manager.py` - Update OpenRouter configuration
2. `agent_ng/error_handler.py` - Add OpenRouter context error handling
3. `agent_ng/message_processor.py` - Add context length validation
4. `agent_ng/langchain_wrapper.py` - Add middle-out transform support

## Expected Outcome

- OpenRouter DeepSeek will work within actual 163,840 token limit
- Automatic context compression for large prompts
- Better error handling and recovery
- Maintained functionality with improved reliability
