# Single Provider Refactor

## Overview
Refactored the LLM system to use a single provider from the `AGENT_PROVIDER` environment variable instead of a complex fallback sequence system. This creates a lean, DRY implementation with no legacy code.

## Changes Made

### 1. LLMManager (`llm_manager.py`)
- **Removed**: `DEFAULT_LLM_SEQUENCE` constant
- **Added**: `get_agent_llm()` method that returns a single LLM instance
- **Simplified**: No more sequence/fallback logic

### 2. LangChainWrapper (`langchain_wrapper.py`)
- **Removed**: `_select_provider()` method with complex logic
- **Added**: `_get_agent_provider()` method for simple provider retrieval
- **Simplified**: `invoke()` and `astream()` methods now use single provider
- **Removed**: All fallback/sequence logic

### 3. CoreAgent (`core_agent.py`)
- **Simplified**: Both `process_question()` and `process_question_stream()` methods
- **Removed**: All sequence/fallback loops
- **Added**: Direct single LLM instance usage

## Key Benefits

1. **Lean & DRY**: No duplicate code or complex fallback logic
2. **Simple**: One provider at a time, controlled by environment variable
3. **Maintainable**: Easy to understand and modify
4. **Flexible**: All providers still supported, just one at a time
5. **No Legacy**: Clean implementation without backward compatibility burden

## Usage

Set the `AGENT_PROVIDER` environment variable to choose your LLM:

```bash
export AGENT_PROVIDER=mistral      # Use Mistral
export AGENT_PROVIDER=openrouter   # Use OpenRouter
export AGENT_PROVIDER=gemini       # Use Gemini
export AGENT_PROVIDER=groq         # Use Groq
```

## Supported Providers

- `mistral` - Mistral AI
- `openrouter` - OpenRouter API
- `gemini` - Google Gemini
- `groq` - Groq API
- `huggingface` - Hugging Face Inference API
- `gigachat` - GigaChat

## Testing

Run the test script to verify the changes:

```bash
python misc_files/test_single_provider.py
```

## Migration Notes

- No breaking changes to public APIs
- All existing functionality preserved
- Simply uses one provider instead of multiple
- Environment variable controls provider selection
