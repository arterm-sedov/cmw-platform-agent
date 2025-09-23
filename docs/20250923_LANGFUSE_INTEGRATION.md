## Langfuse Integration (Lean, Optional)

This change adds an optional Langfuse CallbackHandler to the same streaming loop already wrapped by LangSmith `@traceable`.

### How it works
- A new `agent_ng/langfuse_config.py` provides an env-driven factory for a Langfuse `CallbackHandler`.
- `agent_ng/native_langchain_streaming.py` attaches the handler to the `astream(..., config={"callbacks": [...]})` call if enabled.

### Enable via environment
- `LANGFUSE_ENABLED=true`
- `LANGFUSE_PUBLIC_KEY=...`
- `LANGFUSE_SECRET_KEY=...`
- `LANGFUSE_HOST=https://cloud.langfuse.com` (optional)

When configured, Langfuse traces will be captured alongside LangSmith, without changing the core flow.

### Dependencies
- Added `langfuse>=2.39.0` to `requirements_ng.txt`.

### Files
- `agent_ng/langfuse_config.py` (new)
- `agent_ng/native_langchain_streaming.py` (attach handler in streaming loop)
- `requirements_ng.txt` (dependency)


