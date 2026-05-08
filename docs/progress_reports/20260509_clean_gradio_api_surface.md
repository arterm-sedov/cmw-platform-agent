# Clean Gradio API Surface — Only `/api/v1/chat/completions` Public

**Date:** 20260509
**Status:** Done

## Problem

- `agent_completions` (`gr.api`) was dead legacy — duplicate of `/api/v1/chat/completions` with fake streaming and no auth
- `/ask` and `/ask_stream` were public in Gradio API tab, risking snippet generator crash (Gradio #13278)
- Only real public API should be `POST /api/v1/chat/completions` (native FastAPI, bearer auth, real SSE streaming)

## Changes

### agent_ng/app_ng_modular.py
- Removed `handle_agent_completions_payload` from both import blocks
- Deleted `_api_agent_completions` function (11 lines)
- Removed invisible `gr.Textbox` hack — 6 components + 1 submit handler (18 lines)
- Replaced with 2-line `gr.api(...)` block, both `api_visibility="private"`
- Removed stray comment line
- **Net: ~30 lines removed, 2 added**

### agent_ng/_tests/test_openai_compat.py
- Deleted `test_nextgen_app_registers_gradio_agent_completions_api_name` (6 lines)

### AGENTS.md
- Updated API surface description: zero public Gradio endpoints, only `/api/v1/chat/completions`

## Result

- Gradio API tab: zero public endpoints, no snippet generator processing risk
- Public API surface: only `POST /api/v1/chat/completions` (real SSE streaming, bearer auth)

## Verification

```
test_openai_compat.py — 41 passed
test_queue_manager_api_visibility.py — 3 passed
ruff check — clean on changed files
```
