# Clean Gradio API Surface — Only `/api/v1/chat/completions` Public

**Date:** 20260509
**Status:** In progress

## Problem

- `agent_completions` (`gr.api`) is dead legacy — duplicate of `/api/v1/chat/completions` with fake streaming and no auth
- `/ask` and `/ask_stream` are public in Gradio API tab, causing snippet generator overhead (Gradio #13278)
- The only real public API should be `POST /api/v1/chat/completions` (native FastAPI, bearer auth, real SSE streaming)

## Root Cause: Gradio 6.10 snippet generator crash (#13278)

`gradio_client/snippet.py` calls `json.dumps()` on all public endpoint parameter defaults when rendering the API docs page. Non-JSON-serializable types crash the page. Even when it doesn't crash, processing many endpoints slows it down. Zero public endpoints = zero problem.

## Implementation Plan

### File 1: `agent_ng/app_ng_modular.py` (6 edits)

1. Remove `handle_agent_completions_payload` from `try` import block (line ~147)
2. Remove `handle_agent_completions_payload` from `except` import block (line ~187)
3. Delete `_api_agent_completions` function (lines ~1683-1692)
4. Change `ask_stream` visibility: `"public"` → `"private"` (line ~1708)
5. Change `ask` visibility: `"public"` → `"private"` (line ~1712)
6. Delete `gr.api(_api_agent_completions, api_name="agent_completions")` (line ~1714)

### File 2: `agent_ng/_tests/test_openai_compat.py` (1 edit)

7. Delete `test_nextgen_app_registers_gradio_agent_completions_api_name` test

### File 3: `AGENTS.md` (1 edit)

8. Update API surface description to reflect zero public Gradio API endpoints

## Result

- Gradio API tab: zero public endpoints, no snippet generator processing
- Public API: only `POST /api/v1/chat/completions` (real SSE streaming, bearer auth)
- `_api_ask` and `_api_ask_stream` remain functional but hidden

## Verification

```bash
python lint.py
python -m pytest agent_ng/_tests/test_openai_compat.py -v
python -m pytest agent_ng/_tests/test_queue_manager_api_visibility.py -v
```
