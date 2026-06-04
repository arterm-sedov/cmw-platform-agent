# Robust Tool And Thinking Bubbles Plan

## Goal

Implement cmw-rag-style Gradio chat progress bubbles in `cmw-platform-agent`:

- tool bubbles appear as soon as a tool call is detected;
- pending bubbles show native Gradio progress spinners;
- in-flight bubbles update in place instead of appending duplicates;
- completed bubbles collapse by switching to `metadata.status = "done"`;
- reasoning/thinking bubbles update while streamed reasoning arrives and collapse at turn completion;
- a generating-answer bubble appears while the model transitions from tool work to final response text;
- file attachments, token/cost stats, session status, and existing SSE/API behavior remain non-breaking.

Branch: `feature/robust-tool-thinking-bubbles`.

## Research Summary

### Official References

- Gradio `Chatbot` accepts OpenAI-style message dictionaries with `role` and `content`, and supports metadata for tool/thought display via `metadata={"title": ...}`: [Gradio Chatbot docs](https://www.gradio.app/docs/gradio/chatbot).
- Gradio `Chatbot` has built-in `reasoning_tags`, which extract tagged reasoning into collapsible messages with metadata title. This confirms that collapsible reasoning/tool UI is a supported component pattern, but this repo should keep explicit control because provider reasoning can arrive outside simple tag pairs: [Gradio Chatbot docs](https://www.gradio.app/docs/gradio/chatbot).
- Gradio streaming outputs are implemented with Python generators that yield successive component values, matching this repo's existing full-history chat streaming contract: [Gradio streaming outputs](https://www.gradio.app/guides/streaming-outputs).
- LangChain provides a standard model and agent abstraction that supports tool calling and streaming across providers; this repo already uses `astream()` and provider-normalized `tool_call_chunks`: [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview).

### Local Gradio Source Reference

Use `D:\Repo\gradio` as the source-code reference for Gradio behavior.

Key files:

- `D:\Repo\gradio\gradio\components\chatbot.py`
- `D:\Repo\gradio\js\chatbot\shared\Thought.svelte`
- `D:\Repo\gradio\demo\chatbot_with_tools\run.py`
- `D:\Repo\gradio\demo\chatbot_thoughts\run.py`
- `D:\Repo\gradio\demo\chatbot_reasoning_tags\run.py`

Source-confirmed behavior:

- `MetadataDict.status` accepts `"pending"` and `"done"`.
- `"pending"` shows a spinner and initializes the thought accordion open.
- `"done"` initializes the thought accordion closed.
- missing `status` initializes open with no spinner.
- `metadata.id` and `metadata.parent_id` are supported for stable/nested thoughts.
- `metadata.duration` is supported for displaying elapsed thought/tool time.

### cmw-rag Implementation

Key files:

- `D:\Repo\cmw-rag\rag_engine\api\stream_helpers.py`
- `D:\Repo\cmw-rag\rag_engine\api\app.py`
- `D:\Repo\cmw-rag\rag_engine\tests\test_ui_metadata.py`
- `D:\Repo\cmw-rag\rag_engine\tests\test_chat_with_metadata_analysis.py`

Observed patterns:

- `stream_helpers.py` centralizes message builders such as `yield_thinking_spinner`, `yield_search_started`, `yield_search_bubble`, `yield_thinking_block`, `yield_generating_answer`, and `yield_reasoning_bubble`.
- Pending state is represented directly in Gradio message metadata:
  - `metadata.title` names the collapsible block.
  - `metadata.ui_type` makes UI-only messages filterable.
  - `metadata.status = "pending"` shows the native spinner.
  - `metadata.status = "done"` collapses completed bubbles.
  - `metadata.id` / `search_id` gives stable in-place update keys.
- History helpers mutate the correct bubble:
  - update last matching `ui_type`;
  - update by stable id for parallel/repeated calls;
  - remove transient UI-only blocks when needed;
  - drain `AgentContext.pending_ui_messages`.
- `app.py` always yields the full `gradio_history` list after changes, preserving metadata blocks while answer content streams.
- Reasoning is handled as a separate per-turn state machine:
  - parse `<think>...</think>` variants and Harmony-style streams;
  - update one reasoning bubble as new text arrives;
  - keep only a bounded tail in UI while retaining diagnostics;
  - finalize by setting `metadata.status = "done"`.

### Current cmw-platform-agent State

Key files:

- `D:\Repo\cmw-platform-agent\agent_ng\app_ng_modular.py`
- `D:\Repo\cmw-platform-agent\agent_ng\native_langchain_streaming.py`
- `D:\Repo\cmw-platform-agent\agent_ng\debug_streamer.py`
- `D:\Repo\cmw-platform-agent\agent_ng\_tests\test_chat_file_rendering.py`
- `D:\Repo\cmw-platform-agent\agent_ng\_tests\test_chat_tab_stream_contract.py`

Observed gaps:

- `native_langchain_streaming.py` collects `tool_call_chunks`, but the `tool_start` event is commented out. The UI only sees `tool_end`, so no spinner can appear at tool-call time.
- `app_ng_modular.py` appends separate `tool_start` and `tool_end` accordions when events exist. It uses only `metadata.title`; it does not set `status`, `ui_type`, or stable ids.
- `tool_end` currently appends another message instead of updating the earlier bubble. This causes noisy duplicate tool blocks and prevents completion collapse.
- `debug_streamer.py` has an older `ThinkingTransparency` helper that already knows `status = "pending"` and `"done"`, but it is not wired into the main chat stream and contains silent exception patterns to clean later.
- File attachment rendering is already covered by tests and must remain appended after the completed tool bubble.

## Proposed Contract

Add a small UI-message layer, not a broad streaming rewrite.

Important boundary:

- `agent.memory_manager`, `native_langchain_streaming.py`'s `messages` list, and `turn_complete` ordered snapshots remain the source of truth for agent memory.
- Gradio `working_history` is only a render buffer for the browser.
- Tool/thinking/reasoning bubbles are UI-only and must never be used as canonical memory.
- Do not rebuild model context from Gradio chat history for this feature.
- Do not store UI bubbles in `memory_manager`.

Message metadata contract:

```python
{
    "title": "Tool: list_applications",
    "ui_type": "tool_call",
    "status": "pending",  # later "done"
    "id": "<tool_call_id or generated stable id>",
    "tool_name": "list_applications",
}
```

Completion updates the same message:

```python
metadata["status"] = "done"
content = rendered_result_summary
metadata["duration_ms"] = elapsed_ms
metadata["duplicate_count"] = duplicate_count
```

Reasoning/thinking contract:

```python
{
    "title": "Thinking",
    "ui_type": "reasoning",
    "status": "pending",  # later "done"
    "id": "<turn-local id>",
}
```

UI-only messages must be excluded from LLM context, exports where appropriate, and token counting.

Generating-answer contract:

```python
{
    "title": "Generating answer",
    "ui_type": "generating_answer",
    "status": "pending",  # later "done"
    "id": "<turn-local id>",
}
```

The generating-answer bubble is UI-only. It should be created after tool work completes and before final answer text streams, then marked `done` on the first visible answer token or at turn finalization. Prefer collapsing it instead of removing it so users see a stable phase trace without layout jitter.

## TDD Implementation Plan

### 1. Add Message Helper Tests First

Create `agent_ng/_tests/test_stream_ui_messages.py`.

Expected coverage:

- `make_tool_call_bubble()` returns assistant message with `ui_type="tool_call"`, `status="pending"`, stable id, title, and safe content.
- `complete_tool_call_bubble()` updates the matching bubble in place by id and sets `status="done"`.
- completion keeps one bubble for one tool, not two.
- missing bubble falls back to appending a completed bubble to preserve visibility.
- duplicate count and tool cost metadata are preserved.
- file attachment metadata survives completion so existing `build_file_bubbles()` path still works.
- `make_reasoning_bubble()` creates pending reasoning message.
- `update_reasoning_bubble()` updates content by id and caps display size.
- `complete_reasoning_bubble()` marks done.
- `make_generating_answer_bubble()` creates a pending generation message.
- `complete_generating_answer_bubble()` marks it done without affecting assistant answer content.

### 2. Add A Lean Helper Module

Create `agent_ng/chat_stream_ui.py`.

Responsibilities:

- generate short ids when provider tool ids are absent;
- build tool pending/completed messages;
- update history by `metadata.id` and `ui_type`;
- build/update/finalize reasoning bubbles;
- build/finalize generating-answer bubbles;
- copy metadata before mutation where tests need value semantics;
- keep formatting small and language-ready via existing `format_translation`.

Do not place agent execution logic here.

### 3. Emit Tool Start Events From Streaming Layer

Update `agent_ng/native_langchain_streaming.py`.

Tasks:

- when a new `tool_call_chunk` includes a tool name, yield `StreamingEvent(event_type="tool_start", ...)`;
- include `tool_name`, `tool_call_id`, `title`, and a minimal argument preview if available;
- dedupe start emission per `tool_call_id`;
- preserve existing `tool_end` payload shape for compatibility;
- add `started_at` or `started_monotonic` to internal state for `duration_ms` on completion.

Checkpoint:

- tool start appears before execution begins, not after the tool result returns.

### 4. Update App Chat Adapter To Mutate Bubbles

Update `agent_ng/app_ng_modular.py`.

Tasks:

- replace inline `tool_start` append logic with `chat_stream_ui.upsert_tool_pending(...)`;
- replace `tool_end` append logic with `chat_stream_ui.complete_tool_bubble(...)`;
- after the final tool result in an iteration, show a pending `generating_answer` bubble before the next model call starts producing final text;
- on the first non-empty user-facing answer token, mark `generating_answer` as `done`;
- keep `build_file_bubbles(file_att)` immediately after the completed tool bubble;
- continue yielding `(working_history, "")` exactly as required by `ChatTab`;
- keep sidebar `iteration_progress` behavior intact;
- ensure regular assistant answer streaming still updates only the non-metadata assistant message.
- never feed `working_history` tool/thought bubbles back into agent memory; leave memory writes in `native_langchain_streaming.py`.

Checkpoint:

- a single tool call produces one collapsible message that spins while pending and collapses when done.

### 5. Add Reasoning Bubble Support Conservatively

Update `agent_ng/app_ng_modular.py` with helper-backed per-turn reasoning state.

Tasks:

- parse streamed content for `<think>...</think>`, escaped variants, and optionally provider-specific reasoning content if exposed in metadata/content blocks;
- route reasoning chunks into one `ui_type="reasoning"` bubble;
- keep user-facing answer text free of leaked reasoning tags;
- finalize pending reasoning at `completion`, `turn_complete`, cancellation, and error paths;
- keep this behind a simple helper path so providers without reasoning behave unchanged.

Checkpoint:

- regular models do not gain empty reasoning bubbles.
- thinking models show a live pending bubble that collapses at completion.

### 6. Preserve Context And Export Boundaries

Audit and update filters that convert chat history into model/export/token inputs.

Likely files:

- `agent_ng/token_counter.py`
- `agent_ng/conversation_summary.py`
- chat export helpers covered by `agent_ng/_tests/test_chat_export_gr_content.py`

Tasks:

- exclude `ui_type in {"tool_call", "reasoning", "thinking", "generating_answer"}` from LLM prompts unless explicitly intended;
- preserve visible export summaries if current export behavior includes metadata blocks;
- avoid leaking API-oriented names in user-facing text where CMW terminology says to use friendly names.
- add tests proving UI-only bubbles do not alter `memory_manager` content, `ordered_messages`, or token counting inputs.

### 7. Add Integration Tests

Extend or add tests:

- `agent_ng/_tests/test_chat_stream_tool_bubbles.py`
- `agent_ng/_tests/test_chat_reasoning_bubbles.py`
- targeted additions to `test_chat_file_rendering.py`.

Behavior tests:

- simulated `tool_start` followed by `tool_end` yields one bubble with `status="done"`;
- `tool_end` without prior `tool_start` still displays completed tool output;
- two distinct tool ids create two distinct bubbles;
- duplicate tool calls do not duplicate completed UI blocks;
- reasoning chunks update in place and finalize;
- generating-answer bubble appears before final text after tool work and collapses on first visible token;
- no generating-answer bubble appears for an immediate no-tool response until there is an actual waiting phase worth showing;
- file bubbles still appear after completed tool output;
- stream wrapper still yields exactly two outputs.

### 8. Optional Browser Verification

After implementation, run the app and visually verify in Gradio:

- tool call starts immediately with spinner;
- content updates while tool runs when possible;
- completed tool bubble collapses;
- reasoning bubble updates live and collapses;
- answer streaming remains smooth;
- mobile/narrow layout does not overlap.

Use the Browser plugin for localhost verification if the app port is known.

## Verification Commands

Activate venv first:

```powershell
.venv\Scripts\Activate.ps1
```

Focused checks:

```powershell
python -m pytest agent_ng/_tests/test_stream_ui_messages.py -v
python -m pytest agent_ng/_tests/test_chat_stream_tool_bubbles.py -v
python -m pytest agent_ng/_tests/test_chat_reasoning_bubbles.py -v
python -m pytest agent_ng/_tests/test_chat_file_rendering.py -v
python -m pytest agent_ng/_tests/test_chat_tab_stream_contract.py -v
ruff check agent_ng/chat_stream_ui.py agent_ng/app_ng_modular.py agent_ng/native_langchain_streaming.py
```

Broader safety checks:

```powershell
python -m pytest -m "not slow" agent_ng/_tests/
python lint.py
```

Manual smoke:

```powershell
python agent_ng/app_ng_modular.py
```

## Risks And Mitigations

- Risk: Gradio treats metadata status changes only when full history is yielded.
  Mitigation: preserve the existing full `working_history` yield pattern and test in the app.
- Risk: streamed tool call chunks can arrive without ids or names.
  Mitigation: generate ids only when missing and wait for tool name before showing a bubble.
- Risk: reasoning parsing can remove normal text that merely mentions `<think>`.
  Mitigation: apply conservative leading/block parsing and tests for literal tag examples.
- Risk: UI-only messages enter model context or token stats.
  Mitigation: centralize `is_ui_only_chat_message()` and test conversion paths.
- Risk: existing file attachment rendering regresses.
  Mitigation: keep attachment handling after tool completion and rerun current file rendering tests.

## Implementation Order

1. Tests for helper behavior.
2. `chat_stream_ui.py` helper module.
3. Streaming-layer `tool_start` event emission.
4. App adapter in-place bubble update.
5. Reasoning bubble extraction/finalization.
6. Context/export/token filters.
7. Focused tests, lint, then manual Browser verification.
