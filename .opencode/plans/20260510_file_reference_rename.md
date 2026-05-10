# Rename `file_reference` to `filename` / `generated_filename` for LLM clarity

## Spec

### Problem

The param/key name `file_reference` is opaque to the LLM. When `generate_ai_image` returns `file_reference: 'llm_...png'`, the LLM doesn't understand that this IS the generated image — it reads the key name as a metadata pointer, not the result itself.

### Solution

Rename all LLM-facing occurrences of `file_reference` to clear names:

| Role | Old | New | Rationale |
|------|-----|-----|-----------|
| Tool INPUT param (LLM passes a filename) | `file_reference` | `filename` | "Pass the filename" |
| `generate_ai_image` OUTPUT key | `"file_reference"` | `"generated_filename"` | "This IS the image I created" |
| Fetch tool OUTPUT key (`fetch_record_image_file`, `fetch_record_document_file`) | `"file_reference"` | `"filename"` | "This is the file I fetched" |
| Shared Pydantic constant | `CHAT_FILE_REFERENCE_DESCRIPTION` | `CHAT_FILENAME_DESCRIPTION` | Follows rename |
| Shared Pydantic constant | `CHAT_FILE_REFERENCE_RESULT_HINT` | `CHAT_FILENAME_RESULT_HINT` | Follows rename |
| Internal API `FileUtils.resolve_file_reference` | `resolve_file_reference` | `resolve_filename` | Consistency |
| Internal API `FileUtils.read_file_reference_bytes` | `read_file_reference_bytes` | `read_file_bytes` | Simpler + consistent |
| Internal API `FileUtils.upload_basename_from_reference` | `upload_basename_from_reference` | `upload_basename` | Remove redundant "from_reference" |
| File `file_reference_tool_text.py` | — | (keep as-is) | Internal module, never LLM-facing |

### Non-goals

- No backward compatibility adapters
- No changes to agent internal methods (`register_file`, `get_file_path`) — they're implementation details
- No changes to doc/plan files (not code)
- No variable rename in tests of unrelated code

### Files to modify

| # | File | What changes |
|---|------|-------------|
| 1 | `tools/file_reference_tool_text.py` | Constant names `CHAT_FILE_REFERENCE_*` → `CHAT_FILENAME_*`; description text |
| 2 | `tools/tools.py` | INPUT params `file_reference` → `filename` (7 tools); OUTPUT key in `generate_ai_image` (9 places); docstrings; internal helpers (4 references) |
| 3 | `tools/templates_tools/tool_record_image.py` | Pydantic field name `file_reference` → `filename`; validator; tool sig; docstring; `read_file_bytes` call |
| 4 | `tools/templates_tools/tool_record_document.py` | Same pattern as record_image |
| 5 | `agent_ng/_file_attachment.py` | `tool_result.get("file_reference")` → `tool_result.get("generated_filename") or tool_result.get("filename")` |
| 6 | `tools/file_utils.py` | `resolve_file_reference` → `resolve_filename`; `read_file_reference_bytes` → `read_file_bytes`; `upload_basename_from_reference` → `upload_basename` |
| 7 | ~14 test files | Update param names and dict key assertions |

---

## Plan

### Phase 0: Shared constants (foundation)

**File: `tools/file_reference_tool_text.py`**

- Rename `CHAT_FILE_REFERENCE_DESCRIPTION` → `CHAT_FILENAME_DESCRIPTION`
- Rename `CHAT_FILE_REFERENCE_RESULT_HINT` → `CHAT_FILENAME_RESULT_HINT`
- Update description text: replace `file_reference` references with `filename`

---

### Phase 1: Internal API (`file_utils.py`)

**File: `tools/file_utils.py`**

- Rename method `resolve_file_reference` → `resolve_filename`
- Rename method `read_file_reference_bytes` → `read_file_bytes`
- Rename method `upload_basename_from_reference` → `upload_basename`
- Update param name `file_reference` → `filename` in all 3 methods
- Update docstrings

**Note:** `resolve_file_path` already uses `original_filename` — keep as-is.

---

### Phase 2: Attachment consumer

**File: `agent_ng/_file_attachment.py`**

- Change `tool_result.get("file_reference")` → `tool_result.get("generated_filename") or tool_result.get("filename")`
- Update docstring

---

### Phase 3: Tool source files

**File: `tools/tools.py`**

1. **INPUT params** (7 tools): `file_reference: str` → `filename: str`
2. **Docstrings**: `file_reference (str):` → `filename (str):`
3. **`generate_ai_image` return dict keys**: `"file_reference"` → `"generated_filename"` (9 places)
4. **`generate_ai_image` docstring**: update description text
5. **Internal helpers** (`_resolve_reference_images` docstring, calls to `FileUtils.read_file_bytes`)
6. **Other tools** (`analyze_video`, `analyze_audio`): `file_reference` → `filename`

**Files 3-4: `tools/templates_tools/tool_record_image.py` and `tool_record_document.py`**

- Pydantic field: `file_reference: str = Field(description=CHAT_FILENAME_DESCRIPTION)`
- Validator: `@field_validator("record_id", "attribute_system_name", "filename", mode="before")`
- Tool sig param: `file_reference` → `filename`
- Function body: `FileUtils.read_file_bytes(filename, agent)`
- Docstrings: update text

---

### Phase 4: Tests

Update all test files. For each:

- INPUT params: `file_reference=` → `filename=`
- OUTPUT assertions: `.get("file_reference")` → `.get("generated_filename")` or `.get("filename")`
- Dict construction: `"file_reference": x` → `"generated_filename": x` or `"filename": x`
- Test data fixture dicts

**Key files:** `test_generate_ai_image.py`, `test_platform_document_pipeline.py`, `test_attach_injected_agent_parity.py`, `test_vl_tools.py`, `test_understand_video.py`, `test_understand_audio.py`, `test_integration_image_api_live.py`, `test_integration_document_api_live.py`, `test_analyze_image_pdf.py`, `test_chat_file_rendering.py`, `test_message_content_text.py`, `test_markitdown_support.py`, `test_vl_integration.py`, `playground_sandbox_record_files.py`, `harness_document_attribute_matrix.py`

---

### Phase 5: Verification

```powershell
ruff check --fix --unsafe-fixes tools/tools.py tools/file_utils.py tools/templates_tools/ tools/file_reference_tool_text.py agent_ng/_file_attachment.py

python -m pytest tools/_tests/test_generate_ai_image.py tools/_tests/test_attach_injected_agent_parity.py tools/_tests/test_vl_tools.py tools/_tests/test_understand_video.py tools/_tests/test_understand_audio.py -x

python -m pytest agent_ng/_tests/test_chat_file_rendering.py agent_ng/_tests/test_message_content_text.py agent_ng/_tests/test_markitdown_support.py agent_ng/_tests/test_vl_integration.py -x

python -m pytest tools/_tests/test_platform_document_pipeline.py -x

mypy agent_ng/
```

---

## Execution order

```
Phase 0: file_reference_tool_text.py (constants)
    ↓
Phase 1: file_utils.py (internal API)
    ↓
Phase 2: _file_attachment.py (consumer)
    ↓
Phase 3: tools.py + record_image.py + record_document.py (tool sources)
    ↓
Phase 4: All test files
    ↓
Phase 5: ruff check + mypy + pytest (verification)
```

## Edge cases

- `build_file_attachment` checks `generated_filename` first, then `filename` — handles both `generate_ai_image` and fetch tools.
- Phase 1 must precede Phase 3 (tool files call `FileUtils.read_file_bytes` etc.)
- Phase 0 must precede Phase 3 (tool files import `CHAT_FILENAME_*` constants)
