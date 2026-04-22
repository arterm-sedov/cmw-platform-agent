# Plan: Add Gradio Text File Support for Pasted Text

## Objective

Allow users to paste large text content into the MultimodalTextbox component without receiving "Invalid file type" error. Gradio converts pasted text to a file with extension based on content type.

## Problem Statement

When users paste text exceeding 1000 characters into Gradio's MultimodalTextbox:
1. Gradio frontend creates a `File` object: `new File([text], "pasted_text.txt", { type: "text/plain" })`
2. Backend validates against `file_types` list which lacks `.txt` or text MIME category
3. Error: `"Invalid file type: text/plain. Please upload a file that is one of these formats: [...]"`

## Changes Made

| File | Change | Status |
|------|--------|--------|
| `agent_ng/tabs/chat_tab.py` | Add `.txt` and `"text"` to `file_types` list | ✅ Done |
| `agent_ng/tabs/chat_tab.py` | Improve file registration logging | ✅ Done |
| System prompt | ~~Document text paste capability~~ | ❌ Removed - misleading, Gradio handles all pasted files uniformly |

---

## Step 1: Update `file_types` in MultimodalTextbox

**File:** `D:\Repo\cmw-platform-agent\agent_ng\tabs\chat_tab.py` (lines 105-119)

**Added:**
```python
file_types=[
    ".txt",  # Pasted text files from Gradio
    "text",  # MIME category for all text/*
    ".pdf", ".csv", ".tsv", ".xlsx", ".xls",
    ...
]
```

**Why both:**
- `.txt` - Catches pasted text files via extension check
- `"text"` - Catches any `text/*` MIME type via MIME category check

**Checkpoint:** ✅ Added `.txt` and `"text"` to file_types

---

## Step 2: Improve File Registration Logging

**File:** `D:\Repo\cmw-platform-agent\agent_ng\tabs\chat_tab.py` (lines 1369-1389)

**Changed from:**
```python
print(f"📁 Registered file: {original_filename} -> ...")
```

**Changed to:**
```python
logging.getLogger(__name__).debug(
    "File registered: orig_name=%s, path=%s, session=%s",
    original_filename, file_path, session_id,
)
else:
    logging.getLogger(__name__).warning(
        "Agent or register_file not available: agent=%s, has_register=%s",
        agent,
        agent and hasattr(agent, "register_file"),
    )
```

**Checkpoint:** ✅ Logging improved

---

## Step 3: System Prompt (Skipped)

**Original plan:** Add line about `pasted_text.txt`

**Decision:** Removed - Gradio handles all pasted items (images, text, etc.) as files with appropriate names/extensions. The original line was misleading.

---

## Status: COMPLETED

**Changes made:**
1. ✅ Added `.txt` and `"text"` to `file_types` in `chat_tab.py`
2. ✅ Improved file registration logging with debug/warning levels

**Not done (intentionally):**
- System prompt documentation about pasted_text.txt (misleading, removed)

**Verification:** Python syntax check passed