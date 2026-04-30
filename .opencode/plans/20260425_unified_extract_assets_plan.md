# 20260425 Unified Asset Extraction Plan (Second Iteration)

**Date:** 2026-04-25  
**Status:** Planning / Pre-Implementation  
**Author:** CMW Platform Agent (synthesized from conversation, colleague's plan, codebase analysis, and external research)  
**References:** 
- Colleague's original: [20260425_pdf_image_extraction_plan.md](../20260425_pdf_image_extraction_plan.md)
- Full conversation history in this session (explicit flag preference, unified helper, naming conventions, registry-based MD links)
- Workspace rules (AGENTS.md, .cursor/rules/*.mdc): Research-first, TDD, LangChain-pure, super-dry/lean/modular/pythonic/pydantic, Ruff before commits, tests in `agent_ng/_tests/` or `tools/_tests/`, reports with YYYYMMDD_ prefix, no overengineering, explicit error handling, session isolation.

## Objective

Create a **unified, explicit, backward-compatible** asset extraction system for documents (PDF + Office) that:
1. Returns **text/markdown immediately** on first call (no delay).
2. **Optionally** extracts and saves images (and the full markdown) as session-isolated files.
3. Registers **all** assets (`{stem}_extracted.md`, `{stem}_image_{n:02d}.{ext}`) in the agent's `file_registry` using logical names.
4. Generates markdown with **internal links using registered logical references** (e.g. `![Figure 1](report_image_01.png)`) so the LLM can later call `read_text_based_file("report_image_01.png")`, vision tools, or `attach_file_to_record_image_attribute` without re-extraction or context bloat.
5. Works identically for **chat-attached files** and **platform-fetched documents**.

This is the **second plan** — it refines the first by incorporating explicit-only semantics, a true dispatcher, MarkItDown research, and precise naming/registry integration.

## Research Summary (Ground Truth Basis)

### 1. MarkItDown (Office + PDF text)
- Primary purpose: LLM-ready Markdown from DOCX, XLSX, PPTX, PDF, images (see Microsoft/markitdown GitHub, RealPython article, Kaggle RAG examples).
- **Image handling**: Does **not** extract binary image files by default. It:
  - Generates LLM descriptions/OCR (requires `llm_client` like gpt-4o or `markitdown-ocr` plugin).
  - May output placeholders or base64 URIs with `keep_data_uris=True`.
  - GitHub issues (#56 PPTX images, #269 DOCX) confirm binary extraction needs custom logic (mammoth for DOCX, direct python-pptx/openpyxl).
- Conclusion: Perfect for rich text/tables. For **binary images**, dispatch to format-specific extractors in our unified helper. Start with PDF (strongest support); Office images as Phase 2.

### 2. PDF Image Extraction (PyMuPDF/fitz)
- `pymupdf4llm.to_markdown(..., ignore_images=True)` (current in `tools/pdf_utils.py:57-64`) is deliberate for text focus.
- Underlying `fitz` (already imported in `tools/tools.py:114` for image tools) has excellent `doc.extract_image(xref)` and page `get_images()` APIs (~10 lines).
- We already have `PDFUtils`, `PDFTextResult` (Pydantic), `local_path_text.py:83-98` routing, and `FileUtils.generate_unique_filename(session_id)`.

### 3. Existing Infrastructure (reused 100%)
- **Session isolation**: `agent_ng/langchain_agent.py:194-418` (`file_registry[(session_id, logical_name)]`), `register_file()`, `get_file_path()`, `session_cache_path`.
- **FileUtils** (`tools/file_utils.py`): `resolve_file_reference`, `create_tool_response`, `add_media_to_response`, `create_media_attachment`, `save_base64_to_file(session_id=...)`, media type detection.
- **Tool layer**: `read_text_based_file` (`tools/tools.py:792`), `read_local_path_to_plain_text` (`tools/local_path_text.py`).
- **Vision/Registry**: `agent_ng/vision_input.py` (MediaType.PDF), image analysis tools, platform image/document tools.
- **Chat + Platform**: Both paths feed the same registry (`chat_tab.py:1359`, document pipeline tests).

No new heavy dependencies needed initially (PyMuPDF already present; MarkItDown already used).

## Core Design Decisions (Incorporating All Remarks)

- **Explicit only**: `extract_images: bool = False` (default preserves **all** current behavior). LLM or caller must opt-in. Avoids wasteful extraction of 100-image PDFs.
- **Unified Helper**: **Yes** — `tools/asset_extractor.py` (new, single-responsibility) with:
  ```python
  # High-level spec only - no code yet
  def extract_assets(
      file_path: str,
      extract_images: bool = False,
      session_id: str | None = None,
      agent: Any = None
  ) -> AssetExtractionResult:  # Pydantic
  ```
  Dispatches internally:
  - `.pdf` → `PDFUtils.extract_with_assets(...)` (text via pymupdf4llm + images via fitz).
  - Office (`.docx`/`.xlsx`/`.pptx`) → MarkItDown for text + targeted image extraction (Phase 1: PDF only; Phase 2: direct libs or base64 parsing).
  - Returns `AssetExtractionResult(success, text_content, markdown_path, image_paths, registered_refs, ...)`.

- **Naming & MD Links**:
  - Logical names only: `{original_stem}_extracted.md`, `{original_stem}_image_{02d}.{ext}` (use `Path(file_path).stem` + `FileUtils.generate_unique_filename` logic).
  - Generated markdown contains **registry-friendly internal links**: `![Description from page X](report_image_01.png)`.
  - No absolute system paths ever in output. LLM references logical names → resolved via existing `resolve_file_reference` / registry.
  - Saved `.md` is registered so `read_text_based_file("report_extracted.md")` works cleanly in follow-up turns.

- **First-Call Behavior**:
  - Text/markdown returned **immediately** in tool response.
  - If images extracted: also includes `media_attachments` list + metadata with all registered references.
  - Assets saved to session-isolated location and registered.
  - Future calls reuse registered files (no re-extraction, no duplication in context).

- **Non-breaking & Lean**:
  - Default path unchanged.
  - Minimal new LOC (focus on dispatcher + PDF image path first).
  - TDD: Write tests **before** implementation (behaviors: text fidelity, registration, MD links resolve, platform/chat parity, edge cases).
  - Follow all rules: Ruff, mypy, pydantic models, centralized error handling (no `except: pass`), module docstrings, tests in appropriate `_tests/`.

## Proposed File Structure & Changes (Spec Only)

1. **`tools/asset_extractor.py`** (new): Unified dispatcher, `AssetExtractionResult` model, format-specific helpers.
2. **`tools/pdf_utils.py`**: Add `extract_with_assets(...)` (builds on existing `extract_text_from_pdf`).
3. **`tools/local_path_text.py`**: Call new `extract_assets(...)` when flag set; return enhanced tuple/result.
4. **`tools/tools.py`**: Update `read_text_based_file` signature + schema (`extract_images: bool = False`), pass through to processor, enhance response with `FileUtils` media helpers + registry calls.
5. **Tests**: `tools/_tests/test_asset_extraction.py` or `agent_ng/_tests/test_document_assets.py` (TDD first).
6. **Docs**: Update `docs/20250920_PDF_IMPLEMENTATION_SUMMARY.md` (or new progress report), AGENTS.md if needed, tool docstrings.
7. **New Plan Location**: This file + future implementation plans in `.opencode/plans/`.

## Phased Implementation (TDD-First)

**Phase 0 (This Plan)**: Research complete, TDD tests written for desired behaviors, this document finalized.

**Phase 1: PDF Only (Minimal Viable)**
- Implement `extract_assets` for PDFs.
- Update `read_text_based_file(extract_images=True)`.
- Ensure `.md` with internal registry links + image registration.
- Tests: backward compat, session isolation, first-call references, re-read of extracted.md.

**Phase 2: Office Documents**
- Add dispatch for MarkItDown + image extraction (base64 parsing or direct libs — decide based on minimal deps).
- Handle charts in XLSX/PPTX as images where appropriate.

**Phase 3: Polish & Integration**
- Vision tool synergy, platform document pipeline parity, rich response formatting.
- Update PDF skill, UI hints if needed.
- Performance: limit max images (configurable), lazy extraction.

## Success Criteria

- Existing calls unchanged (`extract_images=False`).
- First call: immediate rich text + registered asset references.
- LLM can say "now analyze image_03 from that report" → resolves via registry.
- Markdown contains usable `[]()` links using logical names.
- All files session-isolated, registered, reusable.
- Passes Ruff, mypy, tests, follows lean principles.
- No context bloat — images are references until explicitly used.

## Open Questions / Trade-offs (For Next Session)

1. Exact threshold for "significant" images (auto-filter small icons?).
2. Should extracted `.md` include full base64 for images (no) or just links (yes, per registry).
3. Office image quality: extract original format vs always PNG?
4. Should we add a dedicated `analyze_document_assets()` tool or keep it in `read_text_based_file`?
5. Integration with CMW record attachment tools (easy via registered names).

## Alignment with Workspace Rules

- Research-first (this document + web search on MarkItDown).
- TDD/SDD.
- Lean/DRY/modular (one dispatcher).
- Explicit flag, no silent failures, pydantic results.
- Session isolation via existing registry.
- Reports in `.opencode/plans/` with timestamp.
- No code written in this plan.

This second plan synthesizes the original, our full conversation, MarkItDown research (text-first, custom work needed for binary images), and existing architecture. It provides a clear, executable blueprint while staying true to "super lean" principles.

**Next Action (when ready)**: Switch to implementation mode, write tests first per TDD skill, then implement the unified helper starting with PDF support.

---
**Compiled:** 2026-04-25  
**Version:** 2.0 (refined from colleague's v1 with explicit flag, unified dispatcher, registry-first MD links)
