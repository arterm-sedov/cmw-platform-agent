# 20260425 Impeccable Consensus Asset Extraction Plan

**Date:** 2026-04-25  
**Status:** Final Consensus Planning Document  
**Version:** 3.0 (Impeccable Synthesis of v1 + v2 + Full Conversation)  
**Author:** CMW Platform Agent (synthesized from both prior plans, user remarks, MarkItDown research, and codebase analysis)  
**References:** 
- [20260425_pdf_image_extraction_plan.md](../20260425_pdf_image_extraction_plan.md) (colleague's original)
- [20260425_unified_extract_assets_plan.md](../20260425_unified_extract_assets_plan.md) (previous synthesis)
- Full conversation thread (explicit flag, unified dispatcher, precise naming, registry-first MD links without system paths, immediate text on first call, session isolation, lean principles)
- Workspace rules (AGENTS.md, .cursor/rules/*.mdc): Research-first, TDD/SDD, LangChain-pure, super-dry/lean/modular/pythonic/pydantic, Ruff, tests in appropriate `_tests/`, timestamped reports, no overengineering, explicit error handling, no silent exceptions, maximize data-ink ratio.

## Objective (Consensus)

Create a single, **unified, explicit, backward-compatible, and impeccably lean** asset extraction system that works seamlessly for:
- PDFs (primary strength)
- Office documents (DOCX, XLSX, PPTX)
- Both **chat-attached files** and **platform-fetched document attributes**

Core invariants (merged from both plans + user remarks):
1. **Immediate text/markdown** is always returned on the first call (no delay or extra steps).
2. **Images are extracted only when explicitly requested** (`extract_images=True` — default `False` to preserve all existing behavior).
3. Extracted assets (full markdown + images) are saved to **session-isolated locations**, registered in the agent's `file_registry` using **logical human-readable names only**.
4. The saved markdown contains **internal links using registered logical references** only (e.g. `![Figure from page 3](report_image_01.png)`). No absolute system paths, no filesystem leakage. The LLM can reference these names in future calls (`read_text_based_file("report_extracted.md")` or vision tools) without duplication or re-extraction.
5. Perfect symmetry between chat uploads and CMW platform document records.
6. Follows "super dry, super lean, abstract, modular, pythonic, pydantic" + TDD-first + research-grounded principles.

This plan represents the **impeccable consensus** — it takes the strongest elements from both prior documents, resolves all open questions from the conversation, and eliminates any remaining ambiguity.

## Research Summary (Ground Truth — Updated)

**MarkItDown** (for Office + PDF text):
- Excellent for LLM-ready Markdown (text, headings, tables, some alt-text/OCR via plugins or LLM client).
- Does **not** extract binary images as separate files by default (confirmed via GitHub issues #56, #269, Microsoft docs, RealPython, and Kaggle examples). It favors descriptions or base64 URIs.
- Conclusion: Use for rich text extraction; layer format-specific image extraction on top.

**PyMuPDF / fitz** (PDF images):
- `pymupdf4llm.to_markdown(ignore_images=True)` is the current deliberate choice (`tools/pdf_utils.py:57-64`).
- `fitz` (already used in `tools/tools.py:114` for image tools) provides clean `doc.extract_image(xref)` and page image APIs (~10-15 lines of lean code).

**Existing Infrastructure** (100% reuse — no new foundations needed):
- Session-isolated registry: `agent_ng/langchain_agent.py:194-418` (`file_registry[(session_id, logical_name)]`, `register_file()`, `get_file_path()`).
- `tools/file_utils.py`: `generate_unique_filename(session_id)`, `resolve_file_reference`, `create_media_attachment`, `add_media_to_response`, `save_base64_to_file(session_id=...)`.
- Central router: `tools/local_path_text.py:83-98` (PDF) + `120-140` (Office via MarkItDown).
- Main tool: `tools/tools.py:792` (`read_text_based_file`).
- Vision and platform tools already understand registered references.

No new heavy dependencies for Phase 1 (PDF + unified dispatcher). Office image extraction can use existing MarkItDown patterns + minimal targeted helpers.

## Consensus Design Decisions

**Unified Entry Point (resolves both plans' approaches):**
- New single-responsibility module: `tools/asset_extractor.py`.
- High-level function:
  ```python
  # Spec only — high-level interface
  def extract_assets(
      file_path: str,
      extract_images: bool = False,
      session_id: str | None = None,
      agent: Any = None
  ) -> AssetExtractionResult:  # New Pydantic model
  ```
  - Dispatches by extension (PDF → enhanced `PDFUtils`; Office → MarkItDown text + image logic).
  - Returns structured `AssetExtractionResult(success, text_content, markdown_path, image_paths, registered_refs, media_attachments, error)`.

**Explicit-Only Flag:**
- `extract_images: bool = False` (default = current text-only behavior everywhere).
- Matches both plans and your explicit preference. Prevents unnecessary extraction of documents with many images.

**Naming Convention (precise consensus):**
- Markdown: `{original_stem}_extracted.md`
- Images: `{original_stem}_image_{n:02d}.{ext}` (zero-padded, traceable)
- All names are **logical only** (passed to `agent.register_file(logical_name, physical_path)`).
- Generated markdown uses these logical names in links. Resolution happens through the existing registry — no system paths ever exposed to the LLM.

**First-Call Behavior (key user requirement):**
- Always returns full markdown text immediately in the tool response.
- When `extract_images=True`: additionally registers assets, returns `media_attachments` list, and metadata with all logical references for immediate future use.
- Saved `.md` is registered so subsequent `read_text_based_file("report_extracted.md")` works cleanly with zero duplication.

**Integration Points:**
- Update `read_text_based_file` (and its schema) to accept and forward `extract_images`.
- Enhance `local_path_text.py` to call the new dispatcher when flag is set.
- Enhance `PDFUtils` with image extraction (builds directly on existing methods).
- Use `FileUtils` media helpers for rich responses.
- Maintain full parity for platform document attributes and chat uploads.

**Error Handling & Lean Principles:**
- Text extraction always takes precedence (images can fail gracefully).
- Centralized validation, no `except: pass`, safe defaults, Pydantic everywhere.
- Minimal LOC: focus on dispatcher + PDF path first.
- No behavior change when flag is False.

## Phased TDD-First Roadmap (Consensus from Both Plans)

**Phase 0 — Current (This Document)**
- Research complete.
- Tests written first for desired behaviors (text fidelity, registration, MD link resolution, backward compat, platform/chat parity, edge cases).
- This consensus plan finalized.

**Phase 1 — PDF + Unified Dispatcher (Minimal Viable, High Impact)**
- Implement `tools/asset_extractor.py` and `AssetExtractionResult`.
- Add `extract_with_assets()` to `PDFUtils` (text via pymupdf4llm, images via fitz).
- Update `local_path_text.py` and `read_text_based_file`.
- Generate markdown with registry links + register all assets.
- Full test suite (TDD).

**Phase 2 — Office Documents**
- Extend dispatcher for MarkItDown + targeted image extraction (base64 parsing or direct libs — minimal footprint).
- Handle charts where appropriate.

**Phase 3 — Polish, Vision Synergy & Documentation**
- Integrate with vision tools, platform pipelines, rich content system.
- Update existing docs (`docs/20250920_PDF_IMPLEMENTATION_SUMMARY.md`, PDF skill, tool docstrings).
- Performance guardrails (image limits, lazy extraction).
- Ruff, mypy, full test coverage.

## Success Criteria (Impeccable)

- Zero behavior change for existing calls.
- First call delivers immediate usable markdown + optional registered asset references.
- LLM can seamlessly reference extracted assets in follow-up turns via logical names.
- Markdown contains clean, functional internal links based on registry.
- All assets are session-isolated, logically named, and reusable.
- Code remains super lean, modular, pythonic, fully tested, and compliant with every workspace rule.
- No context bloat — images remain references until deliberately used.

## Open Questions Resolved in This Consensus

- Unified helper vs per-format methods → **Unified dispatcher** (cleanest abstraction).
- Auto vs explicit → **Explicit only**.
- MD links → **Registry logical names only** (no system paths).
- Office priority → PDF-first (Phase 1), Office in Phase 2 after research validation.
- Immediate text → Explicitly required in all flows.

This document is the single source of truth for implementation. It represents the impeccable merging of both prior plans with all user feedback and research.

**Next Action (when implementation begins):** Follow TDD — write comprehensive tests first, then implement the dispatcher starting with the PDF path.

---
**Compiled as Consensus:** 2026-04-25  
**Purpose:** Definitive, actionable blueprint that eliminates ambiguity while preserving the lean spirit of the original plans.
