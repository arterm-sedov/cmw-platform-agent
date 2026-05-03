# 20260425 Final TDD Lean Impeccable Asset Extraction Implementation Plan

**Date:** 2026-04-25  
**Status:** Final Definitive Implementation Blueprint (Consensus of All Five Plans)  
**Version:** 4.0 — Brilliant, Impeccable, Perfect Synthesis  
**Author:** CMW Platform Agent (full synthesis of all referenced plans + conversation + deep research)  
**References (All Read and Harmonized):**
- `20260425_pdf_image_extraction_plan.md` (original Phase 1 PDF + fitz, Phase 2 Office investigation)
- `20260425_unified_extract_assets_plan.md` (unified dispatcher, explicit flag, detailed naming, registry-first MD links)
- `20260425_impeccable_consensus_asset_extraction_plan.md` (TDD-first emphasis, VL as consumer, immediate text + references)
- `20260425_consolidated_asset_extraction_plan.md` (concise checklist-driven version with unified helper)
- `20260425_image_extraction_strategy_pdf_office.md` (deep research on fitz vs MarkItDown vs VL module — layered approach, PDF with fitz, Office as targeted layer on MarkItDown, VL for semantic understanding only)

**Core Mandate:** This is the **single source of truth** for implementation. It must be:
- **TDD-first** (tests written before any production code)
- **Super lean, DRY, perfect, abstract, modular, pythonic, pydantic**
- **Non-breaking** in every way (`extract_images=False` = zero behavior change)
- **Strictly follows AGENTS.md** (research-first, LangChain-pure where applicable, Ruff before commits, tests in `tools/_tests/` or `agent_ng/_tests/`, session isolation, no silent exceptions, centralized error handling, no overengineering)
- **Brilliant and Impeccable** — elegant, minimal, maximally reusable via existing registry and `FileUtils`

## Objective (Final Consensus)

Implement a unified asset extraction system for PDF and Office documents (DOCX, XLSX, PPTX) that:
1. Returns **rich markdown text immediately** on every call.
2. When `extract_images=True` (explicit only, default=False), additionally extracts images, saves both the full markdown and images to **session-isolated** locations, registers them with **logical human-readable names** in the agent's `file_registry`, and generates markdown containing **registry-friendly internal links**.
3. Enables the LLM to reuse registered assets in future turns via `read_text_based_file("report_extracted.md")` or vision tools without any duplication or context bloat.
4. Works identically for chat-attached files and platform-fetched document attributes.
5. Uses the VL module (`agent_ng/vision_input.py`, `analyze_image_ai`) **only as a consumer** for semantic understanding after extraction — never as the primary extractor.

## Final Architecture (Impeccable Synthesis)

**New Thin Abstraction (Single Responsibility):**
- `tools/asset_extractor.py` — contains:
  - `AssetExtractionResult` (Pydantic model with `success`, `text_content`, `markdown_path`, `image_paths`, `registered_refs`, `media_attachments`, `error`)
  - `def extract_assets(file_path: str, extract_images: bool = False, session_id: str | None = None, agent: Any = None) -> AssetExtractionResult`

**Dispatch Logic (Lean & Abstract):**
- `.pdf` → Enhanced `PDFUtils.extract_with_assets(...)`:
  - Text: Existing `pymupdf4llm.to_markdown(...)` (can optionally disable `ignore_images` for richer output).
  - Images: `fitz.Document` + `page.get_images()` / `doc.extract_image(xref)` (lean, deterministic, already a dependency).
- Office (`.docx`/`.xlsx`/`.pptx`) → MarkItDown for rich text/markdown + lightweight targeted image extraction layer (mammoth/python-docx for DOCX, python-pptx for PPTX, openpyxl for XLSX charts where meaningful). Base64 parsing from MarkItDown output as fallback.
- Always save full markdown as `{stem}_extracted.md`.
- Images saved as `{stem}_image_{n:02d}.{ext}` (PNG preferred for quality).
- Generate markdown with internal links using **logical registry names only** (no system paths ever).
- Register everything via `agent.register_file(logical_name, physical_path)` using existing session-isolated registry (`agent_ng/langchain_agent.py:194-418`).
- Return immediate text + `FileUtils.add_media_to_response(...)` + metadata.

**Tool Integration:**
- Extend `tools/tools.py:792` (`read_text_based_file`) with `extract_images: bool = False` in the Pydantic schema and function signature.
- Route through `tools/local_path_text.py` (update the PDF and Office branches to call the new dispatcher when flag is set).
- Use existing `FileUtils` primitives for responses, media attachments, filename generation, and resolution.

**VL Module Role (Resolved from Research):**
- Consumer only: After assets are registered, use `analyze_image_ai()` or `VisionInput` for semantic description/OCR of complex images/charts.
- Do **not** route bulk document image extraction through vision LLMs (too slow, non-deterministic, costly, against lean principles).

## TDD-First Implementation Mandate (Non-Negotiable)

**All implementation must follow this exact order:**

1. **Phase 0: Tests First** (in `tools/_tests/test_asset_extraction.py` or `agent_ng/_tests/test_document_assets.py`)
   - Backward compatibility (`extract_images=False` produces identical output to today).
   - Session isolation and correct registry population.
   - Generated markdown contains usable logical links that resolve via `resolve_file_reference`.
   - First call returns immediate text + references when flag=True (text file reference is always returned along with the direct text for future use by the agent, image references are returned when the flag=True).
   - Platform document attribute vs chat upload parity.
   - Edge cases: no images, many images, scanned PDFs (extracted as images in any event), Office charts, duplicate images (do not think we can actually identify duplicates at extraction time), errors (text succeeds even if images fail).
   - Registry round-trip (`get_file_path()` works for extracted names).

2. **Phase 1: PDF + Dispatcher** (highest ROI, minimal surface)
   - Implement `tools/asset_extractor.py` (dispatcher + Pydantic result).
   - Extend `tools/pdf_utils.py` with `extract_with_assets(...)` (builds directly on existing `extract_text_from_pdf` and `PDFTextResult`).
   - Update `local_path_text.py` and `read_text_based_file`.
   - Ensure markdown generation with registry links.
   - Run Ruff, mypy, all tests after every change.

3. **Phase 2: Office Layer**
   - Add Office dispatch in `asset_extractor.py`.
   - Targeted image extraction on top of MarkItDown (keep footprint minimal).
   - Full test coverage.

4. **Phase 3: Integration & Polish**
   - Vision synergy (`analyze_image_ai` on registered images).
   - Platform document pipeline parity.
   - Performance (configurable max images, lazy where sensible).
   - Documentation updates (PDF summary, tool docstrings, AGENTS.md if needed).
   - Final Ruff/lint pass on only modified files.

## Success Criteria (Brilliant & Impeccable)

- Zero regression — existing calls and text-only behavior unchanged.
- Explicit, lean, perfectly abstract (one dispatcher, clear separation of concerns).
- Immediate usable markdown on first call; images are registered references for future turns (no context bloat).
- Markdown contains clean, functional internal links using logical registry names only.
- All assets session-isolated, logically named, reusable via existing `FileUtils` and registry.
- TDD complete with high-quality behavior-focused tests (not implementation details).
- Passes Ruff (`ruff check` + `ruff format` on modified files only), mypy, all tests.
- Strictly follows every rule in AGENTS.md and workspace `.mdc` files (research-first, DRY, pythonic, pydantic, no silent exceptions, centralized error handling, progress reports with timestamp, tests in correct location).
- VL used intelligently as understanding layer, not primary extractor.
- The final system feels "perfect" — elegant, minimal, extensible, and a joy to use.

## Open Questions Resolved in This Final Plan

- Unified vs per-format → Single elegant dispatcher.
- Auto vs explicit → Explicit only.
- MD links → Pure logical registry names (no system paths).
- MarkItDown + images → Text-first + lightweight targeted layer.
- VL role → Consumer for semantic analysis after registration.
- Office priority → PDF-first (Phase 1), Office in Phase 2.

This document is the **final, authoritative implementation plan**. All previous plans have been harmonized into this one impeccable blueprint.

**Implementation begins only after tests are written (TDD).**

**Next Action (in Agent mode):** Begin with test file creation, then implement the dispatcher starting with the PDF path, maintaining perfect leanness and non-breaking changes throughout. Run Ruff after every edit on only the modified files.

---
**This is the Definitive Final Plan**  
**Compiled from all five referenced documents + full conversation**  
**Adheres strictly to "TDD, lean, dry, perfect, abstract, non-breaking, follows AGENTS.md, brilliant and impeccable"**
