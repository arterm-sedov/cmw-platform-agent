# 20260425 Image Extraction Strategy for PDF and Office Documents

**Date:** 2026-04-25  
**Status:** Research Synthesis & Strategy Plan  
**Version:** 1.0 (Dedicated Image-Focused Plan)  
**Author:** CMW Platform Agent  
**Purpose:** Standalone reference synthesizing **all prior plans** (`20260425_pdf_image_extraction_plan.md`, `20260425_unified_extract_assets_plan.md`, `20260425_consolidated_asset_extraction_plan.md`, `20260425_impeccable_consensus_asset_extraction_plan.md`) + deep research on fitz/PyMuPDF, MarkItDown, the VL module, and current codebase patterns.

**References** (all read and synthesized):
- Colleague's original plan (PDF-first with `fitz`, Office via direct libs or MarkItDown).
- Unified, Consolidated, and Impeccable Consensus plans (explicit `extract_images=True`, unified dispatcher in `tools/asset_extractor.py`, logical naming `{stem}_extracted.md` + `{stem}_image_{n:02d}.{ext}`, registry-first MD links, immediate text on first call, session isolation, TDD-first, lean principles).
- Full conversation thread (explicit-only flag, preference for layered approach over heavy VL-as-OCR, no context bloat, reuse of `FileUtils` + registry + `local_path_text.py` routing).
- Workspace rules (research-first, super-dry/lean/modular/pythonic/pydantic, no overengineering, explicit error handling, tests in `_tests/`, timestamped plans).

## Executive Summary of Findings

After deep analysis:

1. **fitz (PyMuPDF)** is the correct low-level tool for **PDF image extraction**. It is already a dependency (`PyMuPDF==1.26.5`), imported in `tools/tools.py:114`, and used by `pymupdf4llm`. Direct `fitz.Document.extract_image(xref)` or page `get_images()` is lean (~10-15 LOC) and precise. Current `pdf_utils.py` deliberately ignores images (`ignore_images=True`) to stay text-focused — this is the exact gap to close.

2. **MarkItDown** (used in `local_path_text.py:120-140` for DOCX/XLSX/PPTX/HTML) is **text-first**. It excels at Markdown conversion (tables, headings, some alt-text) but does **not** reliably extract binary images as separate files. It prefers LLM-based descriptions/OCR (via optional `llm_client` or `markitdown-ocr` plugin). Binary image extraction requires additional targeted logic (mammoth for DOCX, python-pptx, openpyxl, or base64 parsing from output). This confirms the **layered approach** is superior.

3. **VL Module (`agent_ng/vision_input.py`)** is **not suitable as a primary OCR plugin** for MarkItDown. It is an input abstraction for feeding media to vision LLMs (`VisionInput`, `MediaType.PDF/IMAGE`, `create_vision_input`). It is excellent for **semantic analysis** of already-extracted images (`analyze_image_ai`), but routing bulk document images through LLM calls would be slow, expensive, non-deterministic, and against lean principles. Use it as a **consumer** of registered assets, not the extractor.

**Consensus Recommendation (synthesized from all plans):**  
**Layered + Unified Dispatcher**. 
- Use a single `extract_assets(file_path, extract_images=True, session_id=None, agent=None)` entrypoint (as proposed in unified/consensus plans).
- Dispatch by format: PDF uses enhanced `PDFUtils` + `fitz`; Office uses MarkItDown for text + lightweight targeted extractors.
- Always return text/markdown immediately.
- When images are requested: save to session-isolated paths, register with logical names via `agent.register_file()`, generate MD with registry-friendly internal links, return `media_attachments` via `FileUtils`.
- Default remains pure text (zero behavior change).

This satisfies every prior plan while resolving open questions (VL role, MarkItDown limitations, explicit flag, naming, immediate text + reusable references).

## Detailed Strategy for PDF Images

**Primary Tool:** `fitz` (PyMuPDF)
- **Text path:** Keep existing `pymupdf4llm.to_markdown(...)` (can optionally set `ignore_images=False` for richer output with image placeholders).
- **Image path:** 
  - `doc = fitz.open(path)`
  - Iterate `page.get_images()` or use xref-based `doc.extract_image(xref)` for embedded images.
  - Handle masks, transparency, different formats (PNG preferred for quality).
  - Save with logical names: `{stem}_image_{02d}.png`.
- **Markdown enrichment:** Insert `![Description (page N, xref M)]({logical_image_name})` using registered names only.
- **Registration:** Use existing `FileUtils.generate_unique_filename(..., session_id)` + `agent.register_file(logical_name, physical_path)`.
- **Integration point:** Extend `PDFUtils` with `extract_with_assets(...)` (builds directly on current `extract_text_from_pdf` and `PDFTextResult`).

**Advantages:** Fast, deterministic, no LLM cost, full control. Matches all plans' PDF recommendations.

## Detailed Strategy for Office Files (DOCX, XLSX, PPTX)

**Primary Tool for Text:** MarkItDown (current implementation in `local_path_text.py`).
**For Images:** Layered targeted extraction (do **not** rely on VL module as primary OCR).

Recommended techniques (minimal footprint):
- **DOCX:** Use `mammoth` or `python-docx` to extract embedded images (common pattern from MarkItDown discussions).
- **PPTX:** `python-pptx` to iterate slides and extract image parts.
- **XLSX:** `openpyxl` for worksheet images/charts (convert charts to images where meaningful).
- **Fallback/Enhancement:** If MarkItDown output contains base64 URIs (`keep_data_uris=True`), parse and save them. Use VL module only for semantic description of complex charts after extraction.

**Dispatcher Logic (in `asset_extractor.py`):**
- If `extract_images=False`: current fast path (MarkItDown + PyMuPDF4LLM).
- If `True`: text via MarkItDown + image extraction layer → save/register both.

This avoids turning MarkItDown into a heavy vision pipeline and keeps the system lean.

## VL Module Role (Clarified from Research)

- **Best used for:** Post-extraction semantic analysis (`analyze_image_ai("registered_image_03.png", prompt=...)` or feeding `VisionInput`).
- **Not recommended as primary OCR plugin** for bulk document processing — too heavyweight, non-deterministic, and costly compared to `fitz` + local libs.
- It complements the strategy perfectly as the "understanding layer" after assets are registered.

## Overall Architecture (Consensus Synthesis)

1. **New thin module:** `tools/asset_extractor.py` — single responsibility dispatcher returning `AssetExtractionResult` (Pydantic).
2. **Update path:** `read_text_based_file(extract_images=False)` → `local_path_text.py` → dispatcher (or direct for PDF).
3. **Registration & Response:** Always register extracted `.md` and images. Use `FileUtils.add_media_to_response()` and logical names in output.
4. **MD Links:** Purely logical registry names — LLM uses `read_text_based_file("...")` or vision tools on them.
5. **Phasing (from all plans):** 
   - Phase 1: PDF + dispatcher (highest ROI).
   - Phase 2: Office targeted extraction.
   - Phase 3: VL integration, performance limits, full tests, docs.

**TDD-First:** Tests must cover:
- Backward compatibility (`extract_images=False`).
- Session isolation and registry correctness.
- MD contains usable logical links.
- First call returns immediate text + references.
- Platform document vs chat upload parity.
- Edge cases (no images, many images, scanned PDFs, Office charts).

## Success Criteria (Merged from All Plans)

- Zero regression on existing text-only behavior.
- Explicit, lean, reusable via registry (no context bloat).
- Immediate usable markdown on first call; images as registered references for future turns.
- Clean internal links in saved `.md` using logical names only.
- Full compliance with AGENTS.md (research-first, TDD, lean, pydantic, Ruff, session isolation, no silent exceptions).
- VL used for understanding, not primary extraction.

This dedicated image strategy document serves as the focused companion to the broader asset extraction plans. It resolves how to handle images specifically while staying true to the "super lean" ethos and all prior consensus.

**Next Action (when implementation begins):** Follow TDD — write tests for image extraction behaviors first, then implement the PDF path in the unified dispatcher.

---
**Compiled:** 2026-04-25  
**Status:** Definitive image-specific strategy synthesizing all referenced plans and deep research on fitz, MarkItDown, and the VL module.
