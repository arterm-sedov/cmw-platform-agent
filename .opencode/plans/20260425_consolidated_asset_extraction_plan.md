# 20260425 Consolidated Asset Extraction Plan
## Unified Image and Text Extraction for PDF and Office Documents

**Date:** 2026-04-25  
**Status:** Planning / Pre-Implementation  
**Author:** Consolidated from colleague's unified plan and codebase analysis  

## Objective
Create a unified, explicit, backward-compatible asset extraction system that:
1. Returns text/markdown immediately on first call
2. Optionally extracts and saves images as session-isolated files  
3. Registers all assets in the agent's file_registry with logical names
4. Generates markdown with internal links using registered references
5. Works for both chat-attached and platform-fetched documents

## Core Design

### Explicit-Only Semantics
- `extract_images: bool = False` (default preserves current behavior)
- LLM or caller must explicitly opt-in to avoid wasteful extraction

### Unified Helper/Dispatcher
New module: `tools/asset_extractor.py`
```python
def extract_assets(
    file_path: str,
    extract_images: bool = False,
    session_id: str | None = None,
    agent: Any = None
) -> AssetExtractionResult:  # Pydantic model
```
Dispatches internally:
- `.pdf` → Enhanced PDF processing (text via pymupdf4llm + images via fitz)
- Office (`.docx`/`.xlsx`/`.pptx`) → MarkItDown for text + targeted image extraction
- Returns structured result with text content, file paths, and registered references

### Naming Convention & Internal Links
- Logical names: `{original_stem}_extracted.md`, `{original_stem}_image_{n:02d}.{ext}`
- Generated markdown contains registry-friendly links: `![Figure 1](report_image_01.png)`
- No absolute paths in output - logical names resolve via existing file registry
- Saved `.md` registered so `read_text_based_file("report_extracted.md")` works in follow-ups

## Technical Implementation

### 1. Enhanced PDF Processing (`tools/pdf_utils.py`)
Add method: `extract_pdf_with_assets(file_path, session_id) -> dict`
- Text extraction: Existing PyMuPDF4LLM call (text focus)
- Image extraction: PyMuPDF/fitz `doc.extract_image(xref)` for binary images
- Handle image masks/transparency properly
- Save images as appropriate format (PNG/JPEG) with session-isolated names
- Generate markdown with internal image links pointing to registered files

### 2. Office Document Processing (`tools/local_path_text.py`)
Investigate MarkItDown capabilities:
- If MarkItDown doesn't extract binary images, add targeted extraction:
  - DOCX: python-docx for embedded images
  - XLSX: openpyxl for worksheet/charts images  
  - PPTX: python-pptx for slide images
- Fall back to MarkItDown for text/markdown conversion
- Apply same session-isolated saving and registry registration

### 3. File Handling Infrastructure (Reuse 100%)
- Session isolation: `file_registry[(session_id, logical_name)]` in `langchain_agent.py`
- FileUtils: `generate_unique_filename()`, `resolve_file_reference()`, media helpers
- Tool layer: Enhanced `read_text_based_file` in `tools/tools.py`

### 4. Tool Integration
Update `tools/tools.py`:
- Add `extract_images: bool = False` parameter to `ReadTextBasedFileSchema`
- Pass through to processing pipeline
- Process returned asset data:
  - Save text/markdown to session-isolated file
  - Register with agent: `agent.register_file(logical_name, file_path)`
  - Register each extracted image similarly
  - Build response with `FileUtils.create_tool_response()` including media attachments
  - Return immediate text content + metadata for registered assets

## Phased Implementation (TDD-First)

### Phase 0: Preparation
- Write tests for desired behaviors in `tools/_tests/test_asset_extraction.py`
- Finalize this plan

### Phase 1: PDF Only (Minimal Viable)
1. Implement `extract_pdf_with_assets()` in `pdf_utils.py`
2. Update `local_path_text.py` to call enhanced PDF processing when flag set
3. Modify `read_text_based_file` tool to accept and pass through `extract_images` parameter
4. Implement file saving, registration, and response enhancement
5. Test: backward compatibility, session isolation, first-call references, re-read of extracted markdown

### Phase 2: Office Documents
1. Add dispatch for MarkItDown + targeted image extraction
2. Handle charts in XLSX/PPTX as images where appropriate
3. Apply same saving/registration/logic as PDF path
4. Test Office document asset extraction

### Phase 3: Polish & Integration
1. Vision tool synergy and platform document pipeline parity
2. Performance considerations (configurable limits, lazy extraction if needed)
3. Documentation updates and UI hints if beneficial

## Success Criteria
- [ ] Existing calls unchanged when `extract_images=False`
- [ ] First call: immediate rich text + registered asset references in response
- [ ] LLM can reference assets via logical names: "analyze image_03 from report"
- [ ] Markdown contains usable `[]()`` links using registered logical names
- [ ] All files session-isolated, registered, and reusable via existing tools
- [ ] Passes Ruff, mypy, and all tests
- [ ] No context bloat - images are references until explicitly used
- [ ] Follows lean, DRY, modular principles from AGENTS.md

## Alignment with Workspace Rules
- [ ] Research-first (based on MarkItDown and PyMuPDF documentation)
- [ ] TDD/SDD (tests written before implementation)
- [ ] Lean/DRY/modular (single unified dispatcher)
- [ ] Explicit flag with no silent failures, Pydantic results
- [ ] Session isolation via existing registry mechanisms
- [ ] Progress reports in `.opencode/plans/` with YYYYMMDD_ prefix
- [ ] Zero code written in this plan phase (read-only compliance)

## Open Questions for Refinement
1. Should we auto-filter small/icon images below a size threshold?
2. Office image format: extract original vs. convert to standard format (PNG)?
3. Extract image metadata (dimensions, DPI, color profile) alongside binary data?
4. Handle duplicate images (same xref referenced multiple locations) efficiently?
5. Integrate with CMW record attachment tools via registered asset names?

## Next Steps (Post-Planning)
1. Implementation mode: Write TDD tests first for PDF asset extraction behaviors
2. Implement unified helper starting with PDF support
3. Iteratively enhance based on testing and Office document findings
4. Maintain strict adherence to "super lean" principles throughout