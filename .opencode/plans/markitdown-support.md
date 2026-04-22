# Plan: Add MarkItDown Support for DOCX, XLSX, PPTX, HTML

## Objective
Add MarkItDown library to `read_text_based_file` tool for handling Office documents (DOCX, XLSX, PPTX) and HTML files. Keep specialized tools for analytics use cases.

## Changes Overview

| File | Change |
|------|--------|
| `requirements.txt` | Add `markitdown[docx,xlsx,pptx]` |
| `tools/tools.py` | Add markitdown import, handle new formats, update docstring |
| `tools/tools.py` | Update `analyze_excel_file` docstring |

## Detailed Steps

---

### Step 1: Update `requirements.txt`

**Action:** Add markitdown with required extras

**File:** `D:\Repo\cmw-platform-agent\requirements.txt`

**Change:** Add new line after existing dependencies:
```
markitdown[docx,xlsx,pptx]
```

**Checkpoint:** Verify file has the new entry

---

### Step 2: Add MarkItDown Import

**Action:** Add conditional import of MarkItDown

**File:** `D:\Repo\cmw-platform-agent\tools\tools.py`

**Location:** Around line 120 (after PyMuPDF import block)

**Change:** Add after existing optional imports:
```python
try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    print("Warning: markitdown not available. Install with: pip install markitdown[docx,xlsx,pptx]")
```

**Checkpoint:** Import works without error (graceful fallback if not installed)

---

### Step 3: Update `read_text_based_file` Tool Signature

**Action:** Add `read_html_as_markdown` parameter

**File:** `D:\Repo\cmw-platform-agent\tools\tools.py`

**Location:** Line 801 (function definition)

**Current:**
```python
@tool
def read_text_based_file(file_reference: str, agent=None) -> str:
```

**Change to:**
```python
class ReadTextBasedFileSchema(BaseModel):
    """Input schema for read_text_based_file tool."""
    file_reference: str = Field(description="Filename, path, or URL to read")
    read_html_as_markdown: bool = Field(
        default=True,
        description="For HTML files only: if True (default), converts HTML to Markdown to save tokens and improve readability. Set to False only when full HTML structure is needed (e.g., extracting specific tags, attributes, or raw markup)."
    )

@tool(args_schema=ReadTextBasedFileSchema)
def read_text_based_file(file_reference: str, read_html_as_markdown: bool = True, agent=None) -> str:
```

**Checkpoint:** Tool accepts new parameter

---

### Step 4: Update `read_text_based_file` Docstring

**Action:** Expand docstring with clear format guidance

**File:** `D:\Repo\cmw-platform-agent\tools\tools.py`

**Location:** Lines 802-828 (current docstring)

**Change:** Replace existing docstring with comprehensive version that covers:

1. **Supported formats list** - Add DOCX, XLSX, PPTX, HTML to existing list
2. **HTML section** - Explain `read_html_as_markdown` parameter in detail
3. **XLSX section** - Explain when to use this tool vs `analyze_excel_file`
4. **DOCX/PPTX** - Simple note about Markdown extraction

**Docstring content:**
```python
"""
Read text-based files and return content as text.
This is the general-purpose text file reader for most formats.
Use this tool when you need file content - for specialized analysis, use dedicated tools.

Supported file types:
- Text files: .txt, .md, .log, .rtf
- Code files: .py, .js, .ts, .html, .htm, .css, .sql, .java, .cpp, .c, .php, .rb, .go
- Configuration files: .ini, .cfg, .conf, .env, .properties, .yaml, .yml
- Structured text: .json, .xml, .svg
- Web files: .html, .htm (see HTML options below)
- Documentation: .md, .rst, .tex
- Office documents: .docx, .xlsx, .pptx (converted to Markdown for clean text extraction)
- PDF files: .pdf (extracts text content as Markdown)

HTML-specific options:
- read_html_as_markdown (bool, default=True): Converts HTML to Markdown to save tokens
  and improve readability. Set to False only when you need the raw HTML structure
  (e.g., extracting specific tags, attributes, CSS, or JavaScript).

XLSX/Excel: When to use this tool vs analyze_excel_file?
- read_text_based_file: Use for simple text extraction - reads Excel content as Markdown
  tables. Good for understanding document structure and reading text content.
- analyze_excel_file: Use when you need data analysis, statistics, pandas operations,
  charts, or need to query/transform the data.

The tool automatically:
- Detects file encoding and handles multiple encodings (UTF-8, Latin-1, CP1252, ISO-8859-1)
- Resolves filenames to full file paths via agent's file registry
- Downloads files from URLs automatically
- Provides file metadata (name, size, encoding) in results

Args:
    file_reference (str): Original filename from user upload OR URL to download
    read_html_as_markdown (bool): For HTML files. If True (default), converts HTML to
        Markdown for token efficiency and readability. Set False only when raw HTML
        structure is needed.

Returns:
    str: The text content of the file with metadata, or an error message if reading fails
"""
```

**Checkpoint:** Docstring is comprehensive and guides agent decision-making

---

### Step 5: Add Format Handling Logic

**Action:** Add MarkItDown handling for DOCX, XLSX, PPTX, HTML

**File:** `D:\Repo\cmw-platform-agent\tools\tools.py`

**Location:** Around line 839 (after PDF handling block)

**Change:** Add new elif block after PDF check:

```python
    # MarkItDown handles DOCX, XLSX, PPTX, and HTML
    elif file_info.extension in ('.docx', '.xlsx', '.pptx', '.html'):
        if not MARKITDOWN_AVAILABLE:
            return FileUtils.create_tool_response(
                "read_text_based_file",
                error="markitdown not available. Install with: pip install markitdown[docx,xlsx,pptx]",
                file_info=file_info
            )
        try:
            md = MarkItDown()
            result = md.convert(file_path)
            content = result.text_content
            if not content or not content.strip():
                return FileUtils.create_tool_response(
                    "read_text_based_file",
                    error=f"No text content found in {file_info.extension} file",
                    file_info=file_info
                )
            # For HTML, check if we should keep raw or convert to markdown
            if file_info.extension == '.html' and not read_html_as_markdown:
                # Read as raw HTML - use FileUtils text reading
                raw_result = FileUtils.read_text_file(file_path)
                if raw_result.success:
                    content = raw_result.content
                    display_name = file_reference if file_reference.startswith(('http://', 'https://', 'ftp://')) else file_reference
                    size_str = FileUtils.format_file_size(file_info.size)
                    result_text = f"File: {display_name} ({size_str}, raw HTML)\n\nContent:\n{content}"
                else:
                    return FileUtils.create_tool_response(
                        "read_text_based_file",
                        error=f"Could not read HTML as raw text: {raw_result.error}",
                        file_info=file_info
                    )
            else:
                # HTML with markdown conversion (default) or other office formats
                display_name = file_reference if file_reference.startswith(('http://', 'https://', 'ftp://')) else file_reference
                size_str = FileUtils.format_file_size(file_info.size)
                if file_info.extension == '.html':
                    result_text = f"File: {display_name} ({size_str}, converted to Markdown)\n\nContent:\n{content}"
                else:
                    result_text = f"File: {display_name} ({size_str})\n\nContent:\n{content}"
            return FileUtils.create_tool_response(
                "read_text_based_file",
                result=result_text,
                file_info=file_info
            )
        except Exception as e:
            return FileUtils.create_tool_response(
                "read_text_based_file",
                error=f"Error processing {file_info.extension}: {str(e)}",
                file_info=file_info
            )
```

**Checkpoint:** All new formats handled, HTML respects `read_html_as_markdown` flag

---

### Step 6: Update `analyze_excel_file` Docstring

**Action:** Clarify when to use this tool vs `read_text_based_file`

**File:** `D:\Repo\cmw-platform-agent\tools\tools.py`

**Location:** Around line 1137 (analyze_excel_file definition)

**Change:** Add "When to use" guidance to existing docstring:

Add after the first sentence:
```
When to use this tool vs read_text_based_file?
- Use analyze_excel_file for: data analysis, statistics, pandas operations, charts,
  querying/transforming data, understanding numerical patterns.
- Use read_text_based_file for: simple text extraction, reading cell content as text,
  understanding document structure when you don't need analytics.
```

**Checkpoint:** Docstring clearly guides agent to correct tool

---

### Step 7: Verification

**Action:** Run linting and type checking

**Commands:**
```bash
ruff check tools/tools.py
mypy tools/tools.py --ignore-missing-imports
```

**Checkpoint:** No errors

---

## Rollback Plan (if needed)

If issues arise, revert:
1. Remove `markitdown[docx,xlsx,pptx]` from `requirements.txt`
2. Remove markitdown import block from `tools/tools.py`
3. Revert `read_text_based_file` signature to original
4. Revert docstrings to original
5. Remove the format handling elif block

---

## Dependencies

| Dependency | Purpose | Keep Existing? |
|-----------|---------|----------------|
| markitdown[docx,xlsx,pptx] | Office docs + HTML text extraction | N/A |
| pymupdf4llm | PDF extraction | ✅ Yes |
| openpyxl | Excel analytics | ✅ Yes (for analyze_excel_file) |
| pandas | Data analysis | ✅ Yes (for analyze_excel_file) |
| pytesseract | OCR | ✅ Yes |

---

## Testing Checklist

- [ ] Install markitdown: `pip install markitdown[docx,xlsx,pptx]`
- [ ] Test DOCX file reading
- [ ] Test XLSX file reading (Markdown output)
- [ ] Test PPTX file reading
- [ ] Test HTML reading with `read_html_as_markdown=True` (default)
- [ ] Test HTML reading with `read_html_as_markdown=False`
- [ ] Verify existing PDF reading still works
- [ ] Verify existing CSV/Excel analysis tools still work
- [ ] Run lint check
- [ ] Run type check
