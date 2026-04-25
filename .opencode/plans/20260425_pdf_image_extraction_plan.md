# Plan: Add Image Extraction to PDF and Office Document Processing

## Objective
Enhance the existing document processing pipeline to extract images from PDF and Office documents (DOCX, XLSX, PPTX) alongside text extraction, saving them as session-isolated files and registering them with the agent's file registry for future access.

## Current State Analysis

### PDF Processing (`tools/pdf_utils.py`)
- Uses PyMuPDF4LLM for text extraction only
- Explicitly sets `ignore_images=True` and `ignore_graphics=True`
- Returns text content as markdown string
- No image extraction capability

### Office Document Processing (`tools/local_path_text.py`)
- Uses MarkItDown for DOCX, XLSX, PPTX conversion
- MarkItDown focuses on text/markdown conversion, image handling unclear
- Current implementation returns only text content

### File Handling Infrastructure
- Session-isolated file registry in `agent_ng/langchain_agent.py`
- FileUtils provides session-aware filename generation and file operations
- `local_path_text.py` serves as central routing for file type processing
- `read_text_based_file` tool in `tools/tools.py` exposes file reading to agents

## Enhancement Strategy

### 1. PDF Image Extraction
Add image extraction capability to `PDFUtils` using PyMuPDF/fitz directly:
- Extract images using `doc.extract_image(xref)` method
- Handle image masks/transparency properly
- Save images as PNG/JPEG with appropriate extensions
- Generate session-isolated filenames using `FileUtils.generate_unique_filename()`

### 2. Office Document Image Extraction
Investigate and implement image extraction for Office docs:
- For MarkItDown: Check if it supports image extraction or if we need direct library usage
- For DOCX: Use python-docx to extract embedded images
- For XLSX: Use openpyxl to extract images from worksheets
- For PPTX: Use python-pptx to extract images from slides

### 3. Integration Points
Modify the processing pipeline to:
- Extract both text and images when requested
- Save extracted assets to session-isolated locations
- Register both text files and image files with agent's file registry
- Return file references that agents can use to access the content later

### 4. API Design
Add optional parameters to control image extraction:
- `extract_images: bool = False` - Whether to extract images alongside text
- `image_format: str = "PNG"` - Preferred image format for extraction
- Return enhanced response containing both text file reference and image file references

## Implementation Plan

### Phase 1: PDF Image Extraction
1. Enhance `PDFUtils` class with image extraction methods
2. Add `extract_pdf_with_images()` method that returns both text and image data
3. Update `local_path_text.py` to optionally use image extraction for PDFs
4. Modify `read_text_based_file` tool to accept image extraction parameters

### Phase 2: Office Document Image Extraction
1. Research MarkItDown's image capabilities
2. If insufficient, add direct library support for DOCX/XLSX/PPTX
3. Integrate with existing processing flow
4. Update tool signatures and documentation

### Phase 3: Testing and Validation
1. Ensure backward compatibility with existing text-only extraction
2. Test session isolation and file registration
3. Verify that extracted files can be accessed via existing file utilities
4. Test with various document types and image formats

## Detailed Implementation Approach

### PDFUtils Enhancements
```python
# In pdf_utils.py
class PDFUtils:
    # Existing methods...
    
    @staticmethod
    def extract_pdf_with_images(file_path: str, session_id: str = "default") -> dict:
        """
        Extract text as markdown AND images from PDF.
        
        Returns:
            {
                'text_path': str,  # Path to saved markdown file
                'image_paths': List[str],  # Paths to saved image files
                'text_content': str,  # Extracted markdown text
                'success': bool,
                'error_message': Optional[str]
            }
        """
        # 1. Extract text using existing PyMuPDF4LLM (with images ignored)
        # 2. Extract images using PyMuPDF/fitz directly
        # 3. Save text to session-isolated markdown file
        # 4. Save images to session-isolated image files
        # 5. Return file paths and content
```

### local_path_text.py Modifications
```python
# Add new helper function or modify existing PDF handling
def read_local_path_to_plain_text_with_images(
    file_path: str,
    *,
    extract_images: bool = False,
    session_id: str = "default",
    _file_info: FileInfo | None = None,
) -> tuple[str, str | None, str | None, dict]:
    """
    Extract text and optionally images from a local file.
    
    Returns:
        (text, error, encoding, image_data)
        image_data: {'text_path': str, 'image_paths': List[str]} or None
    """
    # Handle PDF case with image extraction option
    # Handle Office documents similarly
    # Fall back to existing behavior for other types
```

### Tool Integration
```python
# In tools/tools.py - Update read_text_based_file signature
class ReadTextBasedFileSchema(BaseModel):
    file_reference: str = Field(description="Filename, path, or URL to read")
    read_html_as_markdown: bool = Field(
        default=True,
        description="For HTML files only: if True (default), converts HTML to Markdown..."
    )
    extract_images: bool = Field(
        default=False,
        description="Whether to extract images from PDF and Office documents alongside text"
    )

@tool(args_schema=ReadTextBasedFileSchema)
def read_text_based_file(
    file_reference: str, 
    read_html_as_markdown: bool = True,
    extract_images: bool = False,
    agent=None
) -> str:
    # Pass extract_images and session_id to local_path_text functions
    # Process returned image data to register files with agent
    # Return enhanced response with file references
```

## File Registration and Access
1. Use `FileUtils.generate_unique_filename(original_filename, session_id)` for all saved files
2. Call `agent.register_file(display_name, file_path)` for each extracted file
3. Maintain session isolation throughout the process
4. Ensure extracted files follow same access patterns as existing registered files

## Dependencies
- Already have: PyMuPDF4LLM (for PDF text)
- Need to add: fitz (PyMuPDF) for PDF image extraction - comes with PyMuPDF4LLM
- For Office docs: python-docx, openpyxl, python-pptx (may need to add)
- MarkItDown already present for Office document text conversion

## Error Handling and Fallbacks
- Graceful degradation if image extraction libraries unavailable
- Clear error messages when image extraction fails but text succeeds
- Preserve existing behavior when `extract_images=False`

## Performance Considerations
- Image extraction only performed when explicitly requested
- Efficient image saving (avoid unnecessary conversions)
- Proper resource cleanup (close document handles)
- Session-isolated storage prevents cross-user contamination

## Security
- Maintains existing session isolation protections
- No changes to file access permissions or agent context handling
- Image files subject to same session-based access controls as text files

## Testing Strategy
1. Unit test PDF image extraction with various image formats
2. Test Office document image extraction (if implemented)
3. Verify session isolation - files only accessible within same session
4. Test file registration and retrieval via existing file utilities
5. Ensure backward compatibility - existing code unchanged when extract_images=False
6. Test error conditions (corrupted images, missing libraries, etc.)

## Open Questions for Investigation
1. Does MarkItDown extract images from Office documents, or do we need direct library usage?
2. What image formats should we support/save as (PNG, JPEG, original format)?
3. Should we extract image metadata (dimensions, DPI, etc.) alongside the binary data?
4. How should we handle duplicate images (same image referenced multiple times in PDF)?
5. What naming convention should we use for extracted images to ensure uniqueness and traceability?

## Next Steps
1. Proceed with PDF image extraction implementation (Phase 1)
2. Investigate Office document image handling during implementation
3. Iteratively enhance based on findings
4. Maintain lean, minimal implementation approach consistent with existing codebase