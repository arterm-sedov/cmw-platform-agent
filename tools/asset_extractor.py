# asset_extractor.py - Unified asset extraction for PDF and Office documents
# ruff: noqa: PLC0415
"""
Unified asset extraction dispatcher for PDF and Office documents.

Features:
- Text extraction with optional image extraction
- Session-isolated file handling
- Registry integration for LLM-reusable assets
- Backward compatible: extract_images=False preserves current behavior
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AssetExtractionResult(BaseModel):
    """Result of asset extraction with Pydantic validation."""
    success: bool = Field(default=False, description="Whether extraction was successful")
    text_content: str = Field(default="", description="Extracted text content")
    markdown_content: str = Field(default="", description="Generated markdown with image links")
    image_paths: list[str] = Field(default_factory=list, description="Extracted image file paths")
    error_message: str | None = Field(default=None, description="Error message if failed")
    file_format: str = Field(default="", description="Detected file format (.pdf, .docx, etc.)")


def _get_file_extension(file_path: str) -> str:
    """Get lowercase file extension."""
    return Path(file_path).suffix.lower()


def extract_with_images_fit(
    file_path: str,
    session_id: str | None = None,
    agent: Any = None,
) -> AssetExtractionResult:
    """
    Extract images from PDF using PyMuPDF (fitz).

    Args:
        file_path: Path to PDF file.
        session_id: Optional session ID for isolation.
        agent: Optional agent instance for registry.

    Returns:
        AssetExtractionResult with extracted images.
    """
    result = AssetExtractionResult(file_format=".pdf")

    if not os.path.exists(file_path):
        result.success = False
        result.error_message = f"File not found: {file_path}"
        return result

    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        result.success = False
        result.error_message = f"PyMuPDF not available: {e}"
        return result

    try:
        doc = fitz.open(file_path)
        image_paths: list[str] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_ext = base_image.get("ext", "png")
                image_data = base_image.get("image")

                if not image_data:
                    continue

                original_stem = Path(file_path).stem
                image_filename = f"{original_stem}_p{page_num + 1}_img{img_index + 1}.{image_ext}"

                if session_id and agent:
                    image_filename = f"{session_id}_{image_filename}"

                temp_fd, temp_path = None
                try:
                    import tempfile

                    temp_fd, temp_path = tempfile.mkstemp(suffix=f".{image_ext}")
                    os.close(temp_fd)

                    with open(temp_path, "wb") as f:
                        f.write(image_data)

                    image_paths.append(temp_path)

                except Exception as img_error:
                    logger.warning("Failed to extract image: %s", img_error)
                    continue

        result.image_paths = image_paths
        result.success = True
        doc.close()

    except Exception as e:
        result.success = False
        result.error_message = f"Error extracting images: {e}"

    return result


def extract_with_markitdown(
    file_path: str,
    extract_images: bool = False,
) -> AssetExtractionResult:
    """
    Extract text from Office documents using MarkItDown.

    Args:
        file_path: Path to Office file.
        extract_images: Whether to extract images (optional).

    Returns:
        AssetExtractionResult with extracted text.
    """
    result = AssetExtractionResult(file_format=_get_file_extension(file_path))

    if not os.path.exists(file_path):
        result.success = False
        result.error_message = f"File not found: {file_path}"
        return result

    markitdown = _get_markitdown()
    if not markitdown:
        result.success = False
        result.error_message = "MarkItDown not available. Install with: pip install markitdown"
        return result

    try:
        md = markitdown.MarkItDown()
        extract_result = md.convert(file_path)

        result.text_content = extract_result.text_content or ""
        result.success = True
        result.markdown_content = result.text_content

    except Exception as e:
        result.success = False
        result.error_message = f"Error processing Office file: {e}"

    return result


def _get_markitdown():
    """Lazy import of MarkItDown."""
    try:
        import markitdown

        return markitdown
    except ImportError:
        return None


def _get_pymupdf4llm():
    """Lazy import of PyMuPDF4LLM."""
    try:
        import pymupdf4llm

        return pymupdf4llm
    except ImportError:
        return None


def extract_assets(
    file_path: str,
    extract_images: bool = False,
    session_id: str | None = None,
    agent: Any = None,
) -> AssetExtractionResult:
    """
    Unified asset extraction dispatcher.

    Extracts text and optionally images from PDF and Office documents.
    Backward compatible: extract_images=False produces identical output to current behavior.

    Args:
        file_path: Path to the file to process.
        extract_images: Whether to extract embedded images (default False).
        session_id: Optional session ID for file isolation.
        agent: Optional agent instance for registry integration.

    Returns:
        AssetExtractionResult with extracted text and optional images.
    """
    result = AssetExtractionResult()

    if not file_path or not os.path.exists(file_path):
        result.success = False
        result.error_message = f"File not found: {file_path}"
        return result

    ext = _get_file_extension(file_path)

    if ext == ".pdf":
        return _extract_pdf_assets(file_path, extract_images, session_id, agent)

    if ext in (".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt"):
        return extract_with_markitdown(file_path, extract_images)

    if ext in (".html", ".htm"):
        result.file_format = ext
        return extract_with_markitdown(file_path, extract_images)

    result.success = False
    result.error_message = f"Unsupported file format: {ext}"
    return result


def _extract_pdf_assets(
    file_path: str,
    extract_images: bool,
    session_id: str | None,
    agent: Any,
) -> AssetExtractionResult:
    """
    Extract assets from PDF file.

    Args:
        file_path: Path to PDF file.
        extract_images: Whether to extract images.
        session_id: Optional session ID for file isolation.
        agent: Optional agent instance.

    Returns:
        AssetExtractionResult with extracted content.
    """
    from tools.pdf_utils import PDFUtils

    result = AssetExtractionResult(file_format=".pdf")

    if not os.path.exists(file_path):
        result.success = False
        result.error_message = f"PDF file not found: {file_path}"
        return result

    if not PDFUtils.is_available():
        result.success = False
        result.error_message = "PyMuPDF4LLM not available. Install with: pip install pymupdf4llm"
        return result

    try:
        pdf_result = PDFUtils.extract_text_from_pdf(file_path, use_markdown=True)

        if not pdf_result.success:
            result.success = False
            result.error_message = pdf_result.error_message
            return result

        result.text_content = pdf_result.text_content
        result.success = True
        result.markdown_content = pdf_result.text_content

        if extract_images:
            image_result = extract_with_images_fit(file_path, session_id, agent)
            result.image_paths = image_result.image_paths

            if result.image_paths:
                for idx, img_path in enumerate(result.image_paths):
                    result.markdown_content += f"\n![Figure {idx + 1}]({Path(img_path).name})"

    except Exception as e:
        result.success = False
        result.error_message = f"Error processing PDF: {e}"
        logger.exception("PDF asset extraction failed for %s", file_path)

    return result
