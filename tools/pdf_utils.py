# pdf_utils.py - Minimal PDF processing utilities
"""
Minimal PDF processing utilities for text extraction.
Uses PyMuPDF4LLM - the best library for LLM processing.
"""

import os
from typing import Optional, List, Any
from pydantic import BaseModel, Field


class PDFTextResult(BaseModel):
    """Result of PDF text extraction using Pydantic validation"""
    success: bool = Field(description="Whether extraction was successful")
    text_content: str = Field(default="", description="Extracted text content")
    page_count: int = Field(default=0, description="Number of pages processed")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


def _get_pymupdf4llm():
    """Lazy import of PyMuPDF4LLM - the best library for LLM processing"""
    try:
        import pymupdf4llm
        return pymupdf4llm
    except ImportError:
        return None


class PDFUtils:
    """Minimal PDF processing utilities with graceful fallbacks"""
    
    @staticmethod
    def is_pdf_file(file_path: str) -> bool:
        """Check if file is a valid PDF by examining header"""
        try:
            return (os.path.exists(file_path) and 
                   os.path.splitext(file_path)[1].lower() == '.pdf' and
                   open(file_path, 'rb').read(8).startswith(b'%PDF'))
        except Exception:
            return False
    
    @staticmethod
    def extract_text_from_pdf(file_path: str, use_markdown: bool = True) -> PDFTextResult:
        """Extract text from PDF using PyMuPDF4LLM - the best library for LLM processing"""
        if not PDFUtils.is_pdf_file(file_path):
            return PDFTextResult(success=False, error_message="File is not a valid PDF")
        
        pymupdf4llm = _get_pymupdf4llm()
        if not pymupdf4llm:
            return PDFTextResult(
                success=False,
                error_message="PyMuPDF4LLM not available. Install with: pip install pymupdf4llm"
            )
        
        try:
            # Use optimal parameters for LLM processing
            markdown_text = pymupdf4llm.to_markdown(
                file_path,
                detect_bg_color=True,  # Better text detection
                ignore_alpha=False,    # Include transparent text
                ignore_images=True,    # Ignore images for text focus
                ignore_graphics=True,  # Ignore graphics for text focus
                margins=0,            # Full page content
                page_chunks=False     # Single markdown string
            )
            return PDFTextResult(
                success=True,
                text_content=markdown_text,
                page_count=1  # PyMuPDF4LLM processes all pages as one
            )
        except Exception as e:
            return PDFTextResult(
                success=False,
                error_message=f"Error processing PDF: {str(e)}"
            )
    
    @staticmethod
    def is_available() -> bool:
        """Check if PyMuPDF4LLM is available"""
        return _get_pymupdf4llm() is not None
    
    @staticmethod
    def get_markdown_text(file_path: str) -> str:
        """Get PDF content as Markdown using PyMuPDF4LLM"""
        if not PDFUtils.is_pdf_file(file_path) or not (pymupdf4llm := _get_pymupdf4llm()):
            return ""
        try:
            return pymupdf4llm.to_markdown(
                file_path,
                detect_bg_color=True,  # Better text detection
                ignore_alpha=False,    # Include transparent text
                ignore_images=True,    # Ignore images for text focus
                ignore_graphics=True,  # Ignore graphics for text focus
                margins=0,            # Full page content
                page_chunks=False     # Single markdown string
            )
        except Exception:
            return ""
    
    @staticmethod
    def get_page_chunks(file_path: str) -> List[dict]:
        """Get PDF content as page chunks using PyMuPDF4LLM"""
        if not PDFUtils.is_pdf_file(file_path) or not (pymupdf4llm := _get_pymupdf4llm()):
            return []
        try:
            return pymupdf4llm.to_markdown(
                file_path,
                detect_bg_color=True,
                ignore_alpha=False,
                ignore_images=True,    # Ignore images for text focus
                ignore_graphics=True,  # Ignore graphics for text focus
                margins=0,
                page_chunks=True  # Return page chunks instead of single string
            )
        except Exception:
            return []


# Minimal convenience functions
def extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF file, returns empty string on failure"""
    return PDFUtils.extract_text_from_pdf(file_path).text_content


def is_pdf_file(file_path: str) -> bool:
    """Check if file is a PDF"""
    return PDFUtils.is_pdf_file(file_path)
