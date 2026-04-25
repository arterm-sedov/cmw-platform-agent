#!/usr/bin/env python3
"""
TDD tests for analyze_image_ai PDF handling.

Tests that PDFs are properly rejected with helpful guidance.
"""

import sys
import os
import tempfile
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def create_test_pdf() -> str:
    """Create a minimal test PDF."""
    try:
        import fitz
    except ImportError:
        return ""

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 50), "Test", fontsize=12)
    doc.save(path)
    doc.close()

    return path


def safe_unlink(path: str) -> None:
    """Safely unlink a file."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except (PermissionError, OSError):
        pass


class TestAnalyzeImageAIPDFHandling:
    """Tests for analyze_image_ai PDF rejection."""

    @staticmethod
    def test_pdf_rejected_with_helpful_message():
        """PDF files should be rejected with guidance."""
        from tools.tools import analyze_image_ai

        pdf_path = create_test_pdf()
        if not pdf_path:
            print("⚠️  SKIPPED (fitz not available)")
            return

        try:
            result = analyze_image_ai.invoke({
                "file_reference": pdf_path,
                "prompt": "What is in this image?"
            })
            result_parsed = json.loads(result)

            assert result_parsed.get("error"), "Expected error for PDF"
            error_msg = result_parsed.get("error", "")

            assert "read_text_based_file" in error_msg, "Expected guidance to use read_text_based_file"
            assert "extract_images" in error_msg, "Expected guidance to use extract_images"

            print("✅ test_pdf_rejected_with_helpful_message: PASSED")
        except Exception as e:
            print(f"❌ test_pdf_rejected_with_helpful_message: FAILED - {e}")
        finally:
            safe_unlink(pdf_path)

    @staticmethod
    def test_pdf_with_uppercase_extension():
        """PDF extension case-insensitive."""
        from tools.tools import analyze_image_ai
        import json

        pdf_path = create_test_pdf()
        if not pdf_path:
            print("⚠️  SKIPPED (fitz not available)")
            return

        try:
            renamed = pdf_path.replace(".pdf", ".PDF")
            os.rename(pdf_path, renamed)

            result = analyze_image_ai.invoke({
                "file_reference": renamed,
                "prompt": "What is in this image?"
            })
            result_parsed = json.loads(result)

            assert result_parsed.get("error"), "Expected error for .PDF"
            print("✅ test_pdf_with_uppercase_extension: PASSED")
        except Exception as e:
            print(f"❌ test_pdf_with_uppercase_extension: FAILED - {e}")
        finally:
            safe_unlink(renamed)


import json


def run_all_tests():
    """Run all TDD tests."""
    print("\n" + "=" * 60)
    print("TDD: analyze_image_ai PDF Handling Tests")
    print("=" * 60)

    test_class = TestAnalyzeImageAIPDFHandling

    print(f"\n📋 {test_class.__name__}")
    print("-" * 40)

    for method_name in dir(test_class):
        if method_name.startswith("test_"):
            method = getattr(test_class, method_name)
            method()

    print("\n" + "=" * 60)
    print("Run complete")


if __name__ == "__main__":
    run_all_tests()