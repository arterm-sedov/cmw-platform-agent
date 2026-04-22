#!/usr/bin/env python3
"""
Tests for read_text_based_file tool with MarkItDown support (DOCX, XLSX, PPTX, HTML)
"""

import sys
import os
import json
import tempfile
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_test_file(extension: str, content: str = None) -> str:
    """Create a temporary test file with given extension and content."""
    fd, path = tempfile.mkstemp(suffix=extension)
    if content:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
    else:
        os.close(fd)
    return path


def parse_tool_response(response: str) -> dict:
    """Parse JSON tool response."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw": response[:200]}


class TestReadTextBasedFileMarkitdown:
    """Tests for MarkItDown-supported formats in read_text_based_file."""

    @staticmethod
    def test_read_html_as_markdown():
        """Test HTML reading with markdown conversion (default)."""
        from tools.tools import read_text_based_file

        html_content = """<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
<h1>Hello World</h1>
<p>This is a <strong>test</strong> paragraph.</p>
</body>
</html>"""

        path = create_test_file('.html', html_content)
        try:
            result = read_text_based_file.invoke({
                "file_reference": path,
                "read_html_as_markdown": True
            })
            parsed = parse_tool_response(result)

            assert parsed.get("success") is True, f"Expected success, got: {parsed}"
            assert "Hello World" in parsed.get("result", ""), "Expected markdown heading in result"
            assert "**test**" in parsed.get("result", "") or "<strong>test</strong>" not in parsed.get("result", ""), \
                "Expected markdown formatting or converted content"
            print("✅ test_read_html_as_markdown: PASSED")
            return True
        except Exception as e:
            print(f"❌ test_read_html_as_markdown: FAILED - {e}")
            return False
        finally:
            os.unlink(path)

    @staticmethod
    def test_read_html_raw():
        """Test HTML reading without markdown conversion."""
        from tools.tools import read_text_based_file

        html_content = """<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
<h1>Hello World</h1>
</body>
</html>"""

        path = create_test_file('.html', html_content)
        try:
            result = read_text_based_file.invoke({
                "file_reference": path,
                "read_html_as_markdown": False
            })
            parsed = parse_tool_response(result)

            assert parsed.get("success") is True, f"Expected success, got: {parsed}"
            assert "<html>" in parsed.get("result", ""), "Expected raw HTML in result"
            assert "raw HTML" in parsed.get("result", "").lower(), "Expected indication of raw HTML mode"
            print("✅ test_read_html_raw: PASSED")
            return True
        except Exception as e:
            print(f"❌ test_read_html_raw: FAILED - {e}")
            return False
        finally:
            os.unlink(path)

    @staticmethod
    def test_read_docx():
        """Test DOCX reading via MarkItDown."""
        try:
            from markitdown import MarkItDown
        except ImportError:
            print("⚠️  test_read_docx: SKIPPED (markitdown not installed)")
            return True

        from tools.tools import read_text_based_file

        # Create a minimal DOCX file (ZIP with Word document)
        import zipfile
        docx_path = create_test_file('.docx')
        with zipfile.ZipFile(docx_path, 'w') as zf:
            # Minimal document.xml content
            word_doc = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body><w:p><w:r><w:t>Hello from DOCX</w:t></w:r></w:p></w:body>
</w:document>'''
            zf.writestr('word/document.xml', word_doc)
            zf.writestr('[Content_Types].xml', '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="xml" ContentType="application/xml"/>
</Types>''')

        try:
            result = read_text_based_file.invoke({
                "file_reference": docx_path,
                "read_html_as_markdown": True
            })
            parsed = parse_tool_response(result)

            assert parsed.get("success") is True, f"Expected success, got: {parsed}"
            assert "DOCX" in parsed.get("result", "") or "Hello from DOCX" in parsed.get("result", ""), \
                "Expected DOCX content in result"
            print("✅ test_read_docx: PASSED")
            return True
        except Exception as e:
            print(f"❌ test_read_docx: FAILED - {e}")
            return False
        finally:
            os.unlink(docx_path)

    @staticmethod
    def test_read_xlsx():
        """Test XLSX reading via MarkItDown."""
        try:
            from markitdown import MarkItDown
        except ImportError:
            print("⚠️  test_read_xlsx: SKIPPED (markitdown not installed)")
            return True

        from tools.tools import read_text_based_file

        # Create a minimal XLSX file
        import zipfile
        xlsx_path = create_test_file('.xlsx')
        with zipfile.ZipFile(xlsx_path, 'w') as zf:
            # Minimal sheet1.xml content
            sheet_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetData>
<row><c r="A1"><v>Name</v></c><c r="B1"><v>Value</v></c></row>
<row><c r="A2"><v>Test</v></c><c r="B2"><v>42</v></c></row>
</sheetData>
</worksheet>'''
            zf.writestr('xl/worksheets/sheet1.xml', sheet_xml)
            zf.writestr('[Content_Types].xml', '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="xml" ContentType="application/xml"/>
</Types>''')

        try:
            result = read_text_based_file.invoke({
                "file_reference": xlsx_path,
                "read_html_as_markdown": True
            })
            parsed = parse_tool_response(result)

            assert parsed.get("success") is True, f"Expected success, got: {parsed}"
            assert "XLSX" in parsed.get("result", "") or "Name" in parsed.get("result", "") or "Test" in parsed.get("result", ""), \
                "Expected XLSX content in result"
            print("✅ test_read_xlsx: PASSED")
            return True
        except Exception as e:
            print(f"❌ test_read_xlsx: FAILED - {e}")
            return False
        finally:
            os.unlink(xlsx_path)

    @staticmethod
    def test_read_pptx():
        """Test PPTX reading via MarkItDown."""
        try:
            from markitdown import MarkItDown
        except ImportError:
            print("⚠️  test_read_pptx: SKIPPED (markitdown not installed)")
            return True

        from tools.tools import read_text_based_file

        # Create a minimal PPTX file
        import zipfile
        pptx_path = create_test_file('.pptx')
        with zipfile.ZipFile(pptx_path, 'w') as zf:
            # Minimal slide1.xml content
            slide_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<slides><sldId id="1" r:id="rId1"/></slides>
</p:presentation>'''
            slide_content = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<sp><p:txBody><a:p><a:r><a:t>Hello from PPTX</a:t></a:r></a:p></p:txBody></sp>
</p:sld>'''
            zf.writestr('ppt/presentation.xml', slide_xml)
            zf.writestr('ppt/slides/slide1.xml', slide_content)
            zf.writestr('[Content_Types].xml', '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="xml" ContentType="application/xml"/>
</Types>''')

        try:
            result = read_text_based_file.invoke({
                "file_reference": pptx_path,
                "read_html_as_markdown": True
            })
            parsed = parse_tool_response(result)

            assert parsed.get("success") is True, f"Expected success, got: {parsed}"
            assert "PPTX" in parsed.get("result", "") or "Hello from PPTX" in parsed.get("result", ""), \
                "Expected PPTX content in result"
            print("✅ test_read_pptx: PASSED")
            return True
        except Exception as e:
            print(f"❌ test_read_pptx: FAILED - {e}")
            return False
        finally:
            os.unlink(pptx_path)

    @staticmethod
    def test_markitdown_not_available():
        """Test graceful handling when markitdown is not available."""
        import sys
        import importlib

        from tools.tools import read_text_based_file

        # Create a minimal HTML file
        path = create_test_file('.html', '<html><body>Test</body></html>')

        # Mock markitdown as not available
        original_markitdown = sys.modules.get('markitdown')
        sys.modules['markitdown'] = None

        try:
            result = read_text_based_file.invoke({
                "file_reference": path,
                "read_html_as_markdown": True
            })
            parsed = parse_tool_response(result)

            # Should return error about markitdown not available
            assert "markitdown" in parsed.get("error", "").lower() or parsed.get("success") is False, \
                f"Expected error about markitdown, got: {parsed}"
            print("✅ test_markitdown_not_available: PASSED")
            return True
        except Exception as e:
            print(f"❌ test_markitdown_not_available: FAILED - {e}")
            return False
        finally:
            os.unlink(path)
            if original_markitdown:
                sys.modules['markitdown'] = original_markitdown
            elif 'markitdown' in sys.modules:
                del sys.modules['markitdown']


def run_all_tests():
    """Run all markitdown-related tests."""
    print("\n🧪 Running MarkItDown Support Tests...\n")

    tests = [
        TestReadTextBasedFileMarkitdown.test_read_html_as_markdown,
        TestReadTextBasedFileMarkitdown.test_read_html_raw,
        TestReadTextBasedFileMarkitdown.test_read_docx,
        TestReadTextBasedFileMarkitdown.test_read_xlsx,
        TestReadTextBasedFileMarkitdown.test_read_pptx,
        TestReadTextBasedFileMarkitdown.test_markitdown_not_available,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ {test.__name__}: EXCEPTION - {e}")
            results.append(False)

    passed = sum(1 for r in results if r)
    total = len(results)

    print(f"\n📊 Results: {passed}/{total} tests passed")
    return all(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
