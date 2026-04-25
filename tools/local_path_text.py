"""Read plain text from a local file path — shared by tools and platform document pipeline."""

from __future__ import annotations

from tools.file_utils import FileInfo, FileUtils

# Optional markitdown (same as tools.tools)
try:
    from markitdown import MarkItDown

    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False


def read_local_path_to_plain_text(
    file_path: str,
    *,
    read_html_as_markdown: bool = True,
    extract_images: bool = False,
    _file_info: FileInfo | None = None,
) -> tuple[str, str | None, str | None]:
    """
    Extract human-readable text from a local file.

    Args:
        file_path: Absolute path to the file.
        read_html_as_markdown: Convert HTML to Markdown (default True).
        extract_images: Extract embedded images from PDF/Office (default False).
        _file_info: Pre-computed FileInfo to avoid a redundant stat call.

    Returns:
        (text, error, encoding) — encoding is set only for plain text reads.
    """
    file_info = _file_info if _file_info is not None else FileUtils.get_file_info(file_path)
    if not file_info.exists:
        return "", file_info.error or "File not found", None

    ext = (file_info.extension or "").lower()

    if ext in (".md", ".markdown"):
        result = FileUtils.read_text_file(file_path)
        if not result.success:
            return "", result.error or "Markdown read failed", None
        return result.content or "", None, result.encoding

    if ext in (".mp4", ".webm", ".mkv", ".mov", ".m4v"):
        return (
            "",
            "Video: not converted to text here. In chat, attach the file for a "
            "multimodal/vision model or a transcript tool.",
            None,
        )

    if ext in (".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac"):
        return (
            "",
            "Audio: not transcribed here; use speech-to-text or a multimodal model if the "
            "chat stack supports it.",
            None,
        )

    if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"):
        if not MARKITDOWN_AVAILABLE:
            return (
                "",
                "Image: install markitdown to try extraction; in chat, attach the image for a "
                "vision-capable model.",
                None,
            )
        try:
            md = MarkItDown()
            result = md.convert(file_path)
            t = (result.text_content or "").strip()
            if t:
                return str(result.text_content), None, None
            return (
                "[Image: no text layer; use vision/chat image attachment to describe the file.]",
                None,
                None,
            )
        except Exception as e:
            return "", f"Error processing image: {e!s}", None

    if ext == ".pdf":
        try:
            from tools.asset_extractor import extract_assets  # noqa: PLC0415
            from tools.pdf_utils import PDFUtils  # noqa: PLC0415

            if not PDFUtils.is_available():
                return (
                    "",
                    "PyMuPDF not available. Install with: pip install pymupdf",
                    None,
                )
            if extract_images:
                asset_result = extract_assets(file_path, extract_images=True)
                if not asset_result.success:
                    return "", asset_result.error_message or "PDF extraction failed", None
                content = asset_result.markdown_content or asset_result.text_content
            else:
                pdf_result = PDFUtils.extract_text_from_pdf(file_path, use_markdown=True)
                if not pdf_result.success:
                    return "", pdf_result.error_message or "PDF extraction failed", None
                content = pdf_result.text_content
            return content, None, None
        except Exception as e:
            return "", f"Error processing PDF: {e!s}", None

    if ext == ".bin":
        # Opaque filename from a server (e.g. content-disposition); avoid reading bytes as text.
        try:
            with open(file_path, "rb") as f:
                head = f.read(32)
        except OSError as e:
            return "", str(e), None
        if len(head) >= 12 and head[4:8] == b"ftyp":
            return (
                "",
                "Video: not converted to text here. In chat, attach the file for a "
                "multimodal/vision model or a transcript tool.",
                None,
            )
        return (
            "",
            "Binary file with no recognized extension; no text extraction.",
            None,
        )

    if ext in (".docx", ".xlsx", ".pptx", ".html"):
        if not MARKITDOWN_AVAILABLE:
            return (
                "",
                "markitdown not available. Install with: pip install markitdown[docx,xlsx,pptx]",
                None,
            )
        try:
            if extract_images:
                from tools.asset_extractor import extract_assets  # noqa: PLC0415
                asset_result = extract_assets(file_path, extract_images=True)
                if not asset_result.success:
                    return "", asset_result.error_message or f"{ext} extraction failed", None
                content = asset_result.markdown_content or asset_result.text_content
            else:
                md = MarkItDown()
                result = md.convert(file_path)
                content = result.text_content
                if not content or not str(content).strip():
                    return "", f"No text content found in {ext} file", None
            if ext == ".html" and not read_html_as_markdown:
                raw = FileUtils.read_text_file(file_path)
                if raw.success and raw.content is not None:
                    return raw.content, None, raw.encoding
                return "", raw.error or "Could not read HTML as raw text", None
            return str(content), None, None
        except Exception as e:
            return "", f"Error processing {ext}: {e!s}", None

    result = FileUtils.read_text_file(file_path)
    if not result.success:
        return "", result.error or "Text read failed", None
    return result.content or "", None, result.encoding
