# file_utils.py - Reusable file handling utilities
"""
Modular file handling utilities for tools.
Provides abstracted, reusable functions for common file operations.
"""

import os
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from pydantic import BaseModel, Field, field_validator

class FileInfo(BaseModel):
    """Pydantic model for file information."""
    exists: bool = Field(description="Whether the file exists and is accessible")
    path: Optional[str] = Field(None, description="Full file path")
    name: Optional[str] = Field(None, description="File name with extension")
    size: int = Field(0, description="File size in bytes")
    extension: str = Field("", description="File extension (lowercase)")
    error: Optional[str] = Field(None, description="Error message if file access failed")

    @field_validator('size')
    @classmethod
    def validate_size(cls, v):
        if v < 0:
            raise ValueError('File size cannot be negative')
        return v

class TextFileResult(BaseModel):
    """Pydantic model for text file reading results."""
    success: bool = Field(description="Whether the file was successfully read")
    content: Optional[str] = Field(None, description="File content as text")
    encoding: Optional[str] = Field(None, description="Encoding used to read the file")
    file_info: Optional[FileInfo] = Field(None, description="File information")
    error: Optional[str] = Field(None, description="Error message if reading failed")

class BinaryFileResult(BaseModel):
    """Pydantic model for binary file reading results."""
    success: bool = Field(description="Whether the file was successfully read")
    content: Optional[str] = Field(None, description="Base64 encoded file content")
    file_info: Optional[FileInfo] = Field(None, description="File information")
    error: Optional[str] = Field(None, description="Error message if reading failed")

class ToolResponse(BaseModel):
    """Pydantic model for standardized tool responses."""
    type: str = Field(default="tool_response", description="Response type identifier")
    tool_name: str = Field(description="Name of the tool that generated the response")
    result: Optional[str] = Field(None, description="Tool result content")
    error: Optional[str] = Field(None, description="Error message if tool failed")
    file_info: Optional[FileInfo] = Field(None, description="File information if applicable")
    extra: Optional[Dict[str, Any]] = Field(None, description="Optional structured payload for tool-specific data")

class FileUtils:
    """Utility class for common file operations."""

    @staticmethod
    def file_exists(file_path: str) -> bool:
        """Check if file exists and is accessible."""
        return os.path.exists(file_path) and os.path.isfile(file_path)

    @staticmethod
    def get_file_size(file_path: str) -> int:
        """Get file size in bytes."""
        try:
            return os.path.getsize(file_path)
        except OSError:
            return 0

    @staticmethod
    def get_file_info(file_path: str) -> FileInfo:
        """Get comprehensive file information with Pydantic validation."""
        if not FileUtils.file_exists(file_path):
            return FileInfo(
                exists=False,
                error=f"File not found: {file_path}"
            )

        try:
            return FileInfo(
                exists=True,
                path=file_path,
                name=os.path.basename(file_path),
                size=FileUtils.get_file_size(file_path),
                extension=Path(file_path).suffix.lower()
            )
        except Exception as e:
            return FileInfo(
                exists=False,
                error=f"Error getting file info: {str(e)}"
            )

    @staticmethod
    def read_text_file(file_path: str, encodings: List[str] = None) -> TextFileResult:
        """
        Read text file with multiple encoding fallback and Pydantic validation.

        Args:
            file_path: Path to the text file
            encodings: List of encodings to try (default: ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1'])

        Returns:
            TextFileResult with validated content, encoding used, and metadata
        """
        if encodings is None:
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        file_info = FileUtils.get_file_info(file_path)
        if not file_info.exists:
            return TextFileResult(
                success=False,
                error=file_info.error,
                file_info=file_info
            )

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()

                return TextFileResult(
                    success=True,
                    content=content,
                    encoding=encoding,
                    file_info=file_info
                )
            except UnicodeDecodeError:
                continue
            except Exception as e:
                return TextFileResult(
                    success=False,
                    error=f"Error reading file: {str(e)}",
                    file_info=file_info
                )

        return TextFileResult(
            success=False,
            error="File appears to be binary and cannot be read as text",
            file_info=file_info
        )

    @staticmethod
    def read_binary_file(file_path: str) -> BinaryFileResult:
        """Read binary file and return base64 encoded content with Pydantic validation."""
        file_info = FileUtils.get_file_info(file_path)
        if not file_info.exists:
            return BinaryFileResult(
                success=False,
                error=file_info.error,
                file_info=file_info
            )

        try:
            import base64
            with open(file_path, 'rb') as f:
                content = f.read()

            return BinaryFileResult(
                success=True,
                content=base64.b64encode(content).decode('utf-8'),
                file_info=file_info
            )
        except Exception as e:
            return BinaryFileResult(
                success=False,
                error=f"Error reading binary file: {str(e)}",
                file_info=file_info
            )

    @staticmethod
    def create_tool_response(tool_name: str, result: str = None, error: str = None, 
                           file_info: FileInfo = None, extra: Dict[str, Any] = None) -> str:
        """Create standardized tool response JSON with Pydantic validation."""
        # Sanitize file_info to remove full paths
        if file_info:
            # Create a sanitized copy without the full path
            sanitized_file_info = FileInfo(
                exists=file_info.exists,
                path=None,  # Remove full path for security
                name=file_info.name,
                size=file_info.size,
                extension=file_info.extension,
                error=file_info.error
            )
        else:
            sanitized_file_info = None

        response = ToolResponse(
            tool_name=tool_name,
            result=result,  # Full result, no truncation
            error=error,
            file_info=sanitized_file_info,
            extra=extra
        )

        return response.model_dump_json(indent=2)

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes == 0:
            return "0 bytes"
        elif size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes // 1024} KB"
        else:
            return f"{size_bytes // (1024 * 1024)} MB"

    @staticmethod
    def file_to_base64(file_path: str) -> str:
        """
        Convert file to base64 encoded string.

        Args:
            file_path (str): Path to the file to convert

        Returns:
            str: Base64 encoded file content

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file can't be read
        """
        import base64

        if not FileUtils.file_exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
            return base64.b64encode(file_content).decode('utf-8')
        except Exception as e:
            raise IOError(f"Error reading file {file_path}: {str(e)}")

    @staticmethod
    def download_file_to_path(url: str, target_path: str = None) -> str:
        """
        Download file from URL to local path.

        Args:
            url (str): URL to download from
            target_path (str, optional): Local path to save to. If None, creates temp file.

        Returns:
            str: Path to downloaded file

        Raises:
            requests.RequestException: If download fails
            IOError: If file can't be written
        """
        import requests
        import tempfile
        import os
        import logging
        from urllib.parse import urlparse

        logger = logging.getLogger(__name__)

        try:
            # Add polite bot identification headers
            headers = {
                'User-Agent': 'CMW-Platform-Agent/1.0 (+https://github.com/arterm-sedov/cmw-platform-agent) Mozilla/5.0'
            }
            
            # First make a HEAD request to get Content-Type
            logger.info(f"Attempting to download from URL: {url}")
            head_response = requests.head(url, headers=headers, timeout=30, allow_redirects=True)
            head_response.raise_for_status()
            content_type = head_response.headers.get('content-type', 'unknown')
            logger.info(f"HEAD request successful, Content-Type: {content_type}")

            if target_path is None:
                # Create temp file with proper extension
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path) or "downloaded_file"
                # Extract extension from URL
                _, url_ext = os.path.splitext(filename)

                # Get Content-Type header
                content_type = head_response.headers.get('content-type', '').lower()

                # MIME type to extension mapping
                mime_to_ext = {
                    # Documents
                    'application/pdf': '.pdf',
                    'application/msword': '.doc',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
                    'application/vnd.ms-excel': '.xls',
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
                    'application/vnd.ms-powerpoint': '.ppt',
                    'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
                    'application/rtf': '.rtf',
                    'application/zip': '.zip',
                    'application/x-zip-compressed': '.zip',

                    # Text formats
                    'text/plain': '.txt',
                    'text/html': '.html',
                    'text/css': '.css',
                    'text/javascript': '.js',
                    'text/csv': '.csv',
                    'text/xml': '.xml',
                    'application/json': '.json',
                    'application/xml': '.xml',

                    # Images
                    'image/jpeg': '.jpg',
                    'image/jpg': '.jpg',
                    'image/png': '.png',
                    'image/gif': '.gif',
                    'image/webp': '.webp',
                    'image/svg+xml': '.svg',
                    'image/bmp': '.bmp',
                    'image/tiff': '.tiff',

                    # Audio
                    'audio/mpeg': '.mp3',
                    'audio/wav': '.wav',
                    'audio/ogg': '.ogg',
                    'audio/mp4': '.m4a',

                    # Video
                    'video/mp4': '.mp4',
                    'video/avi': '.avi',
                    'video/quicktime': '.mov',
                    'video/x-msvideo': '.avi',
                }

                # Smart extension detection strategy:
                # 1. If Content-Type is specific and matches known types, use it
                # 2. If URL has a standard extension, use it
                # 3. Fallback to Content-Type if URL extension is non-standard

                ext = None
                content_type_ext = None
                url_ext_valid = False

                # Get extension from Content-Type
                for mime_type, extension in mime_to_ext.items():
                    if mime_type in content_type:
                        content_type_ext = extension
                        break

                # Check if URL extension is valid (standard file extension)
                if url_ext:
                    # Check if URL extension matches any known extension
                    known_extensions = set(mime_to_ext.values())
                    url_ext_valid = url_ext.lower() in known_extensions

                # Decision logic
                if content_type_ext and url_ext_valid:
                    # Both are valid - prefer Content-Type for accuracy
                    ext = content_type_ext
                elif content_type_ext and not url_ext_valid:
                    # Content-Type is valid, URL extension is not standard
                    ext = content_type_ext
                elif not content_type_ext and url_ext_valid:
                    # Only URL extension is valid
                    ext = url_ext
                elif url_ext:
                    # URL extension exists but not standard - use it as fallback
                    ext = url_ext
                else:
                    # No extension found
                    ext = ''

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                target_path = temp_file.name
                temp_file.close()

            # Now download the file
            logger.info(f"Starting download to: {target_path}")
            response = requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True)
            response.raise_for_status()

            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Download completed successfully: {target_path}")
            return target_path

        except Exception as e:
            raise IOError(f"Error downloading file from {url}: {str(e)}")

    @staticmethod
    def generate_unique_filename(original_filename: str, session_id: str = "default") -> str:
        """
        Generate a unique filename with timestamp and hash (no session prefix since we use session folders).

        Args:
            original_filename (str): Original filename from user upload
            session_id (str): Session ID for isolation (used for folder organization)

        Returns:
            str: Unique filename with timestamp and hash
        """
        import hashlib
        import time
        from pathlib import Path

        # Get file extension
        path_obj = Path(original_filename)
        name_without_ext = path_obj.stem
        extension = path_obj.suffix

        # Generate timestamp and hash (include session_id for uniqueness across sessions)
        timestamp = str(int(time.time() * 1000))  # milliseconds
        hash_suffix = hashlib.md5(f"{original_filename}{timestamp}{session_id}".encode()).hexdigest()[:8]

        # Create unique filename with session ID for better uniqueness and clarity
        unique_name = f"{session_id}_{name_without_ext}_{timestamp}_{hash_suffix}{extension}"

        return unique_name

    @staticmethod
    def get_gradio_cache_path() -> str:
        """
        Get the current Gradio cache directory path.

        Returns:
            str: Path to Gradio's cache directory
        """
        import os
        import tempfile

        # Check if GRADIO_TEMP_DIR is set
        gradio_temp = os.environ.get('GRADIO_TEMP_DIR')
        if gradio_temp:
            return gradio_temp

        # Default to system temp directory
        return tempfile.gettempdir()

    @staticmethod
    def resolve_file_reference(file_reference: str, agent=None) -> str:
        """
        Resolve file reference (filename or URL) to full file path.

        Args:
            file_reference (str): Original filename from user upload OR URL
            agent: Agent instance with file registry (optional)

        Returns:
            str: Full path to the file, or None if not found
        """
        # Check if it's a URL
        if file_reference.startswith(('http://', 'https://', 'ftp://')):
            try:
                # Download URL to temp file
                return FileUtils.download_file_to_path(file_reference)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to download URL {file_reference}: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                # Re-raise the exception to get more details
                raise

        # It's a filename - resolve using agent's file registry
        if agent and hasattr(agent, 'get_file_path'):
            return agent.get_file_path(file_reference)

        return None

    @staticmethod
    def resolve_file_path(original_filename: str, agent=None) -> str:
        """
        Resolve original filename to full file path using agent's file registry.

        Args:
            original_filename (str): Original filename from user upload
            agent: Agent instance with file registry (optional)

        Returns:
            str: Full path to the file, or None if not found
        """
        if agent and hasattr(agent, 'get_file_path'):
            return agent.get_file_path(original_filename)

        return None

    @staticmethod
    def resolve_code_input(code_reference: str, agent=None) -> tuple[str, str]:
        """
        Resolve code reference to actual code content and detected language.

        Args:
            code_reference (str): Code content, filename, or URL
            agent: Agent instance for file resolution (optional)

        Returns:
            tuple: (code_content, detected_language)
        """
        # Check if it's a URL
        if code_reference.startswith(('http://', 'https://', 'ftp://')):
            try:
                file_path = FileUtils.download_file_to_path(code_reference)
                result = FileUtils.read_text_file(file_path)
                if not result.success:
                    raise ValueError(f"Failed to read URL content: {result.error}")
                language = FileUtils.detect_language_from_extension(file_path)
                return result.content, language
            except Exception as e:
                raise ValueError(f"Failed to download URL {code_reference}: {str(e)}")

        # Check if it's a file path (try to resolve via agent first, then direct path)
        file_path = None
        if agent and hasattr(agent, 'get_file_path'):
            file_path = agent.get_file_path(code_reference)

        if not file_path and os.path.exists(code_reference):
            file_path = code_reference

        if file_path and os.path.exists(file_path):
            result = FileUtils.read_text_file(file_path)
            if not result.success:
                raise ValueError(f"Failed to read file: {result.error}")
            language = FileUtils.detect_language_from_extension(file_path)
            return result.content, language

        # It's code content - return as-is with no language detection
        return code_reference, None

    @staticmethod
    def detect_language_from_extension(file_path: str) -> str:
        """Detect programming language from file extension."""
        extension_map = {
            '.py': 'python',
            '.sh': 'bash', '.bash': 'bash',
            '.sql': 'sql',
            '.c': 'c', '.h': 'c',
            '.java': 'java',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp',
            '.cs': 'csharp',
            '.php': 'php',
            '.r': 'r',
            '.m': 'matlab',
            '.scala': 'scala',
            '.kt': 'kotlin',
            '.swift': 'swift'
        }
        return extension_map.get(Path(file_path).suffix.lower(), 'python')

    @staticmethod
    def is_text_file(file_path: str) -> bool:
        """Check if file is likely a text file based on extension."""
        text_extensions = {
            '.txt', '.md', '.log', '.json', '.xml', '.yaml', '.yml', 
            '.html', '.htm', '.css', '.js', '.py', '.sql', '.ini', 
            '.cfg', '.conf', '.env', '.csv', '.tsv'
        }
        return Path(file_path).suffix.lower() in text_extensions

    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """Check if file is likely an image file based on extension."""
        image_extensions = {
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.svg'
        }
        return Path(file_path).suffix.lower() in image_extensions

    @staticmethod
    def is_audio_file(file_path: str) -> bool:
        """Check if file is likely an audio file based on extension."""
        audio_extensions = {
            '.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'
        }
        return Path(file_path).suffix.lower() in audio_extensions

    @staticmethod
    def is_video_file(file_path: str) -> bool:
        """Check if file is likely a video file based on extension."""
        video_extensions = {
            '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv'
        }
        return Path(file_path).suffix.lower() in video_extensions

    @staticmethod
    def is_pdf_file(file_path: str) -> bool:
        """Check if file is likely a PDF file based on extension."""
        return Path(file_path).suffix.lower() == '.pdf'

    @staticmethod
    def get_mime_type(file_path: str) -> str:
        """Get MIME type for a file based on extension and content."""
        import mimetypes

        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            return mime_type

        ext = Path(file_path).suffix.lower()
        mime_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.svg': 'image/svg+xml',
            '.tiff': 'image/tiff',
            '.bmp': 'image/bmp',
            '.mp4': 'video/mp4',
            '.webm': 'video/webm',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.ogg': 'audio/ogg',
            '.flac': 'audio/flac',
            '.aac': 'audio/aac',
            '.m4a': 'audio/mp4',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.pdf': 'application/pdf'
        }

        return mime_map.get(ext, 'application/octet-stream')

    @staticmethod
    def detect_media_type(file_path: str) -> str:
        """Detect media type category for a file."""
        if FileUtils.is_image_file(file_path):
            return 'image'
        elif FileUtils.is_video_file(file_path):
            return 'video'
        elif FileUtils.is_audio_file(file_path):
            return 'audio'
        elif Path(file_path).suffix.lower() == '.html':
            return 'html'
        elif Path(file_path).suffix.lower() in ['.png', '.svg'] and 'plot' in file_path.lower():
            return 'plot'
        else:
            return 'unknown'

    @staticmethod
    def create_media_attachment(file_path: str, caption: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a media attachment dictionary for rich content.

        Args:
            file_path: Path to the media file
            caption: Optional caption for the media
            metadata: Optional metadata dictionary

        Returns:
            Dict with media attachment information
        """
        if not FileUtils.file_exists(file_path):
            return {
                "type": "error",
                "error": f"File not found: {file_path}"
            }

        file_info = FileUtils.get_file_info(file_path)
        media_type = FileUtils.detect_media_type(file_path)
        mime_type = FileUtils.get_mime_type(file_path)

        attachment = {
            "type": "media_attachment",
            "media_type": media_type,
            "file_path": file_path,
            "mime_type": mime_type,
            "file_info": file_info.dict() if file_info else None
        }

        if caption:
            attachment["caption"] = caption

        if metadata:
            attachment["metadata"] = metadata

        return attachment

    @staticmethod
    def add_media_to_response(tool_response: Dict[str, Any], file_path: str, 
                            caption: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Add media attachment to an existing tool response.

        Args:
            tool_response: Existing tool response dictionary
            file_path: Path to the media file
            caption: Optional caption for the media
            metadata: Optional metadata dictionary

        Returns:
            Updated tool response with media attachment
        """
        if "media_attachments" not in tool_response:
            tool_response["media_attachments"] = []

        media_attachment = FileUtils.create_media_attachment(file_path, caption, metadata)
        tool_response["media_attachments"].append(media_attachment)

        return tool_response

    @staticmethod
    def extract_media_from_response(tool_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract media attachments from a tool response.

        Args:
            tool_response: Tool response dictionary

        Returns:
            List of media attachment dictionaries
        """
        media_attachments = []

        if "media_attachments" in tool_response:
            media_attachments.extend(tool_response["media_attachments"])

        if "result" in tool_response and isinstance(tool_response["result"], dict):
            result = tool_response["result"]
            for key, value in result.items():
                if isinstance(value, str) and FileUtils.file_exists(value):
                    media_attachment = FileUtils.create_media_attachment(value, f"File: {key}")
                    media_attachments.append(media_attachment)

        return media_attachments

    @staticmethod
    def is_base64_image(data: str) -> bool:
        """Check if string contains base64 image data."""
        import base64

        if data.startswith('data:image/'):
            return True

        if len(data) > 100:
            try:
                clean_data = ''.join(data.split())
                decoded = base64.b64decode(clean_data)
                image_magic = [
                    b'\x89PNG\r\n\x1a\n',
                    b'\xff\xd8\xff',
                    b'GIF87a',
                    b'GIF89a',
                    b'RIFF',
                    b'BM'
                ]
                return any(decoded.startswith(magic) for magic in image_magic)
            except:
                return False

        return False

    @staticmethod
    def save_base64_to_file(base64_data: str, output_path: str = None, 
                          file_extension: str = None, session_id: str = None) -> str:
        """
        Save base64 data to a file.

        Args:
            base64_data: Base64 encoded data (with or without data URI prefix)
            output_path: Optional output file path
            file_extension: Optional file extension for temp file
            session_id: Optional session ID to save in session-isolated directory

        Returns:
            Path to the saved file
        """
        import base64
        import tempfile
        import uuid
        import mimetypes
        from datetime import datetime

        if base64_data.startswith('data:'):
            header, data = base64_data.split(',', 1)
            mime_type = header.split(':')[1].split(';')[0]
            if not file_extension:
                file_extension = mimetypes.guess_extension(mime_type) or '.bin'
        else:
            data = base64_data
            if not file_extension:
                file_extension = '.bin'

        if not output_path:
            if session_id:
                session_dir = Path(f".gradio/sessions/{session_id}")
                session_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                filename = f"llm_image_{timestamp}_{unique_id}{file_extension}"
                output_path = str(session_dir / filename)
            else:
                temp_fd, output_path = tempfile.mkstemp(suffix=file_extension)
                os.close(temp_fd)

        decoded_data = base64.b64decode(data)
        with open(output_path, 'wb') as f:
            f.write(decoded_data)
        return output_path

    @staticmethod
    def create_gallery_attachment(image_paths: List[str], captions: List[str] = None) -> Dict[str, Any]:
        """
        Create a gallery attachment for multiple images.

        Args:
            image_paths: List of image file paths
            captions: Optional list of captions for each image

        Returns:
            Gallery attachment dictionary
        """
        if not image_paths:
            return {"type": "error", "error": "No image paths provided"}

        valid_images = []
        for i, path in enumerate(image_paths):
            if FileUtils.file_exists(path) and FileUtils.is_image_file(path):
                image_info = {
                    "path": path,
                    "caption": captions[i] if captions and i < len(captions) else None
                }
                valid_images.append(image_info)

        if not valid_images:
            return {"type": "error", "error": "No valid image files found"}

        return {
            "type": "gallery_attachment",
            "media_type": "gallery",
            "images": valid_images,
            "count": len(valid_images)
        }