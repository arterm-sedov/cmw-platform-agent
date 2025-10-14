"""
Content Converter Module
========================

Handles automatic detection and conversion of rich content types to Gradio components.
Supports images, plots, videos, galleries, audio, and HTML content with automatic
detection from file paths, base64 data, and URLs.

Key Features:
- Automatic content type detection
- Base64 to component conversion
- File path to component conversion
- Mixed content support (markdown + media)
- MIME type detection and validation
- Temporary file management
- Backward compatibility with markdown-only content

Usage:
    from content_converter import ContentConverter


    converter = ContentConverter()
    rich_content = converter.convert_content("Here's an image: /path/to/image.png")
    # Returns: ["Here's an image: ", gr.Image(value="/path/to/image.png")]
"""

import os
import re
import base64
import tempfile
import mimetypes
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import gradio as gr

# Import FileUtils for file operations
try:
    from tools.file_utils import FileUtils
except ImportError:
    # Fallback for when running as script
    FileUtils = None


class ContentType(Enum):
    """Supported content types for conversion"""
    MARKDOWN = "markdown"
    IMAGE = "image"
    PLOT = "plot"
    VIDEO = "video"
    AUDIO = "audio"
    GALLERY = "gallery"
    HTML = "html"
    FILE_PATH = "file_path"
    BASE64 = "base64"
    URL = "url"
    UNKNOWN = "unknown"


@dataclass
class ConversionResult:
    """Result of content conversion"""
    content_type: ContentType
    component: Optional[gr.Component] = None
    file_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ContentConverter:
    """Converts various content types to appropriate Gradio components"""


    def __init__(self, session_id: str = None):
        """Initialize the content converter
        
        Args:
            session_id: Optional session ID for session-isolated file storage
        """
        self.session_id = session_id
        self.supported_image_formats = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.tiff', '.bmp'}
        self.supported_video_formats = {'.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        self.supported_audio_formats = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac', '.wma'}
        self.supported_plot_formats = {'.png', '.svg', '.html', '.json'}


        # Base64 image magic numbers for detection
        self.base64_magic_numbers = {
            b'\x89PNG\r\n\x1a\n': 'image/png',
            b'\xff\xd8\xff': 'image/jpeg',
            b'GIF87a': 'image/gif',
            b'GIF89a': 'image/gif',
            b'RIFF': 'image/webp',  # WebP starts with RIFF
            b'BM': 'image/bmp',
        }


        # Temporary files for cleanup
        self.temp_files = []


    def convert_content(self, content: Union[str, dict, gr.Component]) -> Union[str, dict, gr.Component, List[Union[str, gr.Component]]]:
        """
        Convert content to appropriate format for Gradio chatbot.


        Args:
            content: Content to convert (string, dict, or Gradio component)


        Returns:
            Converted content suitable for Gradio chatbot
        """
        if isinstance(content, gr.Component):
            # Already a Gradio component
            return content


        if isinstance(content, dict):
            # Handle dict content (tool responses, structured data)
            return self._convert_dict_content(content)


        if isinstance(content, str):
            # Handle string content
            return self._convert_string_content(content)


        # Fallback: return as-is
        return content


    def _convert_dict_content(self, content: dict) -> Union[dict, List[Union[str, gr.Component]]]:
        """Convert dictionary content (tool responses, structured data)"""
        # Check if this is a tool response with rich content
        if self._is_tool_response(content):
            return self._extract_rich_content_from_tool_response(content)


        # Check for file paths in dict values
        rich_items = []
        markdown_items = []


        for key, value in content.items():
            if isinstance(value, str):
                # Check if value is a file path or base64
                conversion_result = self._detect_and_convert_string(value)
                if conversion_result.component:
                    rich_items.append(conversion_result.component)
                else:
                    markdown_items.append(f"**{key}**: {value}")
            else:
                markdown_items.append(f"**{key}**: {value}")


        if rich_items:
            # Return mixed content
            return markdown_items + rich_items
        else:
            # Return as markdown
            return "\n".join(markdown_items)


    def _convert_string_content(self, content: str) -> Union[str, List[Union[str, gr.Component]]]:
        """Convert string content"""
        # Check for mixed content (markdown + file paths)
        mixed_content = self._extract_mixed_content(content)
        if mixed_content:
            return mixed_content


        # Check if entire string is a file path or base64
        conversion_result = self._detect_and_convert_string(content)
        if conversion_result.component:
            return conversion_result.component


        # Return as markdown
        return content


    def _extract_mixed_content(self, content: str) -> Optional[List[Union[str, gr.Component]]]:
        """Extract mixed content from string (markdown + file paths)"""
        # Look for file paths in the content
        file_path_pattern = r'([A-Za-z]:[\\/][^\s]+|[\\/][^\s]+\.(png|jpg|jpeg|gif|webp|svg|tiff|bmp|mp4|webm|avi|mov|wav|mp3|m4a|ogg|flac|aac|html))'
        matches = re.finditer(file_path_pattern, content, re.IGNORECASE)


        if not matches:
            return None


        mixed_content = []
        last_end = 0


        for match in matches:
            # Add text before the file path
            if match.start() > last_end:
                text_part = content[last_end:match.start()].strip()
                if text_part:
                    mixed_content.append(text_part)


            # Convert file path to component
            file_path = match.group(1)
            conversion_result = self._detect_and_convert_string(file_path)
            if conversion_result.component:
                mixed_content.append(conversion_result.component)


            last_end = match.end()


        # Add remaining text
        if last_end < len(content):
            text_part = content[last_end:].strip()
            if text_part:
                mixed_content.append(text_part)


        return mixed_content if mixed_content else None


    def _detect_and_convert_string(self, content: str) -> ConversionResult:
        """Detect content type and convert string to component"""
        content = content.strip()


        # Check for base64 data
        if self._is_base64_data(content):
            return self._convert_base64_to_component(content)


        # Check for file path
        if self._is_file_path(content):
            return self._convert_file_path_to_component(content)


        # Check for URL
        if self._is_url(content):
            return self._convert_url_to_component(content)


        # Return as markdown
        return ConversionResult(ContentType.MARKDOWN, None, None, None, None)


    def _is_base64_data(self, content: str) -> bool:
        """Check if content is base64 data"""
        # Check for data URI
        if content.startswith('data:'):
            return True


        # Check for raw base64 (common patterns)
        if len(content) > 100 and self._is_base64_string(content):
            # Try to decode and check magic numbers
            try:
                decoded = base64.b64decode(content)
                for magic, mime_type in self.base64_magic_numbers.items():
                    if decoded.startswith(magic):
                        return True
            except:
                pass


        return False


    def _is_base64_string(self, content: str) -> bool:
        """Check if string is valid base64"""
        try:
            # Remove whitespace and newlines
            clean_content = re.sub(r'\s+', '', content)
            # Check if it's valid base64
            base64.b64decode(clean_content, validate=True)
            return True
        except:
            return False


    def _is_file_path(self, content: str) -> bool:
        """Check if content is a file path"""
        # Check for absolute paths (Windows and Unix)
        if (content.startswith('/') or 
            (len(content) > 1 and content[1] == ':' and content[2] in ['\\', '/'])):
            return os.path.exists(content)


        # Check for relative paths with extensions
        if '.' in content and any(content.lower().endswith(ext) for ext in 
                                self.supported_image_formats | 
                                self.supported_video_formats | 
                                self.supported_audio_formats |
                                self.supported_plot_formats):
            return os.path.exists(content)


        return False


    def _is_url(self, content: str) -> bool:
        """Check if content is a URL"""
        url_pattern = r'^https?://[^\s]+\.(png|jpg|jpeg|gif|webp|svg|tiff|bmp|mp4|webm|avi|mov|wav|mp3|m4a|ogg|flac|aac|html)$'
        return bool(re.match(url_pattern, content, re.IGNORECASE))


    def _convert_base64_to_component(self, base64_data: str) -> ConversionResult:
        """Convert base64 data to appropriate Gradio component"""
        try:
            # Handle data URI
            if base64_data.startswith('data:'):
                mime_type, data = base64_data.split(',', 1)
                mime_type = mime_type.split(':')[1].split(';')[0]
            else:
                # Raw base64 - try to detect MIME type
                try:
                    decoded = base64.b64decode(base64_data)
                    mime_type = self._detect_mime_type_from_bytes(decoded)
                except:
                    return ConversionResult(ContentType.UNKNOWN, None, None, None, "Invalid base64 data")


            # Create temporary file
            temp_file = self._create_temp_file_from_base64(base64_data, mime_type)
            if not temp_file:
                return ConversionResult(ContentType.UNKNOWN, None, None, None, "Failed to create temporary file")


            # Convert to appropriate component based on MIME type
            if mime_type.startswith('image/'):
                return ConversionResult(ContentType.IMAGE, gr.Image(value=temp_file), temp_file, {"mime_type": mime_type})
            elif mime_type.startswith('video/'):
                return ConversionResult(ContentType.VIDEO, gr.Video(value=temp_file), temp_file, {"mime_type": mime_type})
            elif mime_type.startswith('audio/'):
                return ConversionResult(ContentType.AUDIO, gr.Audio(value=temp_file), temp_file, {"mime_type": mime_type})
            else:
                return ConversionResult(ContentType.UNKNOWN, None, temp_file, {"mime_type": mime_type}, f"Unsupported MIME type: {mime_type}")


        except Exception as e:
            return ConversionResult(ContentType.UNKNOWN, None, None, None, f"Base64 conversion error: {str(e)}")


    def _convert_file_path_to_component(self, file_path: str) -> ConversionResult:
        """Convert file path to appropriate Gradio component"""
        try:
            if not os.path.exists(file_path):
                return ConversionResult(ContentType.UNKNOWN, None, None, None, f"File not found: {file_path}")


            # Get file extension
            ext = Path(file_path).suffix.lower()


            # Determine component type based on extension
            if ext in self.supported_image_formats:
                return ConversionResult(ContentType.IMAGE, gr.Image(value=file_path), file_path, {"extension": ext})
            elif ext in self.supported_video_formats:
                return ConversionResult(ContentType.VIDEO, gr.Video(value=file_path), file_path, {"extension": ext})
            elif ext in self.supported_audio_formats:
                return ConversionResult(ContentType.AUDIO, gr.Audio(value=file_path), file_path, {"extension": ext})
            elif ext == '.html':
                return ConversionResult(ContentType.HTML, gr.HTML(value=file_path), file_path, {"extension": ext})
            else:
                return ConversionResult(ContentType.UNKNOWN, None, file_path, {"extension": ext}, f"Unsupported file type: {ext}")


        except Exception as e:
            return ConversionResult(ContentType.UNKNOWN, None, None, None, f"File path conversion error: {str(e)}")


    def _convert_url_to_component(self, url: str) -> ConversionResult:
        """Convert URL to appropriate Gradio component"""
        try:
            # For now, treat URLs as file paths (Gradio will handle the download)
            # This could be enhanced to download and cache URLs
            return ConversionResult(ContentType.URL, gr.Image(value=url), url, {"url": url})
        except Exception as e:
            return ConversionResult(ContentType.UNKNOWN, None, None, None, f"URL conversion error: {str(e)}")


    def _detect_mime_type_from_bytes(self, data: bytes) -> str:
        """Detect MIME type from byte data"""
        for magic, mime_type in self.base64_magic_numbers.items():
            if data.startswith(magic):
                return mime_type


        # Fallback to generic binary
        return 'application/octet-stream'


    def _create_temp_file_from_base64(self, base64_data: str, mime_type: str) -> Optional[str]:
        """Create file from base64 data in session-isolated storage"""
        try:
            # Determine file extension from MIME type
            ext = mimetypes.guess_extension(mime_type) or '.bin'

            # Use FileUtils to save base64 to session directory if session_id available
            if FileUtils:
                file_path = FileUtils.save_base64_to_file(
                    base64_data=base64_data,
                    file_extension=ext,
                    session_id=self.session_id
                )
                
                # Only track for cleanup if it's a temp file (no session_id)
                if not self.session_id:
                    self.temp_files.append(file_path)
                
                return file_path
            else:
                # Fallback if FileUtils not available
                if base64_data.startswith('data:'):
                    data = base64_data.split(',', 1)[1]
                else:
                    data = base64_data

                decoded = base64.b64decode(data)
                temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
                with os.fdopen(temp_fd, 'wb') as f:
                    f.write(decoded)

                self.temp_files.append(temp_path)
                return temp_path

        except Exception as e:
            print(f"Error creating file from base64: {e}")
            return None


    def _is_tool_response(self, content: dict) -> bool:
        """Check if content is a tool response"""
        return (isinstance(content, dict) and 
                'type' in content and 
                content.get('type') == 'tool_response')


    def _extract_rich_content_from_tool_response(self, tool_response: dict) -> List[Union[str, gr.Component]]:
        """Extract rich content from tool response"""
        rich_content = []


        # Extract text content
        if 'result' in tool_response and isinstance(tool_response['result'], dict):
            result = tool_response['result']


            # Look for file paths and base64 data in result
            for key, value in result.items():
                if isinstance(value, str):
                    conversion_result = self._detect_and_convert_string(value)
                    if conversion_result.component:
                        rich_content.append(conversion_result.component)
                    else:
                        # Add as text with key
                        rich_content.append(f"**{key}**: {value}")
                else:
                    rich_content.append(f"**{key}**: {value}")


        # Add error if present
        if 'error' in tool_response:
            rich_content.append(f"**Error**: {tool_response['error']}")


        return rich_content


    def detect_content_type(self, content: str) -> ContentType:
        """Detect the type of content"""
        if self._is_base64_data(content):
            return ContentType.BASE64
        elif self._is_file_path(content):
            return ContentType.FILE_PATH
        elif self._is_url(content):
            return ContentType.URL
        else:
            return ContentType.MARKDOWN


    def cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Error cleaning up temp file {temp_file}: {e}")


        self.temp_files.clear()


    def get_stats(self) -> Dict[str, Any]:
        """Get converter statistics"""
        return {
            "supported_image_formats": list(self.supported_image_formats),
            "supported_video_formats": list(self.supported_video_formats),
            "supported_audio_formats": list(self.supported_audio_formats),
            "supported_plot_formats": list(self.supported_plot_formats),
            "temp_files_count": len(self.temp_files),
            "base64_magic_numbers": len(self.base64_magic_numbers)
        }


# Global converter instance
_content_converter = None

def get_content_converter(session_id: str = None) -> ContentConverter:
    """Get a content converter instance
    
    Args:
        session_id: Optional session ID for session-isolated file storage
        
    Returns:
        ContentConverter instance
    """
    # Create new instance with session_id for proper session isolation
    return ContentConverter(session_id=session_id)

def reset_content_converter():
    """Reset the global content converter instance"""
    global _content_converter
    if _content_converter:
        _content_converter.cleanup_temp_files()
    _content_converter = None
