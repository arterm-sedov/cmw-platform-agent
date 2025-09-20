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
                           file_info: FileInfo = None) -> str:
        """Create standardized tool response JSON with Pydantic validation."""
        response = ToolResponse(
            tool_name=tool_name,
            result=result,
            error=error,
            file_info=file_info
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
        from urllib.parse import urlparse
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            if target_path is None:
                # Create temp file
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path) or "downloaded_file"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}")
                target_path = temp_file.name
                temp_file.close()
            
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return target_path
            
        except Exception as e:
            raise IOError(f"Error downloading file from {url}: {str(e)}")
    
    @staticmethod
    def generate_unique_filename(original_filename: str) -> str:
        """
        Generate a unique filename with timestamp and hash for Gradio cache.
        
        Args:
            original_filename (str): Original filename from user upload
            
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
        
        # Generate timestamp and hash
        timestamp = str(int(time.time() * 1000))  # milliseconds
        hash_suffix = hashlib.md5(f"{original_filename}{timestamp}".encode()).hexdigest()[:8]
        
        # Create unique filename
        unique_name = f"{name_without_ext}_{timestamp}_{hash_suffix}{extension}"
        
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
    def find_file_in_gradio_cache(original_filename: str, session_id: str = None) -> str:
        """
        Find a file in Gradio cache by original filename.
        
        Args:
            original_filename (str): Original filename to search for
            session_id (str, optional): Specific session ID to search in
            
        Returns:
            str: Full path to the file in Gradio cache, or None if not found
        """
        import os
        import glob
        
        cache_base = FileUtils.get_gradio_cache_path()
        
        if session_id:
            # Search in specific session directory
            session_path = os.path.join(cache_base, session_id)
            if os.path.exists(session_path):
                # Look for files matching the original name pattern
                pattern = os.path.join(session_path, f"*{original_filename}")
                matches = glob.glob(pattern)
                if matches:
                    return matches[0]
        else:
            # Search in all Gradio cache directories
            gradio_dirs = glob.glob(os.path.join(cache_base, "gradio", "*"))
            for gradio_dir in gradio_dirs:
                if os.path.isdir(gradio_dir):
                    # Look for files matching the original name pattern
                    pattern = os.path.join(gradio_dir, f"*{original_filename}")
                    matches = glob.glob(pattern)
                    if matches:
                        return matches[0]
        
        return None
    
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
                print(f"⚠️ Failed to download URL {file_reference}: {e}")
                return None
        
        # It's a filename - resolve using agent's file registry
        if agent and hasattr(agent, 'get_file_path'):
            return agent.get_file_path(file_reference)
        
        # Fallback: search in Gradio cache
        return FileUtils.find_file_in_gradio_cache(file_reference)
    
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
        
        # Fallback: search in Gradio cache
        return FileUtils.find_file_in_gradio_cache(original_filename)
    
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
