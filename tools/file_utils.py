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
