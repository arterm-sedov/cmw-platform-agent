"""
Vision-Language Model Support - Core Infrastructure

This module provides the foundation for multimodal (vision, video, audio) support
across different LLM providers (OpenRouter, Gemini Direct API, etc.)
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import base64
import mimetypes


class MediaType(Enum):
    """Supported media types for VL models"""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    PDF = "pdf"
    TEXT = "text"


@dataclass
class VisionInput:
    """
    Unified input format for vision-language models

    Supports multiple input types:
    - Text prompts
    - Images (file path, URL, or base64)
    - Videos (file path, URL, or base64)
    - Audio (file path, URL, or base64)
    - PDFs (file path or URL)
    """

    # Text prompt (required)
    prompt: str

    # Media inputs (optional)
    image_path: Optional[Union[str, Path]] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None

    video_path: Optional[Union[str, Path]] = None
    video_url: Optional[str] = None
    video_base64: Optional[str] = None

    audio_path: Optional[Union[str, Path]] = None
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None

    pdf_path: Optional[Union[str, Path]] = None
    pdf_url: Optional[str] = None

    # Metadata
    media_type: Optional[MediaType] = None
    mime_type: Optional[str] = None

    def __post_init__(self):
        """Auto-detect media type and mime type"""
        if not self.media_type:
            self.media_type = self._detect_media_type()

        if not self.mime_type:
            self.mime_type = self._detect_mime_type()

    def _detect_media_type(self) -> MediaType:
        """Detect media type from provided inputs"""
        if self.image_path or self.image_url or self.image_base64:
            return MediaType.IMAGE
        elif self.video_path or self.video_url or self.video_base64:
            return MediaType.VIDEO
        elif self.audio_path or self.audio_url or self.audio_base64:
            return MediaType.AUDIO
        elif self.pdf_path or self.pdf_url:
            return MediaType.PDF
        else:
            return MediaType.TEXT

    def _detect_mime_type(self) -> Optional[str]:
        """Detect MIME type from file path"""
        path = (
            self.image_path or 
            self.video_path or 
            self.audio_path or 
            self.pdf_path
        )

        if path:
            mime_type, _ = mimetypes.guess_type(str(path))
            return mime_type

        return None

    def get_media_path(self) -> Optional[Union[str, Path]]:
        """Get the media file path (if any)"""
        return (
            self.image_path or 
            self.video_path or 
            self.audio_path or 
            self.pdf_path
        )

    def get_media_url(self) -> Optional[str]:
        """Get the media URL (if any)"""
        return (
            self.image_url or 
            self.video_url or 
            self.pdf_url
        )

    def get_media_base64(self) -> Optional[str]:
        """Get the media base64 data (if any)"""
        return (
            self.image_base64 or 
            self.video_base64 or 
            self.audio_base64
        )

    def has_media(self) -> bool:
        """Check if this input contains any media"""
        return self.media_type != MediaType.TEXT

    def to_base64(self) -> Optional[str]:
        """
        Convert media file to base64 if path is provided
        Returns existing base64 if already available
        """
        # Return existing base64 if available
        if self.get_media_base64():
            return self.get_media_base64()

        # Convert file to base64
        path = self.get_media_path()
        if path:
            with open(path, 'rb') as f:
                data = f.read()
                return base64.b64encode(data).decode('utf-8')

        return None

    def validate(self) -> bool:
        """
        Validate that the input is properly formed
        Returns True if valid, raises ValueError if invalid
        """
        if not self.prompt:
            raise ValueError("Prompt is required")

        # Check that only one media source is provided per type
        image_sources = sum([
            bool(self.image_path),
            bool(self.image_url),
            bool(self.image_base64)
        ])
        if image_sources > 1:
            raise ValueError("Only one image source (path, URL, or base64) should be provided")

        video_sources = sum([
            bool(self.video_path),
            bool(self.video_url),
            bool(self.video_base64)
        ])
        if video_sources > 1:
            raise ValueError("Only one video source (path, URL, or base64) should be provided")

        audio_sources = sum([
            bool(self.audio_path),
            bool(self.audio_url),
            bool(self.audio_base64)
        ])
        if audio_sources > 1:
            raise ValueError("Only one audio source (path, URL, or base64) should be provided")

        pdf_sources = sum([
            bool(self.pdf_path),
            bool(self.pdf_url)
        ])
        if pdf_sources > 1:
            raise ValueError("Only one PDF source (path or URL) should be provided")

        # Validate file paths exist
        path = self.get_media_path()
        if path and not Path(path).exists():
            raise ValueError(f"Media file not found: {path}")

        return True


def create_vision_input(
    prompt: str,
    media_path: Optional[Union[str, Path]] = None,
    media_url: Optional[str] = None,
    media_base64: Optional[str] = None,
    media_type: Optional[MediaType] = None
) -> VisionInput:
    """
    Convenience function to create VisionInput

    Args:
        prompt: Text prompt for the model
        media_path: Path to media file (image, video, audio, PDF)
        media_url: URL to media file
        media_base64: Base64-encoded media data
        media_type: Type of media (auto-detected if not provided)

    Returns:
        VisionInput instance

    Example:
        >>> input = create_vision_input(
        ...     prompt="Describe this image",
        ...     media_path="test.jpg"
        ... )
    """
    kwargs = {"prompt": prompt}

    if media_type == MediaType.IMAGE or (media_path and str(media_path).lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))):
        kwargs["image_path"] = media_path
        kwargs["image_url"] = media_url
        kwargs["image_base64"] = media_base64
    elif media_type == MediaType.VIDEO or (media_path and str(media_path).lower().endswith(('.mp4', '.avi', '.mov', '.webm'))):
        kwargs["video_path"] = media_path
        kwargs["video_url"] = media_url
        kwargs["video_base64"] = media_base64
    elif media_type == MediaType.AUDIO or (media_path and str(media_path).lower().endswith(('.mp3', '.wav', '.ogg', '.m4a'))):
        kwargs["audio_path"] = media_path
        kwargs["audio_url"] = media_url
        kwargs["audio_base64"] = media_base64
    elif media_type == MediaType.PDF or (media_path and str(media_path).lower().endswith('.pdf')):
        kwargs["pdf_path"] = media_path
        kwargs["pdf_url"] = media_url

    return VisionInput(**kwargs)
