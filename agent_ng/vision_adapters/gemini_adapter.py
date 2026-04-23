"""
Gemini Direct Vision Adapter - Provider-specific implementation for Gemini Direct API

Handles multimodal inputs (image, video, audio) for Gemini models via Direct API
"""

from typing import Dict, Any, Optional
import base64
from pathlib import Path

try:
    from ..vision_input import VisionInput, MediaType
    from ..vision_tool_manager import VisionProviderAdapter
    from ..llm_manager import LLMManager, LLMProvider
except ImportError:
    from agent_ng.vision_input import VisionInput, MediaType
    from agent_ng.vision_tool_manager import VisionProviderAdapter
    from agent_ng.llm_manager import LLMManager, LLMProvider


class GeminiDirectVisionAdapter(VisionProviderAdapter):
    """
    Adapter for Gemini Direct API multimodal support

    Supports:
    - Images (via base64 or file upload)
    - Videos (via file upload, YouTube URLs)
    - Audio (via file upload)

    Uses google.genai SDK for direct Gemini API access
    """

    def __init__(self, llm_manager: LLMManager):
        """
        Initialize Gemini Direct adapter

        Args:
            llm_manager: LLMManager instance for model access
        """
        self.llm_manager = llm_manager
        self.provider = LLMProvider.GEMINI

        # Import Gemini SDK
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
            self.available = True
        except ImportError:
            self.available = False
            print("Warning: google-genai not available. Install with: pip install google-genai")

    def supports_media_type(self, media_type: MediaType) -> bool:
        """
        Check if Gemini Direct supports the given media type

        Args:
            media_type: Type of media to check

        Returns:
            True if supported, False otherwise
        """
        if not self.available:
            return False

        # Gemini supports all media types
        return media_type in [MediaType.IMAGE, MediaType.VIDEO, MediaType.AUDIO]

    def format_input(self, vision_input: VisionInput) -> Dict[str, Any]:
        """
        Format VisionInput into Gemini Direct API format

        Args:
            vision_input: VisionInput to format

        Returns:
            Dict with formatted input for Gemini API
        """
        if not self.available:
            raise RuntimeError("Gemini SDK not available")

        # Get media path
        media_path = None
        if vision_input.image_path:
            media_path = vision_input.image_path
        elif vision_input.video_path:
            media_path = vision_input.video_path
        elif vision_input.audio_path:
            media_path = vision_input.audio_path

        if not media_path:
            raise ValueError("No media path provided")

        return {
            "media_path": media_path,
            "prompt": vision_input.prompt,
            "media_type": vision_input.media_type
        }

    def invoke(self, vision_input: VisionInput, model: Optional[str] = None) -> str:
        """
        Invoke Gemini Direct model with vision input

        Args:
            vision_input: VisionInput to process
            model: Model to use (default: gemini-2.5-flash)

        Returns:
            Model response as string

        Raises:
            RuntimeError: If model invocation fails
        """
        if not self.available:
            raise RuntimeError("Gemini SDK not available")

        # Get Gemini client
        import os
        gemini_key = os.getenv('GEMINI_KEY')
        if not gemini_key:
            raise RuntimeError("GEMINI_KEY not found in environment")

        client = self.genai.Client(api_key=gemini_key)

        # Default model
        if not model:
            model = "gemini-2.5-flash"

        # Format input
        formatted = self.format_input(vision_input)
        media_path = formatted["media_path"]
        prompt = formatted["prompt"]

        try:
            # Upload file to Gemini
            uploaded_file = client.files.upload(file=media_path)

            # Create content with file and prompt
            contents = self.types.Content(
                parts=[
                    self.types.Part(file_data=self.types.FileData(file_uri=uploaded_file.uri)),
                    self.types.Part(text=prompt)
                ]
            )

            # Generate response
            response = client.models.generate_content(
                model=model,
                contents=contents
            )

            return response.text

        except Exception as e:
            raise RuntimeError(f"Gemini Direct API call failed: {e}")
