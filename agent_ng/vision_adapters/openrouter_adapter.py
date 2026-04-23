"""
OpenRouter Vision Adapter - Provider-specific implementation for OpenRouter API

Handles multimodal inputs (image, video, audio) for OpenRouter models
"""

from typing import Dict, Any, Optional
import base64
from pathlib import Path

from .vision_input import VisionInput, MediaType
from .vision_tool_manager import VisionProviderAdapter
from .llm_manager import LLMManager, LLMProvider


class OpenRouterVisionAdapter(VisionProviderAdapter):
    """
    Adapter for OpenRouter API multimodal support

    Supports:
    - Images (via base64 or URL)
    - Videos (via base64 or URL) 
    - Audio (via base64 or URL)

    OpenRouter uses OpenAI-compatible format for multimodal inputs
    """

    def __init__(self, llm_manager: LLMManager):
        """
        Initialize OpenRouter adapter

        Args:
            llm_manager: LLMManager instance for model access
        """
        self.llm_manager = llm_manager
        self.provider = LLMProvider.OPENROUTER

    def supports_media_type(self, media_type: MediaType) -> bool:
        """
        Check if OpenRouter supports the given media type

        OpenRouter supports: image, video, audio (model-dependent)
        """
        return media_type in [MediaType.IMAGE, MediaType.VIDEO, MediaType.AUDIO, MediaType.TEXT]

    def format_input(self, vision_input: VisionInput) -> Dict[str, Any]:
        """
        Format VisionInput into OpenRouter API format

        OpenRouter uses OpenAI-compatible message format:
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "prompt"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        }

        Args:
            vision_input: VisionInput to format

        Returns:
            Formatted message dict
        """
        content = []

        # Add text prompt
        content.append({
            "type": "text",
            "text": vision_input.prompt
        })

        # Add media based on type
        if vision_input.media_type == MediaType.IMAGE:
            image_data = self._get_image_data(vision_input)
            if image_data:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data}
                })

        elif vision_input.media_type == MediaType.VIDEO:
            video_data = self._get_video_data(vision_input)
            if video_data:
                # Some models use "video_url" type
                content.append({
                    "type": "image_url",  # OpenRouter may use image_url for video
                    "image_url": {"url": video_data}
                })

        elif vision_input.media_type == MediaType.AUDIO:
            audio_data = self._get_audio_data(vision_input)
            if audio_data:
                # Audio format varies by model
                content.append({
                    "type": "image_url",  # OpenRouter may use image_url for audio
                    "image_url": {"url": audio_data}
                })

        return {
            "role": "user",
            "content": content
        }

    def _get_image_data(self, vision_input: VisionInput) -> Optional[str]:
        """Get image data as data URL"""
        # Prefer URL if available
        if vision_input.image_url:
            return vision_input.image_url

        # Use base64 if available
        if vision_input.image_base64:
            mime_type = vision_input.mime_type or "image/jpeg"
            return f"data:{mime_type};base64,{vision_input.image_base64}"

        # Convert file to base64
        if vision_input.image_path:
            base64_data = vision_input.to_base64()
            mime_type = vision_input.mime_type or "image/jpeg"
            return f"data:{mime_type};base64,{base64_data}"

        return None

    def _get_video_data(self, vision_input: VisionInput) -> Optional[str]:
        """Get video data as data URL"""
        # Prefer URL if available
        if vision_input.video_url:
            return vision_input.video_url

        # Use base64 if available
        if vision_input.video_base64:
            mime_type = vision_input.mime_type or "video/mp4"
            return f"data:{mime_type};base64,{vision_input.video_base64}"

        # Convert file to base64
        if vision_input.video_path:
            base64_data = vision_input.to_base64()
            mime_type = vision_input.mime_type or "video/mp4"
            return f"data:{mime_type};base64,{base64_data}"

        return None

    def _get_audio_data(self, vision_input: VisionInput) -> Optional[str]:
        """Get audio data as data URL"""
        # Prefer URL if available
        if vision_input.audio_url:
            return vision_input.audio_url

        # Use base64 if available
        if vision_input.audio_base64:
            mime_type = vision_input.mime_type or "audio/mpeg"
            return f"data:{mime_type};base64,{vision_input.audio_base64}"

        # Convert file to base64
        if vision_input.audio_path:
            base64_data = vision_input.to_base64()
            mime_type = vision_input.mime_type or "audio/mpeg"
            return f"data:{mime_type};base64,{base64_data}"

        return None

    def invoke(self, vision_input: VisionInput, model: Optional[str] = None) -> str:
        """
        Invoke OpenRouter model with vision input

        Args:
            vision_input: VisionInput to process
            model: Model to use (e.g., "qwen/qwen3.6-plus", "google/gemini-2.5-flash")

        Returns:
            Model response as string

        Raises:
            RuntimeError: If model invocation fails
        """
        # Get LLM instance
        if not model:
            # Use default from environment or Qwen 3.6 Plus
            import os
            model = os.getenv('VL_DEFAULT_MODEL', 'qwen/qwen3.6-plus')

        llm = self.llm_manager.get_llm(self.provider, model_name=model)
        if not llm:
            raise RuntimeError(f"Failed to load model: {model}")

        # Format input
        message = self.format_input(vision_input)

        # Invoke model
        try:
            # LangChain expects messages list
            from langchain_core.messages import HumanMessage

            # Create HumanMessage with multimodal content
            human_message = HumanMessage(content=message["content"])

            # Invoke
            response = llm.invoke([human_message])

            return response.content

        except Exception as e:
            raise RuntimeError(f"OpenRouter model invocation failed: {e}")
