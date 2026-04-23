"""
Vision Tool Manager - Core infrastructure for multimodal LLM support

Provides unified interface for vision-language models across different providers
with smart routing, fallback handling, and provider-specific adapters.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import os

from .vision_input import VisionInput, MediaType
from .llm_manager import LLMManager, LLMProvider
from .llm_configs import get_default_llm_configs


class VisionProviderAdapter(ABC):
    """
    Abstract base class for provider-specific vision adapters

    Each provider (OpenRouter, Gemini, etc.) implements this interface
    to handle their specific API format for multimodal inputs.
    """

    @abstractmethod
    def supports_media_type(self, media_type: MediaType) -> bool:
        """Check if this provider supports the given media type"""
        pass

    @abstractmethod
    def format_input(self, vision_input: VisionInput) -> Dict[str, Any]:
        """Format VisionInput into provider-specific API format"""
        pass

    @abstractmethod
    def invoke(self, vision_input: VisionInput) -> str:
        """Invoke the model with the given input and return response"""
        pass


class VisionToolManager:
    """
    Central manager for vision-language model operations

    Features:
    - Smart routing based on media type and model capabilities
    - Automatic fallback to alternative models
    - Provider-agnostic interface
    - Cost optimization through model selection

    Usage:
        >>> manager = VisionToolManager()
        >>> input = VisionInput(
        ...     prompt="Describe this image",
        ...     image_path="test.jpg"
        ... )
        >>> result = manager.analyze(input)
    """

    def __init__(self):
        """Initialize VisionToolManager with available providers"""
        self.llm_manager = LLMManager()
        self.configs = get_default_llm_configs()
        self.adapters: Dict[str, VisionProviderAdapter] = {}

        # Load configuration from environment
        self.default_model = os.getenv('VL_DEFAULT_MODEL', 'qwen/qwen3.6-plus')
        self.fast_model = os.getenv('VL_FAST_MODEL', 'google/gemini-2.5-flash')
        self.audio_model = os.getenv('VL_AUDIO_MODEL', 'google/gemini-2.5-flash')
        self.fallback_model = os.getenv('VL_FALLBACK_MODEL', 'google/gemini-2.5-flash')

        # Initialize adapters (will be populated as we implement them)
        self._init_adapters()

    def _init_adapters(self):
        """Initialize provider adapters"""
        # OpenRouter adapter
        try:
            from .vision_adapters import OpenRouterVisionAdapter
            self.adapters['openrouter'] = OpenRouterVisionAdapter(self.llm_manager)
        except ImportError as e:
            print(f"Warning: Could not load OpenRouterVisionAdapter: {e}")

        # Gemini Direct adapter
        try:
            from .vision_adapters import GeminiDirectVisionAdapter
            self.adapters['gemini'] = GeminiDirectVisionAdapter(self.llm_manager)
        except ImportError as e:
            print(f"Warning: Could not load GeminiDirectVisionAdapter: {e}")

    def get_model_for_media_type(self, media_type: MediaType, prefer_fast: bool = False) -> str:
        """
        Select the best model for the given media type

        Args:
            media_type: Type of media to process
            prefer_fast: If True, prefer faster/cheaper models

        Returns:
            Model identifier string

        Routing logic:
        - Audio: Always use audio_model (only Gemini supports audio)
        - Video: Use default_model (Qwen 3.6 Plus for quality)
        - Image (fast): Use fast_model (Gemini 2.5 Flash for speed)
        - Image (quality): Use default_model (Qwen 3.6 Plus)
        - PDF: Use fast_model (Gemini has good PDF support)
        """
        if media_type == MediaType.AUDIO:
            return self.audio_model
        elif media_type == MediaType.VIDEO:
            return self.default_model
        elif media_type == MediaType.IMAGE:
            return self.fast_model if prefer_fast else self.default_model
        elif media_type == MediaType.PDF:
            return self.fast_model
        else:
            return self.default_model

    def get_adapter_for_model(self, model: str) -> Optional[VisionProviderAdapter]:
        """
        Get the appropriate adapter for the given model

        Args:
            model: Model identifier (e.g., "qwen/qwen3.6-plus", "gemini-2.5-flash")

        Returns:
            VisionProviderAdapter instance or None
        """
        # Determine provider from model name
        if model.startswith('google/') or model.startswith('gemini-'):
            # OpenRouter Gemini or Direct Gemini
            if model.startswith('google/'):
                return self.adapters.get('openrouter')
            else:
                return self.adapters.get('gemini')
        elif '/' in model:
            # OpenRouter model (has provider prefix)
            return self.adapters.get('openrouter')
        else:
            # Direct API model
            return self.adapters.get('gemini')

    def analyze(
        self,
        vision_input: VisionInput,
        model: Optional[str] = None,
        prefer_fast: bool = False
    ) -> str:
        """
        Analyze media using vision-language model

        Args:
            vision_input: VisionInput with prompt and media
            model: Specific model to use (optional, auto-selected if not provided)
            prefer_fast: Prefer faster/cheaper models

        Returns:
            Model response as string

        Raises:
            ValueError: If input is invalid
            RuntimeError: If no suitable model/adapter found

        Example:
            >>> manager = VisionToolManager()
            >>> input = VisionInput(
            ...     prompt="What's in this image?",
            ...     image_path="photo.jpg"
            ... )
            >>> result = manager.analyze(input)
        """
        # Validate input
        vision_input.validate()

        # Select model if not provided
        if not model:
            model = self.get_model_for_media_type(
                vision_input.media_type,
                prefer_fast=prefer_fast
            )

        # Get adapter for model
        adapter = self.get_adapter_for_model(model)
        if not adapter:
            raise RuntimeError(f"No adapter found for model: {model}")

        # Check if adapter supports this media type
        if not adapter.supports_media_type(vision_input.media_type):
            # Try fallback model
            fallback = self.fallback_model
            adapter = self.get_adapter_for_model(fallback)
            if not adapter or not adapter.supports_media_type(vision_input.media_type):
                raise RuntimeError(
                    f"Model {model} does not support {vision_input.media_type.value}, "
                    f"and fallback {fallback} also doesn't support it"
                )
            model = fallback

        # Invoke model through adapter
        try:
            return adapter.invoke(vision_input)
        except Exception as e:
            # Try fallback on error
            if model != self.fallback_model:
                fallback_adapter = self.get_adapter_for_model(self.fallback_model)
                if fallback_adapter and fallback_adapter.supports_media_type(vision_input.media_type):
                    return fallback_adapter.invoke(vision_input)
            raise

    def analyze_image(
        self,
        image_path: str,
        prompt: str,
        prefer_fast: bool = True
    ) -> str:
        """
        Convenience method for image analysis

        Args:
            image_path: Path to image file
            prompt: Question or instruction about the image
            prefer_fast: Use faster model (default: True)

        Returns:
            Model response
        """
        vision_input = VisionInput(
            prompt=prompt,
            image_path=image_path
        )
        return self.analyze(vision_input, prefer_fast=prefer_fast)

    def analyze_video(
        self,
        video_path: str,
        prompt: str
    ) -> str:
        """
        Convenience method for video analysis

        Args:
            video_path: Path to video file
            prompt: Question or instruction about the video

        Returns:
            Model response
        """
        vision_input = VisionInput(
            prompt=prompt,
            video_path=video_path
        )
        return self.analyze(vision_input, prefer_fast=False)

    def analyze_audio(
        self,
        audio_path: str,
        prompt: str
    ) -> str:
        """
        Convenience method for audio analysis

        Args:
            audio_path: Path to audio file
            prompt: Question or instruction about the audio

        Returns:
            Model response
        """
        vision_input = VisionInput(
            prompt=prompt,
            audio_path=audio_path
        )
        return self.analyze(vision_input)

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about available models and their capabilities

        Returns:
            Dictionary with model capabilities
        """
        capabilities = {
            "models": {
                "default": self.default_model,
                "fast": self.fast_model,
                "audio": self.audio_model,
                "fallback": self.fallback_model
            },
            "media_types": {
                "image": True,
                "video": True,
                "audio": True,
                "pdf": True
            },
            "providers": list(self.adapters.keys())
        }
        return capabilities
