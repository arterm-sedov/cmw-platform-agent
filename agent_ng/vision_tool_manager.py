"""
Vision Tool Manager - Core infrastructure for multimodal LLM support

Provides unified interface for vision-language models across different providers
with routing and provider-specific adapters (single attempt; errors propagate).
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
    - Routing based on media type and configured models
    - Provider-agnostic interface
    - One invoke per analyze() call (no automatic retries)

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

        # VL model configuration
        self.vl_model = os.getenv('VL_DEFAULT_MODEL', 'qwen/qwen3.6-plus')
        self.vl_audio_model = os.getenv('VL_AUDIO_MODEL', 'gemini-2.5-flash')
        self.vl_youtube_model = os.getenv("VL_YOUTUBE_MODEL", "gemini-2.5-flash")
        self.vl_gemini_provider = os.getenv('VL_GEMINI_PROVIDER', 'auto').strip().lower()
        # If set, overrides VL_GEMINI_PROVIDER only for YouTube videoUrl routing
        ygp = os.getenv("VL_YOUTUBE_GEMINI_PROVIDER", "").strip().lower()
        self.vl_youtube_gemini_provider: Optional[str] = ygp or None

        # Initialize adapters
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

    @staticmethod
    def _is_youtube_url(url: Optional[str]) -> bool:
        """Return True when URL points to YouTube."""
        if not url:
            return False
        lowered = url.lower()
        return "youtube.com" in lowered or "youtu.be" in lowered

    @staticmethod
    def _to_google_direct_model(model: str) -> str:
        """Normalize Gemini model name to Google Direct format."""
        if model.startswith('google/'):
            return model.split('/', 1)[1]
        return model

    @staticmethod
    def _to_openrouter_gemini_model(model: str) -> str:
        """Normalize Gemini model name to OpenRouter format."""
        if model.startswith('google/'):
            return model
        if model.startswith('gemini-'):
            return f"google/{model}"
        return model

    def _resolve_gemini_model(
        self,
        base_model: str,
        *,
        prefer_openrouter_on_auto: bool,
        provider: Optional[str] = None,
    ) -> str:
        """
        Resolve Gemini model using explicit provider preference.

        ``provider`` defaults to :attr:`vl_gemini_provider` (``VL_GEMINI_PROVIDER``).

        VL_GEMINI_PROVIDER (and YouTube when ``VL_YOUTUBE_GEMINI_PROVIDER`` is set):
        - openrouter: always google/<model>
        - google: always gemini-<...>
        - auto: use ``prefer_openrouter_on_auto`` to pick OpenRouter vs Google Direct
        """
        p = (provider or self.vl_gemini_provider).strip().lower()
        if p == "openrouter":
            return self._to_openrouter_gemini_model(base_model)
        if p == "google":
            return self._to_google_direct_model(base_model)
        # auto
        if prefer_openrouter_on_auto:
            return self._to_openrouter_gemini_model(base_model)
        return self._to_google_direct_model(base_model)

    def get_model_for_input(self, vision_input: VisionInput) -> str:
        """
        Select model based on media type and source.

        Routing rules:
        - Audio → VL_AUDIO_MODEL; Gemini ids go through VL_GEMINI_PROVIDER (default auto
          prefers OpenRouter for ``google/gemini-*`` now that OpenRouter uses ``input_audio``)
        - YouTube URLs → ``VL_YOUTUBE_MODEL``; provider: ``VL_YOUTUBE_GEMINI_PROVIDER`` if
          set, else ``VL_GEMINI_PROVIDER``
        - Video files → Qwen via OpenRouter (default)
        - Images → Qwen via OpenRouter (default)
        """
        if vision_input.media_type == MediaType.AUDIO:
            audio_model = self.vl_audio_model
            if audio_model.startswith('gemini-') or audio_model.startswith('google/gemini-'):
                return self._resolve_gemini_model(
                    audio_model, prefer_openrouter_on_auto=True
                )
            return audio_model

        # YouTube: use VL_YOUTUBE_MODEL; optional VL_YOUTUBE_GEMINI_PROVIDER override.
        if vision_input.media_type == MediaType.VIDEO:
            video_url = vision_input.video_url or vision_input.get_media_url()
            if self._is_youtube_url(video_url):
                gm = self.vl_youtube_model
                if gm.startswith('gemini-') or gm.startswith('google/gemini-'):
                    return self._resolve_gemini_model(
                        gm,
                        prefer_openrouter_on_auto=True,
                        provider=self.vl_youtube_gemini_provider,
                    )
                return gm

        # Everything else uses default (Qwen via OpenRouter)
        return self.vl_model

    def get_adapter_for_model(self, model: str) -> Optional[VisionProviderAdapter]:
        """Get the appropriate adapter for the given model."""
        if model.startswith('google/'):
            return self.adapters.get('openrouter')
        elif model.startswith('gemini-'):
            return self.adapters.get('gemini')
        elif '/' in model:
            return self.adapters.get('openrouter')
        else:
            return self.adapters.get('gemini')

    def analyze(
        self,
        vision_input: VisionInput,
        model: Optional[str] = None
    ) -> str:
        """
        Analyze media using vision-language model.

        Args:
            vision_input: VisionInput with prompt and media
            model: Specific model to use (optional, auto-selected if not provided)
        """
        vision_input.validate()

        if not model:
            model = self.get_model_for_input(vision_input)

        adapter = self.get_adapter_for_model(model)
        if not adapter:
            raise RuntimeError(f"No adapter found for model: {model}")

        return adapter.invoke(vision_input, model=model)

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """Analyze image with default VL model"""
        vision_input = VisionInput(prompt=prompt, image_path=image_path)
        return self.analyze(vision_input)

    def analyze_video(self, video_path: str, prompt: str) -> str:
        """Analyze video with default VL model"""
        vision_input = VisionInput(prompt=prompt, video_path=video_path)
        return self.analyze(vision_input)

    def analyze_audio(self, audio_path: str, prompt: str) -> str:
        """Analyze audio using VL_AUDIO_MODEL and VL_GEMINI_PROVIDER resolution."""
        vision_input = VisionInput(prompt=prompt, audio_path=audio_path)
        return self.analyze(vision_input, model=self.vl_audio_model)

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about available models and their capabilities

        Returns:
            Dictionary with model capabilities
        """
        capabilities = {
            "models": {
                "default": self.vl_model,
                "audio": self.vl_audio_model,
                "youtube": self.vl_youtube_model,
            },
            "routing": {
                "gemini_provider": self.vl_gemini_provider,
                "youtube_provider": self.vl_youtube_gemini_provider
                or self.vl_gemini_provider,
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
