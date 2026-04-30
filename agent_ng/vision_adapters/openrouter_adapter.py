"""
OpenRouter Vision Adapter - Provider-specific implementation for OpenRouter API

Handles multimodal inputs (image, video, audio) for OpenRouter models
"""

from typing import Dict, Any, Optional, Tuple
import base64
import urllib.request

try:
    from ..vision_input import VisionInput, MediaType
    from ..vision_tool_manager import VisionProviderAdapter
    from ..llm_manager import LLMManager, LLMProvider
except ImportError:
    from agent_ng.vision_input import VisionInput, MediaType
    from agent_ng.vision_tool_manager import VisionProviderAdapter
    from agent_ng.llm_manager import LLMManager, LLMProvider


class OpenRouterVisionAdapter(VisionProviderAdapter):
    """
    Adapter for OpenRouter API multimodal support

    Supports:
    - Images (via base64 or URL)
    - Videos (via base64 or URL) 
    - Audio (via base64 or URL)

    OpenRouter uses OpenAI-compatible format for multimodal inputs.
    Audio: ``input_audio`` with raw base64 and ``format`` (e.g. mp3) per
    https://openrouter.ai/docs/guides/overview/multimodal/audio
    (``audio_url`` / data-URL delivery is not reliable for transcription.)
    """

    @staticmethod
    def _openrouter_audio_format(
        mime_type: Optional[str], path_hint: Optional[str] = None
    ) -> str:
        """Map MIME or path to OpenRouter ``input_audio.format`` id."""
        m = (mime_type or "").lower()
        p = (path_hint or "").lower()
        if "mpeg" in m or m == "audio/mp3" or p.endswith(".mp3"):
            return "mp3"
        if "wav" in m or p.endswith(".wav"):
            return "wav"
        if "flac" in m or p.endswith(".flac"):
            return "flac"
        if "m4a" in m or m == "audio/x-m4a" or p.endswith(".m4a"):
            return "m4a"
        if "ogg" in m or p.endswith(".ogg"):
            return "ogg"
        if "aac" in m or p.endswith(".aac"):
            return "aac"
        if "aiff" in m or p.endswith((".aiff", ".aif")):
            return "aiff"
        if m == "audio/pcm" or p.endswith(".pcm"):
            return "pcm16"
        return "mp3"

    @classmethod
    def _openrouter_audio_b64_and_format(
        cls, vision_input: VisionInput
    ) -> Optional[Tuple[str, str]]:
        """
        (raw_base64, format) for OpenRouter ``input_audio``, or None.
        Supports path, base64, data: URLs; fetches http(s) URLs to bytes.
        """
        if vision_input.audio_url and vision_input.audio_url.startswith("data:"):
            meta, _, b64 = vision_input.audio_url.partition(",")
            if not b64:
                return None
            mime = "audio/mpeg"
            if meta.startswith("data:") and ";" in meta[5:]:
                mime = meta[5:].split(";")[0]
            return b64, cls._openrouter_audio_format(mime, None)
        if vision_input.audio_url and (
            vision_input.audio_url.startswith("http://")
            or vision_input.audio_url.startswith("https://")
        ):
            req = urllib.request.Request(
                vision_input.audio_url, headers={"User-Agent": "cmw-platform-agent/1.0"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
                raw = resp.read()
            b64 = base64.b64encode(raw).decode("utf-8")
            return b64, cls._openrouter_audio_format(
                vision_input.mime_type, str(vision_input.audio_url)
            )
        if vision_input.audio_base64:
            b64 = vision_input.audio_base64
            mime = vision_input.mime_type or "audio/mpeg"
            if b64.startswith("data:"):
                meta, _, rest = b64.partition(",")
                b64 = rest
                if meta.startswith("data:") and ";" in meta[5:]:
                    mime = meta[5:].split(";")[0]
            return b64, cls._openrouter_audio_format(
                mime, str(vision_input.audio_path) if vision_input.audio_path else None
            )
        if vision_input.audio_path:
            b64 = vision_input.to_base64()
            if not b64:
                return None
            mime = vision_input.mime_type or "audio/mpeg"
            p = str(vision_input.audio_path)
            return b64, cls._openrouter_audio_format(mime, p)
        return None

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
                # OpenRouter uses "video_url" type (not "image_url")
                # https://openrouter.ai/docs/guides/overview/multimodal/videos
                content.append({
                    "type": "video_url",
                    "video_url": {"url": video_data}
                })

        elif vision_input.media_type == MediaType.AUDIO:
            ap = self._openrouter_audio_b64_and_format(vision_input)
            if ap:
                b64, fmt = ap
                content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64,
                            "format": fmt,
                        },
                    }
                )

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

        # Get LLM - find model index for the specified model
        config = self.llm_manager.LLM_CONFIGS.get(self.provider)
        if not config:
            raise RuntimeError(f"No config found for provider: {self.provider}")

        # Find model index by name
        model_index = 0
        for idx, m in enumerate(config.models):
            if m.get("model") == model:
                model_index = idx
                break

        llm_instance = self.llm_manager.get_llm(self.provider.value, use_tools=False, model_index=model_index)
        if not llm_instance:
            raise RuntimeError(f"Failed to load model: {model}")

        # Extract actual LLM from LLMInstance wrapper
        llm = llm_instance.llm

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
