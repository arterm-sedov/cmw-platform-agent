"""
Image model registry for AI-driven image generation.

Source of truth for the small, curated list of image-generation models the
agent knows how to invoke. Kept intentionally separate from
:mod:`agent_ng.llm_configs` because image-generation models are not
interchangeable with chat/vision LLMs:

* Their responses carry binary payloads (``message.images[...]``) that
  LangChain's ``ChatOpenAI`` client discards, so they cannot flow through
  :class:`agent_ng.llm_manager.LLMManager`.
* They should not appear in chat/model selector UIs.
* Per-call pricing is returned by OpenRouter in ``response.usage.cost`` or
  by Polza.ai in ``response.usage.cost_rub``.

Each model lists its supported backends in ``providers`` (ordered by
preference). The engine tries them in order and returns the first success —
so regional or key-availability differences are handled transparently.

The registry is a single Python dict — promote to YAML if the list grows
beyond what is comfortably reviewed in code.

Example:
    >>> cfg = get_model_config("google/gemini-2.5-flash-image")
    >>> cfg.modalities
    ['image', 'text']
    >>> cfg.providers
    ['polza', 'openrouter']
    >>> get_default_model()
    'google/gemini-3.1-flash-image-preview'
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageModelConfig:
    """Static metadata describing one image-generation model.

    Attributes:
        name: Model slug sent to the API (also the registry key).
        providers: Ordered list of provider keys to try. The engine
            iterates through them and returns the first successful
            response, so regional or credential availability differences
            are handled transparently. E.g. ``["polza", "openrouter"]``
            tries Polza.ai first (works in Russia), OpenRouter as
            fallback.
        modalities: Value to send as the ``modalities`` request field.
            Multimodal models (Gemini) use ``["image", "text"]``; image-
            only models (Flux, Seedream) use ``["image"]``.
        supports_image_config: Whether the model documents support for
            ``image_config.aspect_ratio`` / ``image_config.image_size``
            (OpenRouter) or ``aspect_ratio`` / ``quality`` (Polza).
            Only Google Gemini and Seedream 4.5 image models document this.
        max_reference_images: Maximum number of reference images the model
            accepts as input (image-to-image / editing). 0 means the model
            is text-to-image only. Callers should clip the list to this count
            before passing it to the engine.
        description: Operations-facing summary shown in dev docs / logs.
        prompt_style_hint: LLM-facing prompting guidance. Must not leak
            the slug, provider or vendor name.
    """

    name: str
    providers: list[str]
    modalities: list[str]
    supports_image_config: bool
    max_reference_images: int = 0
    """Maximum number of reference images the model accepts (0 = text-to-image only)."""
    description: str = ""
    prompt_style_hint: str = ""
    provider_model_ids: dict[str, str] = field(default_factory=dict)
    """Per-provider model ID overrides.

    When a provider uses a different slug for the same model (e.g. OpenRouter
    uses ``bytedance-seed/seedream-4.5`` but Polza uses
    ``bytedance/seedream-4.5``), add an entry here so the provider adapter
    sends the correct ID.  Falls back to ``name`` when not set.
    """

    @property
    def provider(self) -> str:
        """Primary (first) provider — backward-compat shorthand."""
        return self.providers[0]


# ---------------------------------------------------------------------------
# Registry — edit here to add or remove models.
# ---------------------------------------------------------------------------

# Reusable prompt-style hints, deduplicated. These are the strings the
# LLM actually sees — they must not leak vendor or model names.
_HINT_GEMINI_STYLE = (
    "Accepts natural, conversational prompts. Renders multilingual text "
    "(including Cyrillic) reliably when the words are written out in "
    "the prompt. Good at clean business graphics, icons, diagrams and "
    "banners. You can request a specific shape with `aspect_ratio` and a "
    "resolution tier with `image_size`."
)
_HINT_DIFFUSION_ARTISTIC = (
    "Prefers concise, descriptive prompts listing subject, style, "
    "colors and mood. Strong on photorealistic and richly stylized "
    "artwork; weaker at embedded text — describe text visually "
    "(e.g. 'bold sans-serif headline') rather than relying on the "
    "model to spell it correctly. Ignores `aspect_ratio` / "
    "`image_size` hints and uses its own default dimensions."
)
_HINT_DIFFUSION_MULTILINGUAL = (
    "Accepts either natural or keyword-style prompts. Renders "
    "multilingual text (including Cyrillic) better than most artistic "
    "models — put the exact text in quotes. Produces large canvases by "
    "default. Ignores `aspect_ratio` / `image_size` hints."
)
_HINT_TEXT_FOCUSED = (
    "Specialized for images with readable embedded text (posters, "
    "social cards, banners). Put the exact text in quotes inside the "
    "prompt and describe its role (e.g. 'headline', 'subheadline'). "
    "Ignores `aspect_ratio` / `image_size` hints."
)

# Provider order: polza first (Russian CDN, no regional restrictions),
# openrouter as fallback. Models only available on one provider list it alone.


IMAGE_MODELS: dict[str, ImageModelConfig] = {
    # ------ Google Gemini family (multimodal: image + text output) -----
    "google/gemini-2.5-flash-image": ImageModelConfig(
        name="google/gemini-2.5-flash-image",
        providers=["polza", "openrouter"],
        modalities=["image", "text"],
        supports_image_config=True,
        max_reference_images=8,
        description=(
            "Fast everyday workhorse for icons, diagrams and simple "
            "business illustrations. Balanced quality and speed."
        ),
        prompt_style_hint=_HINT_GEMINI_STYLE,
    ),
    "google/gemini-3.1-flash-image-preview": ImageModelConfig(
        name="google/gemini-3.1-flash-image-preview",
        providers=["polza", "openrouter"],
        modalities=["image", "text"],
        supports_image_config=True,
        max_reference_images=8,
        description=(
            "Newer fast tier (Nano Banana 2) with a larger native canvas "
            "and extra aspect ratios (including 21:9) for banners and "
            "vertical layouts. Supports up to 4K resolution."
        ),
        prompt_style_hint=_HINT_GEMINI_STYLE,
    ),
    "google/gemini-3-pro-image-preview": ImageModelConfig(
        name="google/gemini-3-pro-image-preview",
        providers=["polza", "openrouter"],
        modalities=["image", "text"],
        supports_image_config=True,
        max_reference_images=8,
        description=(
            "Premium quality (Nano Banana Pro) for polished hero imagery, "
            "complex scenes and multilingual text. Slower and pricier."
        ),
        prompt_style_hint=_HINT_GEMINI_STYLE,
    ),
    # ------ OpenAI GPT image family (multimodal: image + text output) --
    "openai/gpt-5-image-mini": ImageModelConfig(
        name="openai/gpt-5-image-mini",
        providers=["polza", "openrouter"],
        modalities=["image", "text"],
        supports_image_config=False,
        max_reference_images=10,
        description=(
            "Budget OpenAI image model. Sometimes unavailable depending "
            "on geography — use an alternative when calls fail."
        ),
        prompt_style_hint=_HINT_GEMINI_STYLE,
    ),
    "openai/gpt-5-image": ImageModelConfig(
        name="openai/gpt-5-image",
        providers=["polza", "openrouter"],
        modalities=["image", "text"],
        supports_image_config=False,
        max_reference_images=10,
        description=(
            "Standard OpenAI image model. Polza used first to avoid "
            "regional restrictions."
        ),
        prompt_style_hint=_HINT_GEMINI_STYLE,
    ),
    "openai/gpt-5.4-image-2": ImageModelConfig(
        name="openai/gpt-5.4-image-2",
        providers=["polza", "openrouter"],
        modalities=["image", "text"],
        supports_image_config=True,
        max_reference_images=1,
        description=(
            "Latest OpenAI image model. Supports 11 aspect ratios, "
            "up to 4 images per request, and reference-image editing. "
            "Polza used first to avoid regional restrictions."
        ),
        prompt_style_hint=_HINT_GEMINI_STYLE,
    ),
    # ------ Black Forest Labs FLUX (image-only output) ------------------
    "black-forest-labs/flux.2-flex": ImageModelConfig(
        name="black-forest-labs/flux.2-flex",
        providers=["polza", "openrouter"],
        modalities=["image"],
        supports_image_config=True,
        max_reference_images=8,
        description=(
            "Fast artistic text-to-image at moderate cost. Good prompt "
            "adherence on stylized or photographic imagery."
        ),
        prompt_style_hint=_HINT_DIFFUSION_ARTISTIC,
    ),
    "black-forest-labs/flux.2-pro": ImageModelConfig(
        name="black-forest-labs/flux.2-pro",
        providers=["polza", "openrouter"],
        modalities=["image"],
        supports_image_config=True,
        max_reference_images=8,
        description=(
            "Production-grade artistic text-to-image up to 4 MP. Best "
            "for photorealistic or richly stylized scenes; weaker at "
            "embedded text."
        ),
        prompt_style_hint=_HINT_DIFFUSION_ARTISTIC,
    ),
    "black-forest-labs/flux.2-max": ImageModelConfig(
        name="black-forest-labs/flux.2-max",
        providers=["openrouter"],
        modalities=["image"],
        supports_image_config=False,
        max_reference_images=0,
        description=(
            "Highest artistic fidelity tier. Use when visual richness "
            "matters more than speed or cost."
        ),
        prompt_style_hint=_HINT_DIFFUSION_ARTISTIC,
    ),
    # ------ ByteDance Seedream (image-only output) ----------------------
    "bytedance-seed/seedream-5-lite": ImageModelConfig(
        name="bytedance-seed/seedream-5-lite",
        providers=["polza"],
        modalities=["image"],
        supports_image_config=True,
        max_reference_images=10,
        provider_model_ids={"polza": "bytedance/seedream-5-lite"},
        description=(
            "Seedream 5.0 Lite via Polza.ai. Generation and image-to-image "
            "editing, 2K/3K output, up to 10 reference images."
        ),
        prompt_style_hint=_HINT_DIFFUSION_MULTILINGUAL,
    ),
    "bytedance-seed/seedream-4.5": ImageModelConfig(
        name="bytedance-seed/seedream-4.5",
        providers=["polza", "openrouter"],
        modalities=["image"],
        supports_image_config=True,
        max_reference_images=14,
        provider_model_ids={"polza": "bytedance/seedream-4.5"},
        description=(
            "Large-canvas (2K/4K) output. Handles multilingual text "
            "(including Cyrillic) and small-text rendering better than "
            "most text-to-image models. Polza: quality=basic(2K)/high(4K)."
        ),
        prompt_style_hint=_HINT_DIFFUSION_MULTILINGUAL,
    ),
    "bytedance-seed/seedream-4": ImageModelConfig(
        name="bytedance-seed/seedream-4",
        providers=["polza"],
        modalities=["image"],
        supports_image_config=True,
        max_reference_images=10,
        description=(
            "Previous-generation Seedream with 4K output and unique "
            "artistic style. Good multilingual text rendering."
        ),
        prompt_style_hint=_HINT_DIFFUSION_MULTILINGUAL,
    ),
    "bytedance-seed/seedream-3": ImageModelConfig(
        name="bytedance-seed/seedream-3",
        providers=["polza"],
        modalities=["image"],
        supports_image_config=True,
        max_reference_images=1,
        provider_model_ids={"polza": "bytedance/seedream"},
        description="Proven Seedream generation with solid multilingual support.",
        prompt_style_hint=_HINT_DIFFUSION_MULTILINGUAL,
    ),
    # ------ Sourceful Riverflow (image-only, text-focused, OR-only) ----
    "sourceful/riverflow-v2-fast": ImageModelConfig(
        name="sourceful/riverflow-v2-fast",
        providers=["openrouter"],
        modalities=["image"],
        supports_image_config=False,
        max_reference_images=0,
        description=(
            "Specialized for readable text inside images (posters, "
            "social cards). Accepts custom font URLs."
        ),
        prompt_style_hint=_HINT_TEXT_FOCUSED,
    ),
    "sourceful/riverflow-v2-pro": ImageModelConfig(
        name="sourceful/riverflow-v2-pro",
        providers=["openrouter"],
        modalities=["image"],
        supports_image_config=False,
        max_reference_images=0,
        description=(
            "Higher-quality text-focused tier. Best when embedded text "
            "must be sharp and perfectly legible."
        ),
        prompt_style_hint=_HINT_TEXT_FOCUSED,
    ),
    # ------ Polza-exclusive models --------------------------------------
    "x-ai/grok-imagine": ImageModelConfig(
        name="x-ai/grok-imagine",
        providers=["polza"],
        modalities=["image"],
        supports_image_config=False,
        max_reference_images=1,
        description=(
            "Grok Imagine via Polza.ai. Photorealistic and artistic "
            "generation from xAI, billed in RUB."
        ),
        prompt_style_hint=_HINT_DIFFUSION_ARTISTIC,
    ),
    "qwen/qwen-vl-max-image": ImageModelConfig(
        name="qwen/qwen-vl-max-image",
        providers=["polza"],
        modalities=["image"],
        supports_image_config=False,
        max_reference_images=1,
        description=(
            "Qwen VL Image via Polza.ai. Alibaba multimodal model, "
            "billed in RUB."
        ),
        prompt_style_hint=_HINT_DIFFUSION_MULTILINGUAL,
    ),
}


DEFAULT_IMAGE_MODEL: str = "google/gemini-3.1-flash-image-preview"
"""Compile-time default; overridable via ``IMAGE_GEN_DEFAULT_MODEL`` env.

Routes through Polza.ai first (Russian CDN, no regional restrictions) with
OpenRouter as fallback — see the ``providers`` list on this entry.

Chosen from live side-by-side comparison of Russian + business prompts —
see ``docs/image_generation/progress_reports/20260425_model_comparison.md``
and the accompanying PNG samples. Balanced quality, multilingual text
rendering and cost.
"""


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


def get_image_models() -> dict[str, ImageModelConfig]:
    """Return the image-model registry.

    Returned dict is the live registry reference — callers must not mutate.
    """
    return IMAGE_MODELS


def get_model_config(slug: object) -> ImageModelConfig | None:
    """Look up a model configuration by slug.

    Returns ``None`` for unknown, empty, or non-string inputs (so callers
    can validate user/LLM-supplied model names defensively).
    """
    if not isinstance(slug, str) or not slug:
        return None
    return IMAGE_MODELS.get(slug)


def get_default_model() -> str:
    """Resolve the default image model.

    Honors ``IMAGE_GEN_DEFAULT_MODEL`` when it names a registered model.
    Unknown overrides log a warning and fall back to
    :data:`DEFAULT_IMAGE_MODEL`.
    """
    override = os.getenv("IMAGE_GEN_DEFAULT_MODEL", "").strip()
    if override and override in IMAGE_MODELS:
        return override
    if override:
        logger.warning(
            "IMAGE_GEN_DEFAULT_MODEL=%r is not in the image model registry; "
            "falling back to %r",
            override,
            DEFAULT_IMAGE_MODEL,
        )
    return DEFAULT_IMAGE_MODEL


def get_default_prompt_style_hint() -> str:
    """Return the LLM-facing prompt-style hint for the active default model.

    Never reveals the model slug, provider or vendor name — only describes
    prompting behavior. Callers (typically the tool description builder)
    use this to help the calling LLM adapt prompt style to the active
    backend without learning which backend is active.

    Returns an empty string if the active model has no hint configured.
    """
    cfg = get_model_config(get_default_model())
    return cfg.prompt_style_hint if cfg else ""


__all__ = [
    "DEFAULT_IMAGE_MODEL",
    "IMAGE_MODELS",
    "ImageModelConfig",
    "get_default_model",
    "get_default_prompt_style_hint",
    "get_image_models",
    "get_model_config",
]
