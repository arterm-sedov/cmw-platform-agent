"""
LLM Manager Module
==================

This module provides a persistent LLM manager that initializes and manages
multiple LLM providers. The manager is designed to be stateless and serve
multiple users without reinitializing LLM instances.

Key Features:
- Persistent LLM instances across requests
- Support for multiple providers (Gemini, Groq, HuggingFace, etc.)
- Automatic fallback and retry logic
- Thread-safe operations
- Configuration-driven initialization

Usage:
    llm_manager = LLMManager()
    llm = llm_manager.get_llm("gemini", use_tools=True)
"""

from dataclasses import dataclass
from enum import Enum
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain imports
# Local imports with robust fallback handling
import sys

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mistralai.chat_models import ChatMistralAI
# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .langsmith_config import get_langsmith_config, get_openai_wrapper
    from .logging_config import _parse_bool
    from .provider_adapters import MistralWrapper, is_mistral_model
    from .utils import ensure_valid_answer
except ImportError:
    try:
        from agent_ng.langsmith_config import get_langsmith_config, get_openai_wrapper
        from agent_ng.logging_config import _parse_bool
        from agent_ng.provider_adapters import MistralWrapper, is_mistral_model
        from agent_ng.utils import ensure_valid_answer
    except ImportError as e:
        print("💥 CRITICAL ERROR: Cannot import required modules in llm_manager!")
        print(f"   Import failed: {e}")
        print("🔧 Please check that all dependencies are installed and modules exist")
        msg = f"Failed to import required modules in llm_manager: {e}"
        raise ImportError(msg)


class LLMProvider(Enum):
    """Enumeration of supported LLM providers"""

    GEMINI = "gemini"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    POLZA = "polza"
    MISTRAL = "mistral"
    GIGACHAT = "gigachat"


@dataclass
class LLMConfig:
    """Configuration for a specific LLM provider"""

    name: str
    type_str: str
    api_key_env: str
    api_base_env: str | None = None
    scope_env: str | None = None
    verify_ssl_env: str | None = None
    max_history: int = 20
    tool_support: bool = False
    force_tools: bool = False
    models: list[dict[str, Any]] = None
    token_per_minute_limit: int | None = None
    enable_chunking: bool = True
    # Vision-Language capabilities
    vision_support: bool = False
    video_support: bool = False
    audio_support: bool = False

    def __post_init__(self):
        if self.models is None:
            self.models = []


@dataclass
class LLMInstance:
    """Wrapper for an initialized LLM instance with metadata"""

    llm: Any
    provider: LLMProvider
    model_name: str
    config: dict[str, Any]
    initialized_at: float
    last_used: float
    is_healthy: bool = True
    error_count: int = 0
    last_error: str | None = None
    bound_tools: bool = False


class LLMManager:
    """
    Persistent LLM Manager that initializes and manages multiple LLM providers.

    This class provides a centralized way to manage LLM instances across the
    application, ensuring they are initialized once and reused efficiently.
    """

    # Single source of truth for LLM configuration
    # Loaded from separate config module for better maintainability
    # Can be enriched at runtime with pricing data from OpenRouter
    LLM_CONFIGS = None  # Will be initialized in __init__

    # Single provider from environment variable
    # No more sequence - use AGENT_PROVIDER from dotenv

    def __init__(self):
        """Initialize the LLM Manager"""
        self._instances: dict[str, LLMInstance] = {}
        self._lock = threading.Lock()
        self._initialization_logs = []
        self._health_check_interval = 300  # 5 minutes
        self._last_health_check = 0
        # Allowed providers allowlist loaded from environment; None means no restriction
        self._allowed_providers = self._load_allowed_providers()
        # Initialize LLM configurations (can be enriched with pricing data)
        # Import here to avoid circular import (llm_configs imports from llm_manager)
        try:
            from .llm_configs import get_default_llm_configs
        except ImportError:
            from agent_ng.llm_configs import get_default_llm_configs
        self.LLM_CONFIGS = get_default_llm_configs()
        # Fetch and update pricing for OpenRouter models in background (non-blocking)
        pricing_thread = threading.Thread(
            target=self._update_openrouter_pricing, daemon=True
        )
        pricing_thread.start()

    def _log_initialization(self, message: str, level: str = "INFO"):
        """Log initialization messages"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self._initialization_logs.append(log_entry)
        print(log_entry)  # Also print to console for real-time feedback

    def _load_pricing_from_json(self, model_names: list[str]) -> dict[str, Any] | None:
        """Load pricing from JSON snapshot file as fallback.

        Returns:
            Pricing map (model_name -> {prompt_price_per_1k, completion_price_per_1k}) or None
        """
        try:
            import json
            from pathlib import Path

            # Locate JSON file in agent_ng directory (next to llm_configs.py)
            current_file = Path(__file__).resolve()
            json_path = current_file.parent / "openrouter_pricing.json"

            if not json_path.exists():
                return None

            with json_path.open("r", encoding="utf-8") as f:
                pricing_data = json.load(f)

            # Filter to only requested models
            pricing_map = {}
            for model_name in model_names:
                # Free models always have 0.0 pricing regardless of JSON values
                if ":free" in model_name.lower():
                    pricing_map[model_name] = {
                        "prompt_price_per_1k": 0.0,
                        "completion_price_per_1k": 0.0,
                    }
                    continue

                # Try exact match first, then base model (without variant)
                pricing = pricing_data.get(model_name) or pricing_data.get(
                    model_name.split(":")[0]
                )
                if pricing:
                    pricing_map[model_name] = pricing

            return pricing_map if pricing_map else None
        except Exception as e:
            logging.getLogger(__name__).debug("Failed to load pricing from JSON: %s", e)
            return None

    def _update_openrouter_pricing(self) -> None:
        """Fetch and update pricing for OpenRouter models at startup.

        Fallback chain:
        1. API fetch from `/models` endpoint (if enabled)
        2. JSON snapshot file (if API fails or disabled)

        If neither source provides pricing, models will use 0.0 (unknown pricing).
        Updates model configs in memory (persistent for this agent run).
        """
        # Check if runtime pricing fetch is enabled
        fetch_at_startup = _parse_bool(
            os.getenv("OPENROUTER_FETCH_PRICING_AT_STARTUP"), True
        )

        config = self.LLM_CONFIGS.get(LLMProvider.OPENROUTER)
        if not config or not config.models:
            return

        # Extract model names
        model_names = [m.get("model", "") for m in config.models if m.get("model")]
        if not model_names:
            return

        pricing_map = None
        pricing_source = None

        # Step 1: Try API fetch (if enabled)
        if fetch_at_startup:
            try:
                api_key = self._get_api_key(config)
                if api_key:
                    base_url = os.getenv(
                        config.api_base_env or "OPENROUTER_BASE_URL",
                        "https://openrouter.ai/api/v1",
                    )
                    self._log_initialization(
                        f"Fetching pricing via endpoints API for {len(model_names)} OpenRouter models (using interquartile mean pricing)...",
                        "INFO",
                    )

                    # Import here to avoid circular dependency
                    from agent_ng.utils.openrouter_pricing import (
                        fetch_pricing_via_endpoints,
                    )

                    # Fetch pricing using /endpoints API and use interquartile mean across endpoints
                    pricing_map = fetch_pricing_via_endpoints(
                        model_names, api_key, base_url
                    )
                    if pricing_map:
                        pricing_source = "API"
            except Exception as e:
                logging.getLogger(__name__).debug("API pricing fetch failed: %s", e)

        # Step 2: Fallback to JSON snapshot (if API failed or disabled)
        if not pricing_map:
            pricing_map = self._load_pricing_from_json(model_names)
            if pricing_map:
                pricing_source = "JSON snapshot"
                self._log_initialization(
                    f"Loaded pricing from JSON snapshot for {len(pricing_map)} models",
                    "INFO",
                )

        # If no pricing found, models will use 0.0 (unknown pricing)
        if not pricing_map:
            if fetch_at_startup:
                self._log_initialization(
                    "No pricing data available (API fetch failed, JSON not found). "
                    "Models will use 0.0 pricing (unknown cost).",
                    "WARNING",
                )
            else:
                self._log_initialization(
                    "Runtime pricing fetch disabled, JSON not found. "
                    "Models will use 0.0 pricing (unknown cost).",
                    "INFO",
                )

        # Update model configs in memory (only if pricing found from API or JSON)
        if pricing_map:
            updated_count = 0
            for model_config in config.models:
                model_name = model_config.get("model", "")
                if not model_name:
                    continue

                # Free models always have 0.0 pricing regardless of API/JSON values
                if ":free" in model_name.lower():
                    model_config["prompt_price_per_1k"] = 0.0
                    model_config["completion_price_per_1k"] = 0.0
                    updated_count += 1
                    continue

                pricing = pricing_map.get(model_name) or pricing_map.get(
                    model_name.split(":")[0]
                )
                if pricing:
                    model_config["prompt_price_per_1k"] = pricing.get(
                        "prompt_price_per_1k", 0.0
                    )
                    model_config["completion_price_per_1k"] = pricing.get(
                        "completion_price_per_1k", 0.0
                    )
                    updated_count += 1
                # If no pricing found, pricing remains None (unknown) in config

            if updated_count > 0:
                self._log_initialization(
                    f"Updated pricing for {updated_count}/{len(model_names)} OpenRouter models from {pricing_source}",
                    "INFO",
                )

    def _get_api_key(
        self, config: LLMConfig, api_key_override: str | None = None
    ) -> str | None:
        """Get API key using unified resolution (override → session → env)"""
        from agent_ng.key_resolution import get_provider_api_key

        return get_provider_api_key(
            provider=config.type_str,
            override_key=api_key_override,
        )

    def _load_allowed_providers(self) -> set[str] | None:
        """Load allowed providers from env. Empty/missing => allow all.

        Var: LLM_ALLOWED_PROVIDERS
        Comma separated list, case-insensitive.
        """
        try:
            load_dotenv()
        except Exception as exc:
            # Non-critical: continue without dotenv if it fails to load
            logging.getLogger(__name__).debug("Failed to load dotenv: %s", exc)
        raw = os.environ.get("LLM_ALLOWED_PROVIDERS") or ""
        normalized = [s.strip().lower() for s in raw.split(",") if s.strip()]
        if not normalized:
            return None
        valid = {p.value for p in LLMProvider}
        allowed = {p for p in normalized if p in valid}
        if not allowed:
            return None
        self._log_initialization(
            f"Allowed providers from env: {sorted(allowed)}", "INFO"
        )
        return allowed

    def _is_provider_allowed(self, provider: LLMProvider) -> bool:
        """Check if provider passes the allowlist (or allow all if None)."""
        if self._allowed_providers is None:
            return True
        return provider.value in self._allowed_providers

    def _initialize_gemini_llm(
        self,
        config: LLMConfig,
        model_config: dict[str, Any],
        api_key_override: str | None = None,
    ) -> Any | None:
        """Initialize Gemini LLM instance"""
        api_key = self._get_api_key(config, api_key_override=api_key_override)
        if not api_key:
            return None

        try:
            llm = ChatGoogleGenerativeAI(
                model=model_config["model"],
                google_api_key=api_key,
                temperature=model_config.get("temperature", 0),
                max_tokens=model_config.get("max_tokens", 2000000),
                disable_streaming=False,  # Enable streaming
            )
            self._log_initialization(
                f"Successfully initialized {config.name} - {model_config['model']}"
            )
            return llm
        except Exception as e:
            self._log_initialization(
                f"Failed to initialize {config.name}: {str(e)}", "ERROR"
            )
            return None

    def _initialize_groq_llm(
        self,
        config: LLMConfig,
        model_config: dict[str, Any],
        api_key_override: str | None = None,
    ) -> Any | None:
        """Initialize Groq LLM instance"""
        api_key = self._get_api_key(config, api_key_override=api_key_override)
        if not api_key:
            return None

        try:
            llm = ChatGroq(
                model=model_config["model"],
                groq_api_key=api_key,
                temperature=model_config.get("temperature", 0),
                max_tokens=model_config.get("max_tokens", 8192),
                streaming=True,  # Enable streaming
            )
            self._log_initialization(
                f"Successfully initialized {config.name} - {model_config['model']}"
            )
            return llm
        except Exception as e:
            self._log_initialization(
                f"Failed to initialize {config.name}: {str(e)}", "ERROR"
            )
            return None

    def _initialize_huggingface_llm(
        self,
        config: LLMConfig,
        model_config: dict[str, Any],
        api_key_override: str | None = None,
    ) -> Any | None:
        """Initialize HuggingFace LLM instance"""
        api_key = self._get_api_key(config, api_key_override=api_key_override)
        if not api_key:
            return None

        try:
            # Convert model to repo_id for HuggingFace
            repo_id = model_config["model"]
            task = model_config.get("task", "text-generation")

            llm = ChatHuggingFace(
                llm=HuggingFaceEndpoint(
                    repo_id=repo_id,
                    task=task,
                    huggingfacehub_api_token=api_key,
                    max_new_tokens=model_config.get("max_new_tokens", 1024),
                    do_sample=model_config.get("do_sample", False),
                    temperature=model_config.get("temperature", 0),
                )
            )
            self._log_initialization(
                f"Successfully initialized {config.name} - {model_config['model']}"
            )
            return llm
        except Exception as e:
            self._log_initialization(
                f"Failed to initialize {config.name}: {str(e)}", "ERROR"
            )
            return None

    def _initialize_openrouter_llm(
        self,
        config: LLMConfig,
        model_config: dict[str, Any],
        api_key_override: str | None = None,
    ) -> Any | None:
        """Initialize OpenRouter LLM instance"""
        api_key = self._get_api_key(config, api_key_override=api_key_override)
        if not api_key:
            return None

        try:
            base_url = os.getenv(config.api_base_env, "https://openrouter.ai/api/v1")  # // pragma: allowlist secret
            from agent_ng.openrouter_native_chat import create_openrouter_native_chat_model

            llm = create_openrouter_native_chat_model(
                model_name=model_config["model"],
                base_url=base_url,
                api_key=api_key,
                temperature=float(model_config.get("temperature", 0)),
                max_tokens=int(model_config.get("max_tokens", 2048)),
            )
            # LangSmith tracing is handled via @traceable decorators
            self._log_initialization(
                f"Successfully initialized {config.name} - {model_config['model']}"
            )
            return llm
        except Exception as e:
            self._log_initialization(
                f"Failed to initialize {config.name}: {str(e)}", "ERROR"
            )
            return None

    def _initialize_openai_llm(
        self,
        config: LLMConfig,
        model_config: dict[str, Any],
        api_key_override: str | None = None,
    ) -> Any | None:
        """Initialize a generic OpenAI-compatible LLM instance.

        Uses OpenRouterNativeChatModel (native OpenAI SDK) so the full usage
        dict is preserved.  Cost is not returned by api.openai.com itself;
        when pointed at a billing-aware compatible endpoint (e.g. Polza.ai via
        a dedicated provider), cost handling should be added in a separate
        normalizing callback — see openrouter_usage_accounting.py.

        Env vars:
            OPENAI_API_KEY  — required
            OPENAI_BASE_URL — optional, defaults to https://api.openai.com/v1
        """
        api_key = self._get_api_key(config, api_key_override=api_key_override)
        if not api_key:
            return None

        try:
            base_url = os.getenv(
                config.api_base_env or "OPENAI_BASE_URL",
                "https://api.openai.com/v1",
            )
            from agent_ng.openrouter_native_chat import create_openrouter_native_chat_model

            llm = create_openrouter_native_chat_model(
                model_name=model_config["model"],
                base_url=base_url,
                api_key=api_key,
                temperature=float(model_config.get("temperature", 0)),
                max_tokens=int(model_config.get("max_tokens", 2048)),
                default_headers={"X-Title": "CMW Platform Agent"},
            )
            self._log_initialization(
                f"Successfully initialized {config.name} - {model_config['model']}"
            )
            return llm
        except Exception as e:
            self._log_initialization(
                f"Failed to initialize {config.name}: {str(e)}", "ERROR"
            )
            return None

    def _initialize_polza_llm(
        self,
        config: LLMConfig,
        model_config: dict[str, Any],
        api_key_override: str | None = None,
    ) -> Any | None:
        """Initialize Polza.ai LLM instance.

        Polza.ai (https://polza.ai/api/v1) is OpenAI-compatible; billing is
        reported in rubles (``usage.cost_rub``).  Uses the same native SDK
        wrapper as OpenRouter so the full usage dict reaches callbacks.

        Env vars:
            POLZA_API_KEY  — required
            POLZA_BASE_URL — optional, defaults to https://polza.ai/api/v1
        """
        api_key = self._get_api_key(config, api_key_override=api_key_override)
        if not api_key:
            return None

        try:
            base_url = os.getenv(
                config.api_base_env or "POLZA_BASE_URL",
                "https://polza.ai/api/v1",
            )
            from agent_ng.openrouter_native_chat import create_openrouter_native_chat_model

            llm = create_openrouter_native_chat_model(
                model_name=model_config["model"],
                base_url=base_url,
                api_key=api_key,
                temperature=float(model_config.get("temperature", 0)),
                max_tokens=int(model_config.get("max_tokens", 2048)),
                default_headers={"X-Title": "CMW Platform Agent"},
            )
            self._log_initialization(
                f"Successfully initialized {config.name} - {model_config['model']}"
            )
            return llm
        except Exception as e:
            self._log_initialization(
                f"Failed to initialize {config.name}: {str(e)}", "ERROR"
            )
            return None

    def _initialize_mistral_llm(
        self,
        config: LLMConfig,
        model_config: dict[str, Any],
        api_key_override: str | None = None,
    ) -> Any | None:
        """Initialize Mistral LLM instance"""
        api_key = self._get_api_key(config, api_key_override=api_key_override)
        if not api_key:
            return None

        try:
            llm = ChatMistralAI(
                model=model_config["model"],
                mistral_api_key=api_key,
                temperature=model_config.get("temperature", 0),
                max_tokens=model_config.get("max_tokens", 2048),
                streaming=True,  # Enable streaming
            )
            # LangSmith tracing handled via @traceable decorators
            self._log_initialization(
                f"Successfully initialized {config.name} - {model_config['model']}"
            )
            return llm
        except Exception as e:
            self._log_initialization(
                f"Failed to initialize {config.name}: {str(e)}", "ERROR"
            )
            return None

    def _initialize_gigachat_llm(
        self,
        config: LLMConfig,
        model_config: dict[str, Any],
        api_key_override: str | None = None,
    ) -> Any | None:
        """Initialize GigaChat LLM instance"""
        try:
            # Use the newer langchain-gigachat package (recommended)
            from langchain_gigachat.chat_models import GigaChat as LC_GigaChat
        except ImportError:
            try:
                # Fallback to langchain-community (deprecated but still works)
                from langchain_community.chat_models import GigaChat as LC_GigaChat

                self._log_initialization(
                    "Using deprecated langchain-community.GigaChat. Consider upgrading to langchain-gigachat",
                    "WARNING",
                )
            except ImportError as e:
                self._log_initialization(
                    f"Neither langchain-gigachat nor langchain-community is installed: {e}",
                    "ERROR",
                )
                self._log_initialization(
                    "Install with: pip install langchain-gigachat", "INFO"
                )
                self._log_initialization("Or: pip install langchain-community", "INFO")
                return None

        # Check for required environment variables
        api_key = self._get_api_key(config, api_key_override=api_key_override)
        if not api_key:
            self._log_initialization(
                f"{config.api_key_env} not found in environment variables. Skipping GigaChat...",
                "WARNING",
            )
            self._log_initialization(
                "To use GigaChat, set GIGACHAT_API_KEY in your environment variables.",
                "INFO",
            )
            return None

        scope = os.environ.get(config.scope_env, "GIGACHAT_SCOPE")
        if not scope:
            self._log_initialization(
                "GIGACHAT_SCOPE not found in environment variables. Using default scope...",
                "WARNING",
            )
            self._log_initialization(
                "Available scopes: GIGACHAT_API_PERS, GIGACHAT_API_B2B, GIGACHAT_API_CORP",
                "INFO",
            )
            scope = "GIGACHAT_API_PERS"  # Default scope

        verify_ssl_env = os.environ.get(config.verify_ssl_env, "GIGACHAT_VERIFY_SSL")
        if verify_ssl_env is None:
            verify_ssl_env = "false"
        verify_ssl = str(verify_ssl_env).strip().lower() in ("1", "true", "yes", "y")

        # Get additional optional parameters
        base_url = os.environ.get(
            "GIGACHAT_BASE_URL", "https://gigachat.devices.sberbank.ru/api/v1"
        )
        timeout = int(os.environ.get("GIGACHAT_TIMEOUT", "30"))

        try:
            # Initialize LangChain GigaChat client with proper parameters
            giga_chat = LC_GigaChat(
                credentials=api_key,
                model=model_config["model"],
                verify_ssl_certs=verify_ssl,
                scope=scope,
                base_url=base_url,
                timeout=timeout,
                temperature=model_config.get("temperature", 0),
                max_tokens=model_config.get("max_tokens", 2048),
                top_p=model_config.get("top_p", 0.9),
                repetition_penalty=model_config.get("repetition_penalty", 1.0),
                streaming=True,  # Enable streaming
            )

            self._log_initialization(
                f"Successfully initialized {config.name} with model {model_config['model']}",
                "INFO",
            )
            return giga_chat

        except Exception as e:
            self._log_initialization(
                f"Failed to initialize {config.name}: {str(e)}", "ERROR"
            )
            return None

    def _initialize_llm_instance(
        self,
        provider: LLMProvider,
        model_index: int = 0,
        api_key_override: str | None = None,
    ) -> LLMInstance | None:
        """Initialize a specific LLM instance"""
        config = self.LLM_CONFIGS.get(provider)
        if not config:
            self._log_initialization(f"Unknown provider: {provider}", "ERROR")
            return None

        if model_index >= len(config.models):
            self._log_initialization(
                f"Model index {model_index} out of range for {provider}", "ERROR"
            )
            return None

        model_config = config.models[model_index]

        # Initialize based on provider type
        llm = None
        if provider == LLMProvider.GEMINI:
            llm = self._initialize_gemini_llm(
                config, model_config, api_key_override=api_key_override
            )
        elif provider == LLMProvider.GROQ:
            llm = self._initialize_groq_llm(
                config, model_config, api_key_override=api_key_override
            )
        elif provider == LLMProvider.HUGGINGFACE:
            llm = self._initialize_huggingface_llm(
                config, model_config, api_key_override=api_key_override
            )
        elif provider == LLMProvider.OPENAI:
            llm = self._initialize_openai_llm(
                config, model_config, api_key_override=api_key_override
            )
        elif provider == LLMProvider.OPENROUTER:
            llm = self._initialize_openrouter_llm(
                config, model_config, api_key_override=api_key_override
            )
        elif provider == LLMProvider.POLZA:
            llm = self._initialize_polza_llm(
                config, model_config, api_key_override=api_key_override
            )
        elif provider == LLMProvider.MISTRAL:
            llm = self._initialize_mistral_llm(
                config, model_config, api_key_override=api_key_override
            )
        elif provider == LLMProvider.GIGACHAT:
            llm = self._initialize_gigachat_llm(
                config, model_config, api_key_override=api_key_override
            )

        if llm is None:
            return None

        # Apply Mistral wrapper if this is a Mistral model (regardless of provider)
        if MistralWrapper and is_mistral_model(model_config["model"]):
            llm = MistralWrapper(llm)
            self._log_initialization(
                f"Applied Mistral wrapper to {model_config['model']}", "INFO"
            )

        # Create LLM instance wrapper
        instance = LLMInstance(
            llm=llm,
            provider=provider,
            model_name=model_config["model"],
            config=model_config,
            initialized_at=time.time(),
            last_used=time.time(),
            is_healthy=True,
            bound_tools=False,
        )

        return instance

    def _get_instance_key(self, provider: LLMProvider, model_index: int = 0) -> str:
        """Generate a unique key for an LLM instance"""
        return f"{provider.value}_{model_index}"

    def get_llm(
        self,
        provider: str,
        use_tools: bool = True,
        model_index: int = 0,
        api_key_override: str | None = None,
    ) -> LLMInstance | None:
        """
        Get an LLM instance for the specified provider.

        Args:
            provider: Provider name (e.g., "gemini", "groq")
            use_tools: Whether the LLM should support tools
            model_index: Index of the model to use (0 for first model)
            api_key_override: Optional API key to use instead of env var

        Returns:
            LLMInstance or None if initialization failed
        """
        try:
            provider_enum = LLMProvider(provider.lower())
        except ValueError:
            self._log_initialization(f"Invalid provider: {provider}", "ERROR")
            return None
        # Enforce allowlist
        if not self._is_provider_allowed(provider_enum):
            self._log_initialization(
                f"Provider '{provider_enum.value}' is not allowed by LLM_ALLOWED_PROVIDERS",
                "ERROR",
            )
            return None

        instance_key = self._get_instance_key(provider_enum, model_index)

        with self._lock:
            # Check if instance already exists and is healthy
            if instance_key in self._instances:
                instance = self._instances[instance_key]
                if instance.is_healthy:
                    instance.last_used = time.time()
                    return instance
                else:
                    # Remove unhealthy instance
                    del self._instances[instance_key]

            # Initialize new instance
            instance = self._initialize_llm_instance(
                provider_enum, model_index, api_key_override=api_key_override
            )
            if instance:
                # Bind tools if requested and provider supports them
                if use_tools and self.LLM_CONFIGS.get(provider_enum, {}).tool_support:
                    tools_list = self.get_tools()
                    if tools_list:
                        try:
                            instance.llm = instance.llm.bind_tools(tools_list)
                            instance.bound_tools = True
                            # Calculate and set global average tool size (once ever)
                            try:
                                from agent_ng.token_budget import (
                                    _GLOBAL_AVG_TOOL_SIZE,
                                    _calculate_avg_tool_size,
                                )

                                # Get bound tools as dicts from kwargs
                                kwargs = getattr(instance.llm, "kwargs", None)
                                if isinstance(kwargs, dict):
                                    bound_tools = kwargs.get("tools")
                                    if bound_tools and _GLOBAL_AVG_TOOL_SIZE is None:
                                        _GLOBAL_AVG_TOOL_SIZE = (
                                            _calculate_avg_tool_size(bound_tools)
                                        )
                            except Exception as exc:
                                # Non-critical: continue if average calculation fails
                                # Tools will still work, just with default 600 token estimate
                                logging.getLogger(__name__).debug(
                                    "Non-critical: failed to calculate tool averages: %s",
                                    exc,
                                )
                            self._log_initialization(
                                f"Tools bound to {provider} instance ({len(tools_list)} tools)",
                                "INFO",
                            )
                        except Exception as e:
                            self._log_initialization(
                                f"Failed to bind tools to {provider}: {e}", "WARNING"
                            )
                            # Don't fail the entire initialization if tool binding fails
                            instance.bound_tools = False

                self._instances[instance_key] = instance
                return instance

        return None

    def get_llm_with_tools(
        self, provider: str, model_index: int = 0
    ) -> LLMInstance | None:
        """
        Get an LLM instance with tools bound for the specified provider.

        Args:
            provider: Provider name (e.g., "gemini", "groq")
            model_index: Index of the model to use (0 for first model)

        Returns:
            LLMInstance or None if initialization failed
        """
        return self.get_llm(provider, use_tools=True, model_index=model_index)

    def create_new_llm_instance(
        self, provider: str, model_index: int = 0, api_key_override: str | None = None
    ) -> LLMInstance | None:
        """
        Create a NEW LLM instance for the specified provider (not cached).
        This is used for session isolation - each session gets its own instance.

        Args:
            provider: Provider name (e.g., "gemini", "groq")
            model_index: Index of the model to use (0 for first model)
            api_key_override: Optional API key to use instead of env var

        Returns:
            LLMInstance or None if initialization failed
        """
        try:
            provider_enum = LLMProvider(provider.lower())
        except ValueError:
            self._log_initialization(f"Invalid provider: {provider}", "ERROR")
            return None
        # Enforce allowlist
        if not self._is_provider_allowed(provider_enum):
            self._log_initialization(
                f"Provider '{provider_enum.value}' is not allowed by LLM_ALLOWED_PROVIDERS",
                "ERROR",
            )
            return None

        # Create new instance without caching
        instance = self._initialize_llm_instance(
            provider_enum, model_index, api_key_override=api_key_override
        )
        if instance:
            # Bind tools if provider supports them
            if self.LLM_CONFIGS.get(provider_enum, {}).tool_support:
                tools_list = self.get_tools()
                if tools_list:
                    try:
                        instance.llm = instance.llm.bind_tools(tools_list)
                        instance.bound_tools = True
                        # Calculate and set global average tool size (once ever)
                        try:
                            from agent_ng.token_budget import (
                                _GLOBAL_AVG_TOOL_SIZE,
                                _calculate_avg_tool_size,
                            )

                            kwargs = getattr(instance.llm, "kwargs", None)
                            if isinstance(kwargs, dict):
                                bound_tools = kwargs.get("tools")
                                if bound_tools and _GLOBAL_AVG_TOOL_SIZE is None:
                                    _GLOBAL_AVG_TOOL_SIZE = _calculate_avg_tool_size(
                                        bound_tools
                                    )
                        except Exception as exc:
                            # Non-critical: continue if average calculation fails
                            # Tools will still work, just with default 600 token estimate
                            logging.getLogger(__name__).debug(
                                "Non-critical: failed to calculate tool averages: %s",
                                exc,
                            )
                        self._log_initialization(
                            f"Tools bound to NEW {provider} instance ({len(tools_list)} tools)",
                            "INFO",
                        )
                    except Exception as e:
                        self._log_initialization(
                            f"Failed to bind tools to NEW {provider}: {e}", "WARNING"
                        )
                        instance.bound_tools = False

            return instance
        return None

    def _find_model_index(self, provider: LLMProvider, model_name: str) -> int | None:
        """
        Find the index of a model by name for a given provider.

        Args:
            provider: The LLM provider
            model_name: The model name to find (exact match, case-sensitive)

        Returns:
            Model index if found, None otherwise
        """
        config = self.LLM_CONFIGS.get(provider)
        if not config or not config.models:
            return None

        # Strip whitespace and do exact match
        model_name = model_name.strip()
        for index, model_config in enumerate(config.models):
            config_model = model_config.get("model", "").strip()
            if config_model == model_name:
                return index

        return None

    def _get_configured_provider_and_model_index(
        self,
        provider_str_override: str | None = None,
    ) -> tuple[LLMProvider | None, int]:
        """Get configured provider and model index from config/env.

        When ``provider_str_override`` is set (e.g. session config for API-key
        bucket), use that provider but still resolve ``AGENT_DEFAULT_MODEL`` /
        settings ``default_model`` within that provider's catalog — never force
        model index 0, which would ignore the user's chosen default model.
        """
        import os

        try:
            from agent_ng.agent_config import get_llm_settings

            llm_settings = get_llm_settings()
            default_provider = llm_settings.get("default_provider", "openrouter")
            default_model = llm_settings.get("default_model")
        except ImportError:
            default_provider = os.environ.get("AGENT_PROVIDER", "openrouter")
            default_model = os.environ.get("AGENT_DEFAULT_MODEL")

        if provider_str_override and str(provider_str_override).strip():
            provider = str(provider_str_override).strip()
        else:
            provider = default_provider

        try:
            provider_enum = LLMProvider(provider.lower())
        except ValueError:
            return None, 0

        model_index = 0
        if default_model and str(default_model).strip():
            found_index = self._find_model_index(
                provider_enum, str(default_model).strip()
            )
            if found_index is not None:
                model_index = found_index

        return provider_enum, model_index

    def get_agent_llm(self) -> LLMInstance | None:
        """Get the single LLM instance from AGENT_PROVIDER and AGENT_DEFAULT_MODEL"""
        provider_enum, model_index = self._get_configured_provider_and_model_index()
        if not provider_enum:
            return None

        instance = self.get_llm(provider_enum.value, model_index=model_index)
        if instance:
            self._log_initialization(
                f"✅ Using {provider_enum.value}/{instance.model_name}", "INFO"
            )
        return instance

    def get_available_providers(self) -> list[str]:
        """Get list of available providers that can be initialized"""
        available = []
        for provider in LLMProvider:
            # Check if API key is available without initializing the LLM
            config = self.LLM_CONFIGS.get(provider)
            if (
                config
                and self._get_api_key(config)
                and self._is_provider_allowed(provider)
            ):
                available.append(provider.value)
        # Sort provider identifiers alphabetically for stable UI ordering
        return sorted(available)

    def get_provider_config(self, provider: str) -> LLMConfig | None:
        """Get configuration for a specific provider"""
        try:
            provider_enum = LLMProvider(provider.lower())
            return self.LLM_CONFIGS.get(provider_enum)
        except ValueError:
            return None

    def health_check(self) -> dict[str, Any]:
        """Perform health check on all instances"""
        current_time = time.time()
        if current_time - self._last_health_check < self._health_check_interval:
            return {"status": "skipped", "reason": "too_recent"}

        self._last_health_check = current_time

        with self._lock:
            healthy_count = 0
            total_count = len(self._instances)

            for instance in self._instances.values():
                # Simple health check - could be enhanced with actual API calls
                if (
                    instance.is_healthy and (current_time - instance.last_used) < 3600
                ):  # 1 hour
                    healthy_count += 1

        return {
            "status": "completed",
            "healthy_instances": healthy_count,
            "total_instances": total_count,
            "timestamp": current_time,
        }

    def get_initialization_logs(self) -> list[str]:
        """Get initialization logs"""
        return self._initialization_logs.copy()

    def clear_logs(self):
        """Clear initialization logs"""
        self._initialization_logs.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about managed LLM instances"""
        with self._lock:
            stats = {
                "total_instances": len(self._instances),
                "providers": {},
                "initialization_logs_count": len(self._initialization_logs),
            }

            for instance in self._instances.values():
                provider = instance.provider.value
                if provider not in stats["providers"]:
                    stats["providers"][provider] = {
                        "count": 0,
                        "healthy": 0,
                        "models": [],
                    }

                stats["providers"][provider]["count"] += 1
                if instance.is_healthy:
                    stats["providers"][provider]["healthy"] += 1
                stats["providers"][provider]["models"].append(instance.model_name)

        return stats

    def get_current_llm_context_window(self) -> int:
        """Get the context window size for the current LLM instance"""
        current_instance = self.get_agent_llm()
        if current_instance and current_instance.config:
            # Get token_limit from the current instance's config
            return current_instance.config.get("token_limit", 0)

        # No fallback needed - if no current instance, return 0
        return 0

    def get_tools(self) -> list[Any]:
        """Get all available tools from tools module (avoiding duplicates) - cached"""
        # Return cached tools if available
        if hasattr(self, "_cached_tools"):
            return self._cached_tools

        tool_list = []
        tool_names = set()  # Track tool names to avoid duplicates

        # Load tools from main tools module (primary source)
        try:
            import tools.tools as tools_module

            self._load_tools_from_module(
                tools_module, tool_list, "tools.tools", tool_names
            )
        except ImportError:
            self._log_initialization("Could not import tools.tools module", "WARNING")

        # Load tools from attributes_tools submodule (only if not already loaded)
        try:
            import tools.attributes_tools as attributes_tools_module

            self._load_tools_from_module(
                attributes_tools_module, tool_list, "tools.attributes_tools", tool_names
            )
        except ImportError:
            self._log_initialization(
                "Could not import tools.attributes_tools module", "WARNING"
            )

        # Load tools from applications_tools submodule (only if not already loaded)
        try:
            import tools.applications_tools as applications_tools_module

            self._load_tools_from_module(
                applications_tools_module,
                tool_list,
                "tools.applications_tools",
                tool_names,
            )
        except ImportError:
            self._log_initialization(
                "Could not import tools.applications_tools module", "WARNING"
            )

        # Load tools from templates_tools submodule (only if not already loaded)
        try:
            import tools.templates_tools as templates_tools_module

            self._load_tools_from_module(
                templates_tools_module, tool_list, "tools.templates_tools", tool_names
            )
        except ImportError:
            self._log_initialization(
                "Could not import tools.templates_tools module", "WARNING"
            )

        # Cache the tools list
        self._cached_tools = tool_list
        return tool_list

    def _load_tools_from_module(
        self,
        module,
        tool_list: list[Any],
        module_name: str,
        tool_names: set | None = None,
    ):
        """Load tools from a specific module (avoiding duplicates)"""
        if tool_names is None:
            tool_names = set()

        try:
            from langchain_core.tools.base import BaseTool as _BaseTool
        except ImportError:
            _BaseTool = None  # type: ignore[assignment,misc]

        _EXCLUDED = {
            "CmwAgent",
            "CodeInterpreter",
            "submit_answer",
            "submit_intermediate_step",
            "web_search_deep_research_exa_ai",
        }

        for name, obj in module.__dict__.items():
            # Identify LangChain tools robustly across versions:
            # - LangChain 1.x: isinstance(obj, BaseTool) covers both @tool
            #   StructuredTool instances and BaseTool subclass instances.
            # - Fallback duck-type check for edge cases.
            is_tool = (
                _BaseTool is not None and isinstance(obj, _BaseTool)
            ) or (
                callable(obj)
                and not isinstance(obj, type)
                and hasattr(obj, "name")
                and hasattr(obj, "description")
                and hasattr(obj, "args_schema")
            )
            if (
                is_tool
                and not name.startswith("_")
                and name not in _EXCLUDED
            ):
                if (
                    hasattr(obj, "name")
                    and hasattr(obj, "description")
                    and hasattr(obj, "args_schema")
                    and (hasattr(obj, "func") or hasattr(obj, "run"))
                ):
                    # This is a proper @tool decorated function
                    tool_name = obj.name

                    # Skip if already loaded
                    if tool_name in tool_names:
                        self._log_initialization(
                            f"Skipped duplicate tool: {tool_name} from {module_name}",
                            "DEBUG",
                        )
                        continue

                    tool_list.append(obj)
                    tool_names.add(tool_name)
                    self._log_initialization(
                        f"Loaded LangChain tool: {name} from {module_name}", "INFO"
                    )


# Global instance for application-wide use
_llm_manager = None
_manager_lock = threading.Lock()


def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance (singleton pattern)"""
    global _llm_manager
    if _llm_manager is None:
        with _manager_lock:
            if _llm_manager is None:
                _llm_manager = LLMManager()
    return _llm_manager


def reset_llm_manager_singleton() -> None:
    """Clear the cached manager so the next ``get_llm_manager()`` reads fresh env.

    Used by tests and scripts that call ``load_dotenv(override=True)`` before
    constructing LLMs (singleton would otherwise keep earlier instances).
    """
    global _llm_manager
    with _manager_lock:
        _llm_manager = None
