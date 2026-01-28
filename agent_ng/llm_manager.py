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

import logging
import os
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain imports
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI

# Local imports with robust fallback handling
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .utils import ensure_valid_answer
    from .provider_adapters import MistralWrapper, is_mistral_model
    from .langsmith_config import get_langsmith_config, get_openai_wrapper
    from .logging_config import _parse_bool
except ImportError:
    try:
        from agent_ng.utils import ensure_valid_answer
        from agent_ng.provider_adapters import MistralWrapper, is_mistral_model
        from agent_ng.langsmith_config import get_langsmith_config, get_openai_wrapper
        from agent_ng.logging_config import _parse_bool
    except ImportError as e:
        print(f"💥 CRITICAL ERROR: Cannot import required modules in llm_manager!")
        print(f"   Import failed: {e}")
        print("🔧 Please check that all dependencies are installed and modules exist")
        raise ImportError(f"Failed to import required modules in llm_manager: {e}")


class LLMProvider(Enum):
    """Enumeration of supported LLM providers"""
    GEMINI = "gemini"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    OPENROUTER = "openrouter"
    MISTRAL = "mistral"
    GIGACHAT = "gigachat"


@dataclass
class LLMConfig:
    """Configuration for a specific LLM provider"""
    name: str
    type_str: str
    api_key_env: str
    api_base_env: Optional[str] = None
    scope_env: Optional[str] = None
    verify_ssl_env: Optional[str] = None
    max_history: int = 20
    tool_support: bool = False
    force_tools: bool = False
    models: List[Dict[str, Any]] = None
    token_per_minute_limit: Optional[int] = None
    enable_chunking: bool = True

    def __post_init__(self):
        if self.models is None:
            self.models = []


@dataclass
class LLMInstance:
    """Wrapper for an initialized LLM instance with metadata"""
    llm: Any
    provider: LLMProvider
    model_name: str
    config: Dict[str, Any]
    initialized_at: float
    last_used: float
    is_healthy: bool = True
    error_count: int = 0
    last_error: Optional[str] = None
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
        self._instances: Dict[str, LLMInstance] = {}
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
        # Fetch and update pricing for OpenRouter models at startup
        self._update_openrouter_pricing()

    def _log_initialization(self, message: str, level: str = "INFO"):
        """Log initialization messages"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self._initialization_logs.append(log_entry)
        print(log_entry)  # Also print to console for real-time feedback

    def _load_pricing_from_json(self, model_names: List[str]) -> Optional[Dict[str, Any]]:
        """Load pricing from JSON snapshot file as fallback.

        Returns:
            Pricing map (model_name -> {prompt_price_per_1k, completion_price_per_1k}) or None
        """
        try:
            from pathlib import Path
            import json

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
                        "completion_price_per_1k": 0.0
                    }
                    continue

                # Try exact match first, then base model (without variant)
                pricing = pricing_data.get(model_name) or pricing_data.get(model_name.split(":")[0])
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
        fetch_at_startup = _parse_bool(os.getenv("OPENROUTER_FETCH_PRICING_AT_STARTUP"), True)

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
                    base_url = os.getenv(config.api_base_env or "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                    self._log_initialization(
                        f"Fetching pricing via endpoints API for {len(model_names)} OpenRouter models (averaging endpoints)...", "INFO"
                    )

                    # Import here to avoid circular dependency
                    from agent_ng.utils.openrouter_pricing import fetch_pricing_via_endpoints

                    # Fetch pricing using /endpoints API and average across endpoints
                    pricing_map = fetch_pricing_via_endpoints(model_names, api_key, base_url)
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
                    f"Loaded pricing from JSON snapshot for {len(pricing_map)} models", "INFO"
                )

        # If no pricing found, models will use 0.0 (unknown pricing)
        if not pricing_map:
            if fetch_at_startup:
                self._log_initialization(
                    "No pricing data available (API fetch failed, JSON not found). "
                    "Models will use 0.0 pricing (unknown cost).", "WARNING"
                )
            else:
                self._log_initialization(
                    "Runtime pricing fetch disabled, JSON not found. "
                    "Models will use 0.0 pricing (unknown cost).", "INFO"
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

                pricing = pricing_map.get(model_name) or pricing_map.get(model_name.split(":")[0])
                if pricing:
                    model_config["prompt_price_per_1k"] = pricing["prompt_price_per_1k"]
                    model_config["completion_price_per_1k"] = pricing["completion_price_per_1k"]
                    updated_count += 1
                # If no pricing found, pricing remains None (unknown) in config

            if updated_count > 0:
                self._log_initialization(
                    f"Updated pricing for {updated_count}/{len(model_names)} OpenRouter models from {pricing_source}", "INFO"
                )


    def _get_api_key(self, config: LLMConfig) -> Optional[str]:
        """Get API key from environment variables"""
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            self._log_initialization(f"API key not found for {config.name} ({config.api_key_env})", "WARNING")
        return api_key

    def _load_allowed_providers(self) -> Optional[set[str]]:
        """Load allowed providers from env. Empty/missing => allow all.

        Var: LLM_ALLOWED_PROVIDERS
        Comma separated list, case-insensitive.
        """
        try:
            load_dotenv()
        except Exception as exc:
            # Non-critical: continue without dotenv if it fails to load
            logging.getLogger(__name__).debug(
                "Failed to load dotenv: %s", exc
            )
        raw = (
            os.environ.get("LLM_ALLOWED_PROVIDERS")
            or ""
        )
        normalized = [s.strip().lower() for s in raw.split(",") if s.strip()]
        if not normalized:
            return None
        valid = {p.value for p in LLMProvider}
        allowed = {p for p in normalized if p in valid}
        if not allowed:
            return None
        self._log_initialization(f"Allowed providers from env: {sorted(allowed)}", "INFO")
        return allowed

    def _is_provider_allowed(self, provider: LLMProvider) -> bool:
        """Check if provider passes the allowlist (or allow all if None)."""
        if self._allowed_providers is None:
            return True
        return provider.value in self._allowed_providers

    def _initialize_gemini_llm(self, config: LLMConfig, model_config: Dict[str, Any]) -> Optional[Any]:
        """Initialize Gemini LLM instance"""
        api_key = self._get_api_key(config)
        if not api_key:
            return None

        try:
            llm = ChatGoogleGenerativeAI(
                model=model_config["model"],
                google_api_key=api_key,
                temperature=model_config.get("temperature", 0),
                max_tokens=model_config.get("max_tokens", 2000000),
                disable_streaming=False  # Enable streaming
            )
            self._log_initialization(f"Successfully initialized {config.name} - {model_config['model']}")
            return llm
        except Exception as e:
            self._log_initialization(f"Failed to initialize {config.name}: {str(e)}", "ERROR")
            return None

    def _initialize_groq_llm(self, config: LLMConfig, model_config: Dict[str, Any]) -> Optional[Any]:
        """Initialize Groq LLM instance"""
        api_key = self._get_api_key(config)
        if not api_key:
            return None

        try:
            llm = ChatGroq(
                model=model_config["model"],
                groq_api_key=api_key,
                temperature=model_config.get("temperature", 0),
                max_tokens=model_config.get("max_tokens", 8192),
                streaming=True  # Enable streaming
            )
            self._log_initialization(f"Successfully initialized {config.name} - {model_config['model']}")
            return llm
        except Exception as e:
            self._log_initialization(f"Failed to initialize {config.name}: {str(e)}", "ERROR")
            return None

    def _initialize_huggingface_llm(self, config: LLMConfig, model_config: Dict[str, Any]) -> Optional[Any]:
        """Initialize HuggingFace LLM instance"""
        api_key = self._get_api_key(config)
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
                    temperature=model_config.get("temperature", 0)
                )
            )
            self._log_initialization(f"Successfully initialized {config.name} - {model_config['model']}")
            return llm
        except Exception as e:
            self._log_initialization(f"Failed to initialize {config.name}: {str(e)}", "ERROR")
            return None

    def _initialize_openrouter_llm(self, config: LLMConfig, model_config: Dict[str, Any]) -> Optional[Any]:
        """Initialize OpenRouter LLM instance"""
        api_key = self._get_api_key(config)
        if not api_key:
            return None

        try:
            base_url = os.getenv(config.api_base_env, "https://openrouter.ai/api/v1")
            llm = ChatOpenAI(
                model=model_config["model"],
                api_key=api_key,
                base_url=base_url,
                temperature=model_config.get("temperature", 0),
                max_tokens=model_config.get("max_tokens", 2048),
                streaming=True  # Enable streaming
            )
            # LangSmith tracing is handled via @traceable decorators
            self._log_initialization(f"Successfully initialized {config.name} - {model_config['model']}")
            return llm
        except Exception as e:
            self._log_initialization(f"Failed to initialize {config.name}: {str(e)}", "ERROR")
            return None

    def _initialize_mistral_llm(self, config: LLMConfig, model_config: Dict[str, Any]) -> Optional[Any]:
        """Initialize Mistral LLM instance"""
        api_key = self._get_api_key(config)
        if not api_key:
            return None

        try:
            llm = ChatMistralAI(
                model=model_config["model"],
                mistral_api_key=api_key,
                temperature=model_config.get("temperature", 0),
                max_tokens=model_config.get("max_tokens", 2048),
                streaming=True  # Enable streaming
            )
            # LangSmith tracing handled via @traceable decorators
            self._log_initialization(f"Successfully initialized {config.name} - {model_config['model']}")
            return llm
        except Exception as e:
            self._log_initialization(f"Failed to initialize {config.name}: {str(e)}", "ERROR")
            return None

    def _initialize_gigachat_llm(self, config: LLMConfig, model_config: Dict[str, Any]) -> Optional[Any]:
        """Initialize GigaChat LLM instance"""
        try:
            # Use the newer langchain-gigachat package (recommended)
            from langchain_gigachat.chat_models import GigaChat as LC_GigaChat
        except ImportError:
            try:
                # Fallback to langchain-community (deprecated but still works)
                from langchain_community.chat_models import GigaChat as LC_GigaChat
                self._log_initialization("Using deprecated langchain-community.GigaChat. Consider upgrading to langchain-gigachat", "WARNING")
            except ImportError as e:
                self._log_initialization(f"Neither langchain-gigachat nor langchain-community is installed: {e}", "ERROR")
                self._log_initialization("Install with: pip install langchain-gigachat", "INFO")
                self._log_initialization("Or: pip install langchain-community", "INFO")
                return None

        # Check for required environment variables
        api_key = self._get_api_key(config)
        if not api_key:
            self._log_initialization(f"{config.api_key_env} not found in environment variables. Skipping GigaChat...", "WARNING")
            self._log_initialization("To use GigaChat, set GIGACHAT_API_KEY in your environment variables.", "INFO")
            return None

        scope = os.environ.get(config.scope_env, "GIGACHAT_SCOPE")
        if not scope:
            self._log_initialization("GIGACHAT_SCOPE not found in environment variables. Using default scope...", "WARNING")
            self._log_initialization("Available scopes: GIGACHAT_API_PERS, GIGACHAT_API_B2B, GIGACHAT_API_CORP", "INFO")
            scope = "GIGACHAT_API_PERS"  # Default scope

        verify_ssl_env = os.environ.get(config.verify_ssl_env, "GIGACHAT_VERIFY_SSL")
        if verify_ssl_env is None:
            verify_ssl_env = "false"
        verify_ssl = str(verify_ssl_env).strip().lower() in ("1", "true", "yes", "y")

        # Get additional optional parameters
        base_url = os.environ.get("GIGACHAT_BASE_URL", "https://gigachat.devices.sberbank.ru/api/v1")
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
                streaming=True  # Enable streaming
            )

            self._log_initialization(f"Successfully initialized {config.name} with model {model_config['model']}", "INFO")
            return giga_chat

        except Exception as e:
            self._log_initialization(f"Failed to initialize {config.name}: {str(e)}", "ERROR")
            return None

    def _initialize_llm_instance(self, provider: LLMProvider, model_index: int = 0) -> Optional[LLMInstance]:
        """Initialize a specific LLM instance"""
        config = self.LLM_CONFIGS.get(provider)
        if not config:
            self._log_initialization(f"Unknown provider: {provider}", "ERROR")
            return None

        if model_index >= len(config.models):
            self._log_initialization(f"Model index {model_index} out of range for {provider}", "ERROR")
            return None

        model_config = config.models[model_index]

        # Initialize based on provider type
        llm = None
        if provider == LLMProvider.GEMINI:
            llm = self._initialize_gemini_llm(config, model_config)
        elif provider == LLMProvider.GROQ:
            llm = self._initialize_groq_llm(config, model_config)
        elif provider == LLMProvider.HUGGINGFACE:
            llm = self._initialize_huggingface_llm(config, model_config)
        elif provider == LLMProvider.OPENROUTER:
            llm = self._initialize_openrouter_llm(config, model_config)
        elif provider == LLMProvider.MISTRAL:
            llm = self._initialize_mistral_llm(config, model_config)
        elif provider == LLMProvider.GIGACHAT:
            llm = self._initialize_gigachat_llm(config, model_config)

        if llm is None:
            return None

        # Apply Mistral wrapper if this is a Mistral model (regardless of provider)
        if MistralWrapper and is_mistral_model(model_config["model"]):
            llm = MistralWrapper(llm)
            self._log_initialization(f"Applied Mistral wrapper to {model_config['model']}", "INFO")

        # Create LLM instance wrapper
        instance = LLMInstance(
            llm=llm,
            provider=provider,
            model_name=model_config["model"],
            config=model_config,
            initialized_at=time.time(),
            last_used=time.time(),
            is_healthy=True,
            bound_tools=False
        )

        return instance

    def _get_instance_key(self, provider: LLMProvider, model_index: int = 0) -> str:
        """Generate a unique key for an LLM instance"""
        return f"{provider.value}_{model_index}"

    def get_llm(self, provider: str, use_tools: bool = True, model_index: int = 0) -> Optional[LLMInstance]:
        """
        Get an LLM instance for the specified provider.

        Args:
            provider: Provider name (e.g., "gemini", "groq")
            use_tools: Whether the LLM should support tools
            model_index: Index of the model to use (0 for first model)

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
            self._log_initialization(f"Provider '{provider_enum.value}' is not allowed by CMW_ALLOWED_PROVIDERS", "ERROR")
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
            instance = self._initialize_llm_instance(provider_enum, model_index)
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
                                from agent_ng.token_budget import _calculate_avg_tool_size, _GLOBAL_AVG_TOOL_SIZE
                                # Get bound tools as dicts from kwargs
                                kwargs = getattr(instance.llm, "kwargs", None)
                                if isinstance(kwargs, dict):
                                    bound_tools = kwargs.get("tools")
                                    if bound_tools and _GLOBAL_AVG_TOOL_SIZE is None:
                                        _GLOBAL_AVG_TOOL_SIZE = _calculate_avg_tool_size(bound_tools)
                            except Exception as exc:
                                # Non-critical: continue if average calculation fails
                                # Tools will still work, just with default 600 token estimate
                                logging.getLogger(__name__).debug(
                                    "Non-critical: failed to calculate tool averages: %s", exc
                                )
                            self._log_initialization(f"Tools bound to {provider} instance ({len(tools_list)} tools)", "INFO")
                        except Exception as e:
                            self._log_initialization(f"Failed to bind tools to {provider}: {e}", "WARNING")
                            # Don't fail the entire initialization if tool binding fails
                            instance.bound_tools = False

                self._instances[instance_key] = instance
                return instance

        return None

    def get_llm_with_tools(self, provider: str, model_index: int = 0) -> Optional[LLMInstance]:
        """
        Get an LLM instance with tools bound for the specified provider.

        Args:
            provider: Provider name (e.g., "gemini", "groq")
            model_index: Index of the model to use (0 for first model)

        Returns:
            LLMInstance or None if initialization failed
        """
        return self.get_llm(provider, use_tools=True, model_index=model_index)

    def create_new_llm_instance(self, provider: str, model_index: int = 0) -> Optional[LLMInstance]:
        """
        Create a NEW LLM instance for the specified provider (not cached).
        This is used for session isolation - each session gets its own instance.

        Args:
            provider: Provider name (e.g., "gemini", "groq")
            model_index: Index of the model to use (0 for first model)

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
            self._log_initialization(f"Provider '{provider_enum.value}' is not allowed by LLM_ALLOWED_PROVIDERS", "ERROR")
            return None

        # Create new instance without caching
        instance = self._initialize_llm_instance(provider_enum, model_index)
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
                            from agent_ng.token_budget import _calculate_avg_tool_size, _GLOBAL_AVG_TOOL_SIZE
                            kwargs = getattr(instance.llm, "kwargs", None)
                            if isinstance(kwargs, dict):
                                bound_tools = kwargs.get("tools")
                                if bound_tools and _GLOBAL_AVG_TOOL_SIZE is None:
                                    _GLOBAL_AVG_TOOL_SIZE = _calculate_avg_tool_size(bound_tools)
                        except Exception as exc:
                            # Non-critical: continue if average calculation fails
                            # Tools will still work, just with default 600 token estimate
                            logging.getLogger(__name__).debug(
                                "Non-critical: failed to calculate tool averages: %s", exc
                            )
                        self._log_initialization(f"Tools bound to NEW {provider} instance ({len(tools_list)} tools)", "INFO")
                    except Exception as e:
                        self._log_initialization(f"Failed to bind tools to NEW {provider}: {e}", "WARNING")
                        instance.bound_tools = False

            return instance
        return None

    def _find_model_index(self, provider: LLMProvider, model_name: str) -> Optional[int]:
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

    def _get_configured_provider_and_model_index(self) -> Tuple[Optional[LLMProvider], int]:
        """Get configured provider and model index from config/env"""
        import os
        try:
            from agent_ng.agent_config import get_llm_settings
            llm_settings = get_llm_settings()
            provider = llm_settings.get('default_provider', 'openrouter')
            default_model = llm_settings.get('default_model')
        except ImportError:
            provider = os.environ.get("AGENT_PROVIDER", "openrouter")
            default_model = os.environ.get("AGENT_DEFAULT_MODEL")

        try:
            provider_enum = LLMProvider(provider.lower())
        except ValueError:
            return None, 0

        model_index = 0
        if default_model and default_model.strip():
            found_index = self._find_model_index(provider_enum, default_model.strip())
            if found_index is not None:
                model_index = found_index

        return provider_enum, model_index

    def get_agent_llm(self) -> Optional[LLMInstance]:
        """Get the single LLM instance from AGENT_PROVIDER and AGENT_DEFAULT_MODEL"""
        provider_enum, model_index = self._get_configured_provider_and_model_index()
        if not provider_enum:
            return None

        instance = self.get_llm(provider_enum.value, model_index=model_index)
        if instance:
            self._log_initialization(f"✅ Using {provider_enum.value}/{instance.model_name}", "INFO")
        return instance

    def get_available_providers(self) -> List[str]:
        """Get list of available providers that can be initialized"""
        available = []
        for provider in LLMProvider:
            # Check if API key is available without initializing the LLM
            config = self.LLM_CONFIGS.get(provider)
            if config and self._get_api_key(config) and self._is_provider_allowed(provider):
                available.append(provider.value)
        # Sort provider identifiers alphabetically for stable UI ordering
        return sorted(available)

    def get_provider_config(self, provider: str) -> Optional[LLMConfig]:
        """Get configuration for a specific provider"""
        try:
            provider_enum = LLMProvider(provider.lower())
            return self.LLM_CONFIGS.get(provider_enum)
        except ValueError:
            return None

    def health_check(self) -> Dict[str, Any]:
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
                if instance.is_healthy and (current_time - instance.last_used) < 3600:  # 1 hour
                    healthy_count += 1

        return {
            "status": "completed",
            "healthy_instances": healthy_count,
            "total_instances": total_count,
            "timestamp": current_time
        }

    def get_initialization_logs(self) -> List[str]:
        """Get initialization logs"""
        return self._initialization_logs.copy()

    def clear_logs(self):
        """Clear initialization logs"""
        self._initialization_logs.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about managed LLM instances"""
        with self._lock:
            stats = {
                "total_instances": len(self._instances),
                "providers": {},
                "initialization_logs_count": len(self._initialization_logs)
            }

            for instance in self._instances.values():
                provider = instance.provider.value
                if provider not in stats["providers"]:
                    stats["providers"][provider] = {
                        "count": 0,
                        "healthy": 0,
                        "models": []
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

    def get_tools(self) -> List[Any]:
        """Get all available tools from tools module (avoiding duplicates) - cached"""
        # Return cached tools if available
        if hasattr(self, '_cached_tools'):
            return self._cached_tools

        tool_list = []
        tool_names = set()  # Track tool names to avoid duplicates

        # Load tools from main tools module (primary source)
        try:
            import tools.tools as tools_module
            self._load_tools_from_module(tools_module, tool_list, "tools.tools", tool_names)
        except ImportError:
            self._log_initialization("Could not import tools.tools module", "WARNING")

        # Load tools from attributes_tools submodule (only if not already loaded)
        try:
            import tools.attributes_tools as attributes_tools_module
            self._load_tools_from_module(attributes_tools_module, tool_list, "tools.attributes_tools", tool_names)
        except ImportError:
            self._log_initialization("Could not import tools.attributes_tools module", "WARNING")

        # Load tools from applications_tools submodule (only if not already loaded)
        try:
            import tools.applications_tools as applications_tools_module
            self._load_tools_from_module(applications_tools_module, tool_list, "tools.applications_tools", tool_names)
        except ImportError:
            self._log_initialization("Could not import tools.applications_tools module", "WARNING")

        # Load tools from templates_tools submodule (only if not already loaded)
        try:
            import tools.templates_tools as templates_tools_module
            self._load_tools_from_module(templates_tools_module, tool_list, "tools.templates_tools", tool_names)
        except ImportError:
            self._log_initialization("Could not import tools.templates_tools module", "WARNING")

        # Cache the tools list
        self._cached_tools = tool_list
        return tool_list

    def _load_tools_from_module(self, module, tool_list: List[Any], module_name: str, tool_names: set = None):
        """Load tools from a specific module (avoiding duplicates)"""
        if tool_names is None:
            tool_names = set()

        for name, obj in module.__dict__.items():
            # Only include actual tool objects (decorated with @tool)
            # Check if it's a proper @tool decorated function
            if (callable(obj) and
                not name.startswith("_") and
                not isinstance(obj, type) and  # Exclude classes
                hasattr(obj, '__module__') and  # Must have __module__ attribute
                (obj.__module__ == module_name or obj.__module__ == 'langchain_core.tools.structured') and  # Include both tools module and LangChain tools
                name not in ["CmwAgent", "CodeInterpreter", "submit_answer", "submit_intermediate_step", "web_search_deep_research_exa_ai"]):  # Exclude specific classes and internal tools

                # Check if it's a proper @tool decorated function
                # @tool decorated functions have specific attributes that indicate they're LangChain tools
                if (hasattr(obj, 'name') and
                    hasattr(obj, 'description') and
                    hasattr(obj, 'args_schema') and
                    hasattr(obj, 'func')):
                    # This is a proper @tool decorated function
                    tool_name = obj.name

                    # Skip if already loaded
                    if tool_name in tool_names:
                        self._log_initialization(f"Skipped duplicate tool: {tool_name} from {module_name}", "DEBUG")
                        continue

                    tool_list.append(obj)
                    tool_names.add(tool_name)
                    self._log_initialization(f"Loaded LangChain tool: {name} from {module_name}", "INFO")


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


