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
except ImportError:
    try:
        from agent_ng.utils import ensure_valid_answer
    except ImportError:
        ensure_valid_answer = lambda x: str(x) if x is not None else "No answer provided"


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
    LLM_CONFIGS = {
        LLMProvider.GEMINI: LLMConfig(
            name="Google Gemini",
            type_str="gemini",
            api_key_env="GEMINI_KEY",
            max_history=25,
            tool_support=True,
            force_tools=True,
            models=[
                {
                    "model": "gemini-2.5-pro",
                    "token_limit": 2000000,
                    "max_tokens": 2000000,
                    "temperature": 0
                }
            ],
            enable_chunking=False
        ),
        LLMProvider.GROQ: LLMConfig(
            name="Groq",
            type_str="groq",
            api_key_env="GROQ_API_KEY",
            max_history=15,
            tool_support=True,
            force_tools=True,
            models=[
                {
                    "model": "groq/compound",
                    "token_limit": 131072,
                    "max_tokens": 8192,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "llama-3.3-70b-versatile",
                    "token_limit": 131072,
                    "max_tokens": 32768,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "llama-3.3-70b-8192",
                    "token_limit": 16000,
                    "max_tokens": 4096,
                    "temperature": 0,
                    "force_tools": True
                }
            ],
            enable_chunking=False
        ),
        LLMProvider.HUGGINGFACE: LLMConfig(
            name="HuggingFace",
            type_str="huggingface",
            api_key_env="HUGGINGFACE_API_KEY",
            max_history=20,
            tool_support=False,
            force_tools=False,
            models=[
                {
                    "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
                    "task": "text-generation",
                    "token_limit": 3000,
                    "max_new_tokens": 1024,
                    "do_sample": False,
                    "temperature": 0
                },
                {
                    "model": "microsoft/DialoGPT-medium",
                    "task": "text-generation",
                    "token_limit": 1000,
                    "max_new_tokens": 512,
                    "do_sample": False,
                    "temperature": 0
                },
                {
                    "model": "gpt2",
                    "task": "text-generation",
                    "token_limit": 1000,
                    "max_new_tokens": 256,
                    "do_sample": False,
                    "temperature": 0
                }
            ],
            enable_chunking=True
        ),
        LLMProvider.OPENROUTER: LLMConfig(
            name="OpenRouter",
            type_str="openrouter",
            api_key_env="OPENROUTER_API_KEY",
            api_base_env="OPENROUTER_BASE_URL",
            max_history=20,
            tool_support=True,
            force_tools=False,
            models=[
                {
                    "model": "openrouter/sonoma-dusk-alpha",
                    "token_limit": 2000000,
                    "max_tokens": 2048,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "qwen/qwen3-coder:free",
                    "token_limit": 262144,
                    "max_tokens": 2048,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "deepseek/deepseek-chat-v3.1:free",
                    "token_limit": 163840,
                    "max_tokens": 2048,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "mistralai/mistral-small-3.2-24b-instruct:free",
                    "token_limit": 131072,
                    "max_tokens": 2048,
                    "temperature": 0
                }
            ],
            enable_chunking=False
        ),
        LLMProvider.MISTRAL: LLMConfig(
            name="Mistral AI",
            type_str="mistral",
            api_key_env="MISTRAL_API_KEY",
            max_history=20,
            tool_support=True,
            force_tools=True,
            models=[
                {
                    "model": "mistral-large-latest",
                    "token_limit": 32000,
                    "max_tokens": 2048,
                    "temperature": 0
                },
                {
                    "model": "mistral-small-latest",
                    "token_limit": 32000,
                    "max_tokens": 2048,
                    "temperature": 0
                },
                {
                    "model": "mistral-medium-latest",
                    "token_limit": 32000,
                    "max_tokens": 2048,
                    "temperature": 0
                }
            ],
            token_per_minute_limit=500000,
            enable_chunking=False
        ),
        LLMProvider.GIGACHAT: LLMConfig(
            name="Sber GigaChat",
            type_str="gigachat",
            api_key_env="GIGACHAT_API_KEY",
            scope_env="GIGACHAT_SCOPE",
            verify_ssl_env="GIGACHAT_VERIFY_SSL",
            max_history=20,
            tool_support=True,
            force_tools=True,
            models=[
                {
                    "model": "GigaChat-2",
                    "token_limit": 128000,
                    "max_tokens": 2048,
                    "temperature": 0,
                    "top_p": 0.9,
                    "repetition_penalty": 1.0
                },
                {
                    "model": "GigaChat-2-Pro",
                    "token_limit": 128000,
                    "max_tokens": 2048,
                    "temperature": 0,
                    "top_p": 0.9,
                    "repetition_penalty": 1.0
                },
                {
                    "model": "GigaChat-2-Max",
                    "token_limit": 128000,
                    "max_tokens": 2048,
                    "temperature": 0,
                    "top_p": 0.9,
                    "repetition_penalty": 1.0
                }
            ],
            enable_chunking=False
        ),
    }
    
    # Single provider from environment variable
    # No more sequence - use AGENT_PROVIDER from dotenv
    
    def __init__(self):
        """Initialize the LLM Manager"""
        self._instances: Dict[str, LLMInstance] = {}
        self._lock = threading.Lock()
        self._initialization_logs = []
        self._health_check_interval = 300  # 5 minutes
        self._last_health_check = 0
        
    def _log_initialization(self, message: str, level: str = "INFO"):
        """Log initialization messages"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self._initialization_logs.append(log_entry)
        print(log_entry)  # Also print to console for real-time feedback
        
    def _get_api_key(self, config: LLMConfig) -> Optional[str]:
        """Get API key from environment variables"""
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            self._log_initialization(f"API key not found for {config.name} ({config.api_key_env})", "WARNING")
        return api_key
        
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
        
    def get_agent_llm(self) -> Optional[LLMInstance]:
        """
        Get the single LLM instance from AGENT_PROVIDER environment variable.
        
        Returns:
            Single LLM instance or None if not available
        """
        import os
        
        # Get provider from environment variable
        agent_provider = os.environ.get("AGENT_PROVIDER", "mistral")
        
        # Get the single instance
        instance = self.get_llm(agent_provider)
        if instance:
            return instance
        else:
            print(f"Error: AGENT_PROVIDER '{agent_provider}' not available")
            return None
        
    def get_available_providers(self) -> List[str]:
        """Get list of available providers that can be initialized"""
        available = []
        for provider in LLMProvider:
            instance = self.get_llm(provider.value)
            if instance:
                available.append(provider.value)
        return available
        
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
        import os
        
        # Get the current provider from environment
        agent_provider = os.environ.get("AGENT_PROVIDER", "mistral")
        
        try:
            provider_enum = LLMProvider(agent_provider.lower())
            config = self.LLM_CONFIGS.get(provider_enum)
            
            if config and config.models:
                # Get the first model's token limit as the context window
                return config.models[0].get("token_limit", 0)
        except ValueError:
            pass
        
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


def reset_llm_manager():
    """Reset the global LLM manager (useful for testing)"""
    global _llm_manager
    with _manager_lock:
        _llm_manager = None
