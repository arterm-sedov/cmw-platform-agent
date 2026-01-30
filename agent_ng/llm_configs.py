"""
LLM Configuration Module
========================

This module contains the default LLM provider configurations.
Configurations can be enriched at runtime with pricing data from OpenRouter.

The configurations are defined as a dictionary mapping LLMProvider enum values
to LLMConfig dataclass instances. This separation improves maintainability
and allows for easier enrichment with dynamic pricing data.
"""

from .llm_manager import LLMConfig, LLMProvider


def get_default_llm_configs() -> dict[LLMProvider, LLMConfig]:
    """
    Returns the default LLM configurations.

    This function can be extended to load configurations from JSON files
    or enrich them with pricing data from external APIs (e.g., OpenRouter).

    Returns:
        Dictionary mapping LLMProvider to LLMConfig instances.
    """
    return {
        LLMProvider.GEMINI: LLMConfig(
            name="Google Gemini",
            type_str="gemini",
            api_key_env="GEMINI_KEY",
            max_history=25,
            tool_support=True,
            force_tools=True,
            models=[
                {
                    "model": "gemini-2.5-flash",
                    "token_limit": 1048576,
                    "max_tokens": 65536,
                    "temperature": 0
                },
                {
                    "model": "gemini-2.5-pro",
                    "token_limit": 1048576,
                    "max_tokens": 65536,
                    "temperature": 0
                },
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
                # DeepSeek Models
                {
                    "model": "deepseek/deepseek-v3.1-terminus:exacto",
                    "token_limit": 131000,
                    "max_tokens": 32768,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "deepseek/deepseek-v3.1-terminus",
                    "token_limit": 131000,
                    "max_tokens": 65536,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "deepseek/deepseek-v3.2-speciale",
                    "token_limit": 163840,
                    "max_tokens": 65536,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "deepseek/deepseek-chat-v3.1:free",
                    "token_limit": 131000,
                    "max_tokens": 65536,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "deepseek/deepseek-r1-0528",
                    "token_limit": 131000,
                    "max_tokens": 65536,
                    "temperature": 0,
                    "force_tools": True
                },
                # Grok (xAI) Models
                {
                    "model": "x-ai/grok-4-fast:free",
                    "token_limit": 2000000,
                    "max_tokens": 8192,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "x-ai/grok-code-fast-1",
                    "token_limit": 256000,
                    "max_tokens": 10000,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "x-ai/grok-4-fast",
                    "token_limit": 2000000,
                    "max_tokens": 30000,
                    "temperature": 0,
                    "force_tools": True
                },
                # Qwen Models
                {
                    "model": "qwen/qwen3-coder:free",
                    "token_limit": 262144,
                    "max_tokens": 4096,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "qwen/qwen3-coder-flash",
                    "token_limit": 128000,
                    "max_tokens": 4096,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "qwen/qwen3-max",
                    "token_limit": 256000,
                    "max_tokens": 32768,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "qwen/qwen3-coder-plus",
                    "token_limit": 128000,
                    "max_tokens": 65536,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "qwen/qwen3-coder:exacto",
                    "token_limit": 262144,
                    "max_tokens": 65536,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "qwen/qwen-plus-2025-07-28",
                    "token_limit": 1000000,
                    "max_tokens": 32768,
                    "temperature": 0,
                    "force_tools": True
                },
                # MoonshotAI (Kimi) Models
                {
                    "model": "moonshotai/kimi-k2-0905:exacto",
                    "token_limit": 262144,
                    "max_tokens": 65536,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "moonshotai/kimi-k2-thinking",
                    "token_limit": 262144,
                    "max_tokens": 65536,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "moonshotai/kimi-k2.5",
                    "token_limit": 262144,
                    "max_tokens": 65536,
                    "temperature": 0,
                    "force_tools": True
                },
                # Z.AI (GLM) Models
                {
                    "model": "z-ai/glm-4.6:exacto",
                    "token_limit": 200000,
                    "max_tokens": 128000,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "z-ai/glm-4.7",
                    "token_limit": 200000,
                    "max_tokens": 65536,
                    "temperature": 0,
                    "force_tools": True
                },
                # Other Models
                {
                    "model": "google/gemini-3-flash-preview",
                    "token_limit": 1048576,
                    "max_tokens": 65536,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "google/gemini-3-pro-preview",
                    "token_limit": 1048576,
                    "max_tokens": 65536,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "anthropic/claude-sonnet-4.5",
                    "token_limit": 1000000,
                    "max_tokens": 64000,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "openai/gpt-oss-120b:exacto",
                    "token_limit": 131072,
                    "max_tokens": 32768,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "openai/gpt-5-mini",
                    "token_limit": 400000,
                    "max_tokens": 32768,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "nvidia/nemotron-nano-9b-v2:free",
                    "token_limit": 128000,
                    "max_tokens": 4096,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "mistralai/codestral-2508",
                    "token_limit": 256000,
                    "max_tokens": 4096,
                    "temperature": 0,
                    "force_tools": True
                },
                # MiniMax Models
                {
                    "model": "minimax/minimax-m2.1",
                    "token_limit": 196608,
                    "max_tokens": 65536,
                    "temperature": 0,
                    "force_tools": True
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
