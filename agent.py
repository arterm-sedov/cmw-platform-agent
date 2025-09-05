"""
CMW Platform Agent
By Arte(r)m Sedov
==================================

This module implements the main agent logic for the CMW Platform Agent.

Usage:
    agent = GaiaAgent(provider="google")
    answer = agent(question)

Environment Variables:
    - GEMINI_KEY: API key for Gemini model (if using Google provider)
    - SUPABASE_URL: URL for Supabase instance
    - SUPABASE_KEY: Key for Supabase access

Files required in the same directory:
    - system_prompt.json
"""
# Standard library imports
import base64
import builtins
import csv
import datetime
import io
import json
import os
import random
import re
import sys
import tempfile
import time
from io import StringIO
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
import tiktoken

# LangChain imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI

# Local imports
import tools
from dataset_manager import dataset_manager
from tools import *
from utils import TRACES_DIR, ensure_valid_answer
from vector_store import (
    get_embeddings,
    get_reference_answer,
    get_retriever_tool,
    get_vector_store,
    vector_answers_match,
    vector_store_manager,
)

def trace_prints_with_context(context_type: str):
    """
    Decorator that traces all print calls in a function and attaches them to specific execution contexts.
    Automatically captures print output and adds it to the appropriate context in the agent's trace.
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Store original print
            original_print = print
            
            # Store current context
            old_context = getattr(self, '_current_trace_context', None)
            self._current_trace_context = context_type
            
            def trace_print(*print_args, **print_kwargs):
                # Original print functionality
                original_print(*print_args, **print_kwargs)
                
                # Write to current LLM's stdout buffer if available
                if hasattr(self, 'current_llm_stdout_buffer') and self.current_llm_stdout_buffer:
                    try:
                        message = " ".join(str(arg) for arg in print_args)
                        self.current_llm_stdout_buffer.write(message + "\n")
                    except Exception as e:
                        # Fallback if buffer write fails
                        original_print(f"[Buffer Error] Failed to write to stdout buffer: {e}")
                
                # Add to appropriate context
                if hasattr(self, 'question_trace') and self.question_trace is not None:
                    try:
                        self._add_log_to_context(" ".join(str(arg) for arg in print_args), func.__name__)
                    except Exception as e:
                        # Fallback to basic logging if trace fails
                        original_print(f"[Trace Error] Failed to add log entry: {e}")
            
            # Override print for this function call
            builtins.print = trace_print
            
            try:
                result = func(self, *args, **kwargs)
            finally:
                # Restore original print
                builtins.print = original_print
                # Restore previous context
                self._current_trace_context = old_context
            
            return result
        return wrapper
    return decorator

def trace_prints(func):
    """
    Decorator that traces all print calls in a function.
    Automatically captures print output and adds it to the agent's trace.
    """
    def wrapper(self, *args, **kwargs):
        # Store original print
        original_print = print
        
        def trace_print(*print_args, **print_kwargs):
            # Original print functionality
            original_print(*print_args, **print_kwargs)
            
            # Write to current LLM's stdout buffer if available
            if hasattr(self, 'current_llm_stdout_buffer') and self.current_llm_stdout_buffer:
                try:
                    message = " ".join(str(arg) for arg in print_args)
                    self.current_llm_stdout_buffer.write(message + "\n")
                except Exception as e:
                    # Fallback if buffer write fails
                    original_print(f"[Buffer Error] Failed to write to stdout buffer: {e}")
            
            # Add to trace
            if hasattr(self, 'question_trace') and self.question_trace is not None:
                try:
                    log_entry = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "level": "info",
                        "message": " ".join(str(arg) for arg in print_args),
                        "function": func.__name__
                    }
                    self.question_trace.setdefault("logs", []).append(log_entry)
                except Exception as e:
                    # Fallback to basic logging if trace fails
                    original_print(f"[Trace Error] Failed to add log entry: {e}")
        
        # Override print for this function call
        builtins.print = trace_print
        
        try:
            result = func(self, *args, **kwargs)
        finally:
            # Restore original print
            builtins.print = original_print
        
        return result
    return wrapper

class Tee:
    """
    Tee class to duplicate writes to multiple streams (e.g., sys.stdout and a buffer).
    """
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

class GaiaAgent:
    """
    Main agent for the CMW Platform benchmark.

    This agent:
      - Uses the tools.py (math, code, file, image, web, etc.)
      - Integrates a supabase retriever for similar Q/A and context
      - Strictly follows the system prompt in system_prompt
      - Is modular and extensible for future tool/model additions
      - Includes rate limiting and retry logic for API calls
      - Uses Google Gemini for first attempt, Groq for retry
      - Implements LLM-specific token management (no limits for Gemini, conservative for others)

    Args:
        provider (str): LLM provider to use. One of "google", "groq", or "huggingface".

    Attributes:
        system_prompt (str): The loaded system prompt template.
        sys_msg (SystemMessage): The system message for the LLM.
        vector_store_manager: Vector store manager instance for similarity operations.
        llm_primary: Primary LLM instance (Google Gemini).
        llm_fallback: Fallback LLM instance (Groq).
        llm_third_fallback: Third fallback LLM instance (HuggingFace).
        tools: List of callable tool functions.
        llm_primary_with_tools: Primary LLM instance with tools bound for tool-calling.
        llm_fallback_with_tools: Fallback LLM instance with tools bound for tool-calling.
        llm_third_fallback_with_tools: Third fallback LLM instance with tools bound for tool-calling.
        last_request_time (float): Timestamp of the last API request for rate limiting.
        min_request_interval (float): Minimum time between requests in seconds.
        token_limits: Dictionary of token limits for different LLMs
        max_message_history: Maximum number of messages to keep in history
        original_question: Store the original question for reuse
        similarity_threshold: Minimum similarity score (0.0-1.0) to consider answers similar
        tool_calls_similarity_threshold: Silarity for tool deduplication
        max_summary_tokens: Global token limit for summaries
    """
    
    # Single source of truth for LLM configuration
    LLM_CONFIG = {
        "default": {
            "type_str": "default",
            "token_limit": 2500,
            "max_history": 20,
            "tool_support": False,
            "force_tools": False,
            "models": [],
            "token_per_minute_limit": None,
            "enable_chunking": True
        },
        "gemini": {
            "name": "Google Gemini",
            "type_str": "gemini",
            "api_key_env": "GEMINI_KEY",
            "max_history": 25,
            "tool_support": True,
            "force_tools": True,
            "models": [
                {
                    "model": "gemini-2.5-pro",
                    "token_limit": 2000000,
                    "max_tokens": 2000000,
                    "temperature": 0
                }
            ],
            "token_per_minute_limit": None,
            "enable_chunking": False
        },
        "groq": {
            "name": "Groq",
            "type_str": "groq",
            "api_key_env": "GROQ_API_KEY",
            "max_history": 15,
            "tool_support": True,
            "force_tools": True,
            "models": [
                {
                    "model": "qwen-qwq-32b",
                    "token_limit": 16000,
                    "max_tokens": 2048,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "llama-3.1-8b-instant",
                    "token_limit": 16000,
                    "max_tokens": 2048,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "llama-3.3-70b-8192",
                    "token_limit": 16000,
                    "max_tokens": 2048,
                    "temperature": 0,
                    "force_tools": True
                }
            ],
            "token_per_minute_limit": None,  # Model-specific limits used instead
            "enable_chunking": False
        },
        "huggingface": {
            "name": "HuggingFace",
            "type_str": "huggingface",
            "api_key_env": "HUGGINGFACEHUB_API_TOKEN",
            "max_history": 20,
            "tool_support": False,
            "force_tools": False,
            "models": [
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
            "token_per_minute_limit": None,
            "enable_chunking": True
        },
        "openrouter": {
            "name": "OpenRouter",
            "type_str": "openrouter",
            "api_key_env": "OPENROUTER_API_KEY",
            "api_base_env": "OPENROUTER_BASE_URL",
            "max_history": 20,
            "tool_support": True,
            "force_tools": False,
            "models": [
                {
                    "model": "deepseek/deepseek-chat-v3.1:free",
                    "token_limit": 100000,
                    "max_tokens": 2048,
                    "temperature": 0,
                    "force_tools": True
                },
                {
                    "model": "mistralai/mistral-small-3.2-24b-instruct:free",
                    "token_limit": 90000,
                    "max_tokens": 2048,
                    "temperature": 0
                },
                {
                    "model": "openrouter/cypher-alpha:free",
                    "token_limit": 1000000,
                    "max_tokens": 2048,
                    "temperature": 0
                }
            ],
            "token_per_minute_limit": None,
            "enable_chunking": False
        },
        "mistral": {
            "name": "Mistral AI",
            "type_str": "mistral",
            "api_key_env": "MISTRAL_API_KEY",
            "max_history": 20,
            "tool_support": True,
            "force_tools": True,  # Keep tools enabled by default, disable only on consistent failures
            "models": [
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
            "token_per_minute_limit": 500000,
            "enable_chunking": False
        },
        "gigachat": {
            "name": "Sber GigaChat",
            "type_str": "gigachat",
            "api_key_env": "GIGACHAT_API_KEY",
            "scope_env": "GIGACHAT_SCOPE",
            "verify_ssl_env": "GIGACHAT_VERIFY_SSL",
            "max_history": 20,
            "tool_support": True,
            "force_tools": True,
            "models": [
                {
                    "model": "GigaChat-2",
                    "token_limit": 128_000,
                    "max_tokens": 2048,
                    "temperature": 0
                },
                {
                    "model": "GigaChat-2-Pro",
                    "token_limit": 128_000,
                    "max_tokens": 2048,
                    "temperature": 0
                },
                {
                    "model": "GigaChat-2-Max",
                    "token_limit": 128_000,
                    "max_tokens": 2048,
                    "temperature": 0
                }
            ],
            "token_per_minute_limit": None,
            "enable_chunking": False
        },
    }
    
    # Default LLM sequence order - references LLM_CONFIG keys
    DEFAULT_LLM_SEQUENCE = [
        #"openrouter",
        "gigachat",
        # "mistral",
        # "gemini",
        # "groq",
        # "huggingface"
    ]
    # Print truncation length for debug output
    MAX_PRINT_LEN = 1000
    
    def __init__(self, provider: str = "groq"):
        """
        Initialize the agent, loading the system prompt, tools, retriever, and LLM.

        Args:
            provider (str): LLM provider to use. One of "google", "groq", or "huggingface".

        Raises:
            ValueError: If an invalid provider is specified.
        """
        # --- Capture stdout for debug output and tee to console ---
        debug_buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = Tee(old_stdout, debug_buffer)
        try:
            # Store the config of the successfully initialized model per provider
            self.active_model_config = {} 
            self.system_prompt = self._load_system_prompt()
            self.sys_msg = SystemMessage(content=self.system_prompt)
            self.original_question = None
            # Global threshold. Minimum similarity score (0.0-1.0) to consider answers similar
            self.similarity_threshold = 0.95
            # Tool calls deduplication threshold
            self.tool_calls_similarity_threshold = 0.90
            # Global token limit for summaries
            # self.max_summary_tokens = 255
            self.last_request_time = 0
            # Track the current LLM type and model for rate limiting
            self.current_llm_type = None
            self.current_model_name = None
            self.token_limits = {}
            for provider_key, config in self.LLM_CONFIG.items():
                models = config.get("models", [])
                if models:
                    self.token_limits[provider_key] = [model.get("token_limit", self.LLM_CONFIG["default"]["token_limit"]) for model in models]
                else:
                    self.token_limits[provider_key] = [self.LLM_CONFIG["default"]["token_limit"]]
            
            # Initialize token usage tracking for rate limiting
            self._provider_token_usage = {}
            # Unified LLM tracking system
            self.llm_tracking = {}
            for llm_type in self.DEFAULT_LLM_SEQUENCE:
                self.llm_tracking[llm_type] = {
                    "successes": 0,
                    "failures": 0,
                    "threshold_passes": 0,
                    "submitted": 0,      # Above threshold, submitted
                    "low_submit": 0,        # Below threshold, submitted
                    "total_attempts": 0
                }
            self.total_questions = 0
            
            # Initialize tracing system
            self.question_trace = None
            self.current_llm_call_id = None

            # Vector store functionality is now handled by vector_store.py module
            # All Supabase operations are disabled by default
            self.vector_store_manager = vector_store_manager

            # Arrays for all initialized LLMs and tool-bound LLMs, in order (initialize before LLM setup loop)
            self.llms = []
            self.llms_with_tools = []
            self.llm_provider_names = []
            # Track initialization results for summary
            self.llm_init_results = []
            # Get the LLM types that should be initialized based on the sequence
            llm_types_to_init = self.DEFAULT_LLM_SEQUENCE
            llm_names = [self.LLM_CONFIG[llm_type]["name"] for llm_type in llm_types_to_init]
            print(f"🔄 Initializing LLMs based on sequence:")
            for i, name in enumerate(llm_names, 1):
                print(f"   {i}. {name}")
            # Prepare storage for LLM instances
            self.llm_instances = {}
            self.llm_instances_with_tools = {}
            # Only gather tools if at least one LLM supports tools
            any_tool_support = any(self.LLM_CONFIG[llm_type].get("tool_support", False) for llm_type in llm_types_to_init)
            self.tools = self._gather_tools() if any_tool_support else []
            for idx, llm_type in enumerate(llm_types_to_init):
                config = self.LLM_CONFIG[llm_type]
                llm_name = config["name"]
                for model_config in config["models"]:
                    model_id = model_config.get("model", "")
                    print(f"🔄 Initializing LLM {llm_name} (model: {model_id}) ({idx+1} of {len(llm_types_to_init)})")
                    llm_instance = None
                    model_config_used = None
                    plain_ok = False
                    tools_ok = None
                    error_plain = None
                    error_tools = None
                    try:
                        def get_llm_instance(llm_type, config, model_config):
                            if llm_type == "gemini":
                                return self._init_gemini_llm(config, model_config)
                            elif llm_type == "groq":
                                return self._init_groq_llm(config, model_config)
                            elif llm_type == "huggingface":
                                return self._init_huggingface_llm(config, model_config)
                            elif llm_type == "openrouter":
                                return self._init_openrouter_llm(config, model_config)
                            elif llm_type == "mistral":
                                return self._init_mistral_llm(config, model_config)
                            elif llm_type == "gigachat":
                                return self._init_gigachat_llm(config, model_config)
                            else:
                                return None
                        llm_instance = get_llm_instance(llm_type, config, model_config)
                        if llm_instance is not None:
                            try:
                                plain_ok = self._ping_llm(f"{llm_name} (model: {model_id})", llm_type, use_tools=False, llm_instance=llm_instance)
                            except Exception as e:
                                plain_ok, error_plain = self._handle_llm_error(e, llm_name, llm_type, phase="init", context="plain")
                                if not plain_ok:
                                    # Do not add to available LLMs, break out
                                    break
                        else:
                            error_plain = "instantiation returned None"
                        if config.get("tool_support", False) and self.tools and llm_instance is not None and plain_ok:
                            try:
                                # Filter tools for provider-specific schema limitations (e.g., GigaChat JSON Schema)
                                safe_tools = self._filter_tools_for_llm(self.tools, llm_type)
                                llm_with_tools = llm_instance.bind_tools(safe_tools)
                                try:
                                    tools_ok = self._ping_llm(f"{llm_name} (model: {model_id}) (with tools)", llm_type, use_tools=True, llm_instance=llm_with_tools)
                                except Exception as e:
                                    tools_ok, error_tools = self._handle_llm_error(e, llm_name, llm_type, phase="init", context="tools")
                                    if not tools_ok:
                                        break
                            except Exception as e:
                                tools_ok = False
                                error_tools = str(e)
                        else:
                            tools_ok = None
                        # Store result for summary
                        self.llm_init_results.append({
                            "provider": llm_name,
                            "llm_type": llm_type,
                            "model": model_id,
                            "plain_ok": plain_ok,
                            "tools_ok": tools_ok,
                            "error_plain": error_plain,
                            "error_tools": error_tools
                        })
                        # Special handling for models with force_tools: always bind tools if tool support is enabled, regardless of tools_ok
                        # Check force_tools at both provider and model level
                        force_tools = config.get("force_tools", False) or model_config.get("force_tools", False)
                        if llm_instance and plain_ok and (
                            not config.get("tool_support", False) or tools_ok or (force_tools and config.get("tool_support", False))
                        ):
                            self.active_model_config[llm_type] = model_config
                            self.llm_instances[llm_type] = llm_instance
                            if config.get("tool_support", False):
                                # Bind filtered tool set for this provider
                                safe_tools = self._filter_tools_for_llm(self.tools, llm_type)
                                self.llm_instances_with_tools[llm_type] = llm_instance.bind_tools(safe_tools)
                                if force_tools and not tools_ok:
                                    print(f"⚠️ {llm_name} (model: {model_id}) (with tools) test returned empty or failed, but binding tools anyway (force_tools=True: tool-calling is known to work in real use).")
                            else:
                                self.llm_instances_with_tools[llm_type] = None
                            self.llms.append(llm_instance)
                            self.llms_with_tools.append(self.llm_instances_with_tools[llm_type])
                            self.llm_provider_names.append(llm_type)
                            print(f"✅ LLM ({llm_name}) initialized successfully with model {model_id}")
                            break
                        else:
                            self.llm_instances[llm_type] = None
                            self.llm_instances_with_tools[llm_type] = None
                            print(f"⚠️ {llm_name} (model: {model_id}) failed initialization (plain_ok={plain_ok}, tools_ok={tools_ok})")
                    except Exception as e:
                        print(f"⚠️ Failed to initialize {llm_name} (model: {model_id}): {e}")
                        self.llm_init_results.append({
                            "provider": llm_name,
                            "llm_type": llm_type,
                            "model": model_id,
                            "plain_ok": False,
                            "tools_ok": False,
                            "error_plain": str(e),
                            "error_tools": str(e)
                        })
                        self.llm_instances[llm_type] = None
                        self.llm_instances_with_tools[llm_type] = None
            # Legacy assignments for backward compatibility
            self.tools = self._gather_tools()
            # Print summary table after all initializations
            self._print_llm_init_summary()
        finally:
            sys.stdout = old_stdout
        debug_output = debug_buffer.getvalue()
        # --- Save LLM initialization summary to log file and commit to repo ---  
        try:
            # Create structured init data
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_table = self._format_llm_init_summary(as_str=True)
            summary_json = self._get_llm_init_summary_json()
            
            init_data = {
                "timestamp": timestamp,
                "init_summary": summary_table,
                "init_summary_json": json.dumps(summary_json, ensure_ascii=False) if not isinstance(summary_json, str) else summary_json,
                "debug_output": debug_output,
                "llm_config": json.dumps(self.LLM_CONFIG, ensure_ascii=False) if not isinstance(self.LLM_CONFIG, str) else self.LLM_CONFIG,
                "available_models": json.dumps(self._get_available_models(), ensure_ascii=False) if not isinstance(self._get_available_models(), str) else self._get_available_models(),
                "tool_support": self._get_tool_support_status()
            }
            
            # Upload to dataset
            success = dataset_manager.upload_init_summary(init_data)
            if success:
                print(f"✅ LLM initialization summary uploaded to dataset")
            else:
                print(f"⚠️ Failed to upload LLM initialization summary to dataset")
                
        except Exception as e:
            print(f"⚠️ Failed to upload LLM initialization summary: {e}")

    def _load_system_prompt(self):
        """
        Load the system prompt from the system_prompt.json file as a JSON string.
        """
        try:
            with open("system_prompt.json", "r", encoding="utf-8") as f:
                taxonomy = json.load(f)
                return json.dumps(taxonomy, ensure_ascii=False)
        except FileNotFoundError:
            print("⚠️ system_prompt.json not found, using default system prompt")
        except Exception as e:
            print(f"⚠️ Error reading system_prompt.json: {e}")
        return "You are a helpful assistant. Please provide clear and accurate responses."
    
    def _handle_rate_limit_throttling(self, error, llm_name, llm_type, max_retries=3):
        """
        Handle rate limit errors by throttling and retrying instead of immediate fallback.
        
        Args:
            error: The rate limit error
            llm_name: Name of the LLM
            llm_type: Type of the LLM
            max_retries: Maximum number of retry attempts
            
        Returns:
            bool: True if retry should be attempted, False if max retries exceeded
        """
        # Extract retry-after from error if available
        retry_after = None
        error_str = str(error)
        
        # Look for retry-after in error message or headers
        if "retry-after" in error_str.lower():
            match = re.search(r'retry-after[:\s]*(\d+)', error_str, re.IGNORECASE)
            if match:
                retry_after = int(match.group(1))
        
        # Default retry delays for different error types
        if "service_tier_capacity_exceeded" in error_str or "3505" in error_str:
            # Mistral capacity exceeded - wait longer
            retry_after = retry_after or 30
        elif "invalid_request_message_order" in error_str or "3230" in error_str:
            # Mistral message ordering error - short wait and retry
            retry_after = retry_after or 5
        elif "429" in error_str or "rate limit" in error_str.lower():
            # Generic rate limit - moderate wait
            retry_after = retry_after or 10
        else:
            # Unknown rate limit - short wait
            retry_after = retry_after or 5
            
        # Check if we've exceeded max retries
        retry_key = f"{llm_type}_{llm_name}"
        if not hasattr(self, '_rate_limit_retry_count'):
            self._rate_limit_retry_count = {}
            
        current_retries = self._rate_limit_retry_count.get(retry_key, 0)
        if current_retries >= max_retries:
            print(f"⏰ Max retries ({max_retries}) exceeded for {llm_name} rate limiting. Falling back to next LLM.")
            self._rate_limit_retry_count[retry_key] = 0  # Reset for future use
            return False
            
        # Increment retry count
        self._rate_limit_retry_count[retry_key] = current_retries + 1
        
        print(f"⏰ Rate limit hit for {llm_name}. Waiting {retry_after}s before retry ({current_retries + 1}/{max_retries})...")
        time.sleep(retry_after)
        
        return True

    def _rate_limit(self):
        """
        Implement rate limiting to avoid hitting API limits.
        Waits if necessary to maintain minimum interval between requests.
        For providers with a token_per_minute_limit, throttle based on tokens sent in the last 60 seconds.
        """
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        # Determine wait time based on current LLM type
        min_interval = 20
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        llm_type = self.current_llm_type
        config = self.LLM_CONFIG.get(llm_type, {})
        
        # Check for model-specific rate limit first, then provider-level limit
        tpm_limit = None
        if hasattr(self, 'current_model_name') and self.current_model_name:
            # Look for model-specific rate limit
            models = config.get("models", [])
            for model_config in models:
                if model_config.get("model") == self.current_model_name:
                    tpm_limit = model_config.get("token_per_minute_limit")
                    break
        
        # Fall back to provider-level rate limit if no model-specific limit found
        if tpm_limit is None:
            tpm_limit = config.get("token_per_minute_limit")
            
        if tpm_limit:
            # Initialize token usage tracker for this provider
            if llm_type not in self._provider_token_usage:
                self._provider_token_usage[llm_type] = []  # List of (timestamp, tokens)
            # Remove entries older than 60 seconds
            self._provider_token_usage[llm_type] = [
                (ts, tok) for ts, tok in self._provider_token_usage[llm_type]
                if current_time - ts < 60
            ]
            # Estimate tokens for the next request (should be set before _rate_limit is called)
            next_tokens = getattr(self, '_next_request_tokens', None)
            if next_tokens is None:
                next_tokens = 0
            # Calculate total tokens in the last 60 seconds
            tokens_last_minute = sum(tok for ts, tok in self._provider_token_usage[llm_type])
            # If sending now would exceed the TPM limit, wait
            if tokens_last_minute + next_tokens > tpm_limit:
                # Calculate how long to wait: find the soonest token batch to expire
                oldest_ts = min(ts for ts, tok in self._provider_token_usage[llm_type]) if self._provider_token_usage[llm_type] else current_time
                wait_time = 60 - (current_time - oldest_ts) + 60  # Add 1 min safety
                print(f"⏳ [TPM Throttle] Waiting {wait_time:.1f}s to respect {tpm_limit} TPM for {llm_type}...")
                time.sleep(wait_time)
            # After waiting, add this request to the tracker
            self._provider_token_usage[llm_type].append((time.time(), next_tokens))
        self.last_request_time = time.time()

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using tiktoken for accurate counting.
        """
        try:
            # Use GPT-4 encoding as a reasonable approximation for most models
            encoding = tiktoken.encoding_for_model("gpt-4")
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            # Fallback to character-based estimation if tiktoken fails
            print(f"⚠️ Tiktoken failed, using fallback: {e}")
            return len(text) // 4

    def _truncate_messages(self, messages: List[Any], llm_type: str = None) -> List[Any]:
        """
        Truncate message history to prevent token overflow.
        Keeps system message and a recent chronological window of messages.
        Ensures conversation order is preserved and that any 'tool' message
        retains its preceding assistant tool-call message to satisfy providers
        that require function-call continuity (e.g., GigaChat/OpenAI-style).
        
        Args:
            messages: List of messages to truncate
            llm_type: Type of LLM for context-aware truncation
        """
        # Always read max_history from LLM_CONFIG, using global default when provider has none
        max_history = self.LLM_CONFIG.get(llm_type, {}).get(
            "max_history",
            self.LLM_CONFIG["default"]["max_history"]
        )

        if len(messages) <= max_history:
            return messages

        # Identify system message (first message may be system)
        system_msg = messages[0] if messages and hasattr(messages[0], 'type') and messages[0].type == 'system' else None

        # Start with a simple tail window preserving chronological order
        # Keep the last (max_history - 1) messages after the system message (if any)
        keep_after_system = max_history - 1 if system_msg else max_history
        tail_messages = messages[1:] if system_msg else messages
        kept = tail_messages[-keep_after_system:]

        # If the first kept message is a tool message, ensure we also include the
        # immediately preceding assistant message that issued the tool call
        def _has_tool_calls(ai_msg: Any) -> bool:
            try:
                return (hasattr(ai_msg, 'type') and ai_msg.type in ('ai', 'assistant')) and bool(getattr(ai_msg, 'tool_calls', None))
            except Exception:
                return False

        if kept and hasattr(kept[0], 'type') and kept[0].type == 'tool':
            # Find the original index of this tool message
            first_tool_idx_global = (1 if system_msg else 0) + (len(tail_messages) - len(kept))
            # Look one message back in the full list for the assistant tool-call
            prev_idx = first_tool_idx_global - 1
            if prev_idx >= 0:
                prev_msg = messages[prev_idx]
                if _has_tool_calls(prev_msg):
                    # Prepend the assistant tool-call message to kept
                    kept = [prev_msg] + kept
                else:
                    # Walk further back until we either find an assistant with tool_calls or hit a non-ai
                    scan_idx = prev_idx
                    while scan_idx >= 0:
                        candidate = messages[scan_idx]
                        if _has_tool_calls(candidate):
                            kept = [candidate] + kept
                            break
                        # Stop if we encounter a human/system before an AI tool-call
                        if hasattr(candidate, 'type') and candidate.type in ('human', 'system'):
                            break
                        scan_idx -= 1

        # If we exceeded the allowed window due to pairing, trim from the start but never drop the paired assistant
        # Keep chronological order: system (optional) + kept
        truncated_messages = []
        if system_msg:
            truncated_messages.append(system_msg)
        # If still too long, drop earliest items after system, but do not start with a dangling tool message
        if len(truncated_messages) + len(kept) > max_history:
            drop = (len(truncated_messages) + len(kept)) - max_history
            # Avoid starting with a tool message after drop by ensuring we don't drop the paired assistant
            start_idx = 0
            while drop > 0 and start_idx < len(kept):
                # Do not drop if dropping would make the new first item a tool without preceding assistant
                next_first = kept[start_idx + 1] if start_idx + 1 < len(kept) else None
                if next_first is not None and hasattr(next_first, 'type') and next_first.type == 'tool':
                    break
                start_idx += 1
                drop -= 1
            kept = kept[start_idx:]

        truncated_messages.extend(kept)
        return truncated_messages

    @trace_prints_with_context("tool_execution")
    def _execute_tool(self, tool_name: str, tool_args: dict, tool_registry: dict, call_id: str = None) -> str:
        """
        Execute a tool with the given name and arguments.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            tool_registry: Registry of available tools
            
        Returns:
            str: Result of tool execution
        """
        # Inject file data if available and needed
        if isinstance(tool_args, dict):
            tool_args = self._inject_file_data_to_tool_args(tool_name, tool_args)
        
        # Create truncated copy for logging only
        truncated_args = self._deep_trim_dict_max_length(tool_args)
        print(f"[Tool Loop] Running tool: {tool_name} with args: {truncated_args}")
        print(f"[Tool Loop] Tool args type: {type(tool_args)}, Tool args value: {tool_args}")
        
        # Start timing for trace
        start_time = time.time()
        
        tool_func = tool_registry.get(tool_name)
        
        if not tool_func:
            tool_result = f"Tool '{tool_name}' not found."
            print(f"[Tool Loop] Tool '{tool_name}' not found.")
        else:
            try:
                # Check if it's a proper LangChain tool (has invoke method and tool attributes)
                if (hasattr(tool_func, 'invoke') and 
                    hasattr(tool_func, 'name') and 
                    hasattr(tool_func, 'description')):
                    # This is a proper LangChain tool, use invoke method
                    if isinstance(tool_args, dict):
                        tool_result = tool_func.invoke(tool_args)
                    else:
                        # For non-dict args, assume it's a single value that should be passed as 'input'
                        tool_result = tool_func.invoke({'input': tool_args})
                else:
                    # This is a regular function, call it directly
                    if isinstance(tool_args, dict):
                        tool_result = tool_func(**tool_args)
                    else:
                        # For non-dict args, pass directly
                        tool_result = tool_func(tool_args)
                print(f"[Tool Loop] Tool '{tool_name}' executed successfully.")
                # Only trim for printing, not for LLM
                self._print_tool_result(tool_name, tool_result)
            except Exception as e:
                tool_result = f"Error running tool '{tool_name}': {e}"
                print(f"[Tool Loop] Error running tool '{tool_name}': {e}")
        
        # Add tool execution to trace if call_id is provided
        if call_id and self.question_trace:
            execution_time = time.time() - start_time
            llm_type = self.current_llm_type
            self._add_tool_execution_trace(llm_type, call_id, tool_name, tool_args, tool_result, execution_time)
        
        return str(tool_result)

    def _has_tool_messages(self, messages: List) -> bool:
        """
        Check if the message history contains ToolMessage objects.
        
        Args:
            messages: List of message objects
            
        Returns:
            bool: True if ToolMessage objects are present, False otherwise
        """
        return any(
            hasattr(msg, 'type') and msg.type == 'tool' and hasattr(msg, 'content') 
            for msg in messages
        )

    @trace_prints_with_context("final_answer")
    def _force_final_answer(self, messages, tool_results_history, llm):
        """
        Handle duplicate tool calls by forcing final answer using LangChain's native mechanisms.
        For Gemini, always include tool results in the reminder. For others, only if not already present.
        Args:
            messages: Current message list
            tool_results_history: History of tool results (can be empty)
            llm: LLM instance
        Returns:
            Response from LLM or direct FINAL ANSWER from tool results
        """
        # 1. Scan tool results for FINAL ANSWER using _has_final_answer_marker
        for result in reversed(tool_results_history):  # Prefer latest
            if self._has_final_answer_marker(result):
                # Extract the final answer text using _extract_final_answer
                answer = self._extract_final_answer(result)
                if answer:
                    ai_msg = AIMessage(content=f"FINAL ANSWER: {answer}")
                    messages.append(ai_msg)
                    return ai_msg
        
        # Initialize include_tool_results variable at the top
        include_tool_results = False
        
        # Extract llm_type from llm
        llm_type = getattr(llm, 'llm_type', None) or getattr(llm, 'type_str', None) or ''
        
        # Create a more explicit reminder to provide final answer
        reminder = self._get_reminder_prompt(
            reminder_type="final_answer_prompt",
            messages=messages,
            tools=self.tools,
            tool_results_history=tool_results_history
        )
        # Gemini-specific: add explicit instructions for extracting numbers or lists
        if llm_type == "gemini":
            reminder += (
                "\n\nIMPORTANT: If the tool result contains a sentence with a number spelled out or as a digit, "
                "extract only the number and provide it as the FINAL ANSWER in the required format. "
                "If the tool result contains a list of items (such as ingredients, or any items), "
                "extract the list and provide it as a comma-separated list in the FINAL ANSWER as required."
            )
        # Check if tool results are already in message history as ToolMessage objects
        has_tool_messages = self._has_tool_messages(messages)
        
        # Determine whether to include tool results in the reminder
        if tool_results_history:
            if llm_type == "gemini":
                include_tool_results = True
            else:
                # For non-Gemini LLMs, only include if not already in message history
                if not has_tool_messages:
                    include_tool_results = True
        
        if include_tool_results:
            tool_results_text = "\n\nTOOL RESULTS:\n" + "\n".join([f"Result {i+1}: {result}" for i, result in enumerate(tool_results_history)])
            reminder += tool_results_text
        
        # Add the reminder to the existing message history
        messages.append(HumanMessage(content=reminder))
        
        try:
            print(f"[Tool Loop] Trying to force the final answer with {len(tool_results_history)} tool results.")
            final_response = self._invoke_llm_provider(llm, messages)
            if hasattr(final_response, 'content') and final_response.content:
                print(f"[Tool Loop] ✅ Final answer generated: {final_response.content[:200]}...")
                return final_response
            else:
                print("[Tool Loop] ❌ LLM returned empty response")
                return AIMessage(content="Unable to determine the answer from the available information.")
        except Exception as e:
            print(f"[Tool Loop] ❌ Failed to get final answer: {e}")
            return AIMessage(content="Error occurred while processing the question.")
        # If Gemini, use a minimal, explicit prompt
        if llm_type == "gemini" and tool_results_history:
            tool_result = tool_results_history[-1]  # Use the latest tool result
            original_question = None
            for msg in messages:
                if hasattr(msg, 'type') and msg.type == 'human':
                    original_question = msg.content
                    break
            if not original_question:
                original_question = "[Original question not found]"
            prompt = (
                "You have already used the tool and obtained the following result:\n\n"
                f"TOOL RESULT:\n{tool_result}\n\n"
                f"QUESTION:\n{original_question}\n\n"
                "INSTRUCTIONS:\n"
                "Extract the answer from the TOOL RESULT above. Your answer must start with 'FINAL ANSWER: [answer]"
                "and follow the system prompt without any extra text numbers, just answer concisely and directly."
            )
            minimal_messages = [self.sys_msg, HumanMessage(content=prompt)]
            
            try:
                final_response = self._invoke_llm_provider(llm, minimal_messages)
                if hasattr(final_response, 'content') and final_response.content:
                    return final_response
                else:
                    # Fallback: return the tool result directly
                    return AIMessage(content=f"RESULT: {tool_result}")
            except Exception as e:
                print(f"[Tool Loop] ❌ Gemini failed to extract final answer: {e}")
                return AIMessage(content=f"RESULT: {tool_result}")

    @trace_prints_with_context("tool_loop")
    def _run_tool_calling_loop(self, llm, messages, tool_registry, llm_type="unknown", model_index: int = 0, call_id: str = None):
        """
        Run a tool-calling loop: repeatedly invoke the LLM, detect tool calls, execute tools, and feed results back until a final answer is produced.
        - Uses adaptive step limits based on LLM type (Gemini: 25, Groq: 15, HuggingFace: 20, unknown: 20).
        - Tracks called tools to prevent duplicate calls and tool results history for fallback handling.
        - Monitors progress by tracking consecutive steps without meaningful changes in response content.
        - Handles LLM invocation failures gracefully with error messages.
        - Detects when responses are truncated due to token limits and adjusts accordingly.
        
        Args:
            llm: The LLM instance (with or without tools bound)
            messages: The message history (list)
            tool_registry: Dict mapping tool names to functions
            llm_type: Type of LLM ("gemini", "groq", "huggingface", or "unknown")
            model_index: Index of the model to use for token limits
        Returns:
            The final LLM response (with content)
        """

        # Adaptive step limits based on LLM type and progress
        base_max_steps = {
            "gemini": 25,    # More steps for Gemini due to better reasoning
            "groq": 5,       # Reduced from 10 to 5 to prevent infinite loops
            "huggingface": 20,  # Conservative for HuggingFace
            "unknown": 20
        }
        max_steps = base_max_steps.get(llm_type, 8)
        
        # Tool calling configuration       
        called_tools = []  # Track which tools have been called to prevent duplicates (stores dictionaries with name, embedding, args)
        tool_results_history = []  # Track tool results for better fallback handling
        current_step_tool_results = []  # Track results from current step only
        consecutive_no_progress = 0  # Track consecutive steps without progress
        last_response_content = ""  # Track last response content for progress detection
        max_total_tool_calls = 10  # Reduced from 15 to 8 to prevent excessive tool usage
        max_tool_calls_per_step = 5  # Maximum tool calls allowed per step
        total_tool_calls = 2  # Track total tool calls to prevent infinite loops
        
        # Simplified tool usage tracking - no special handling for search tools
        tool_usage_limits = {
            'default': 3,
            'wiki_search': 2,
            'web_search': 3, 
            'arxiv_search': 2,
            'analyze_excel_file': 2,
            'analyze_csv_file': 2,
            'analyze_image': 2,
            'extract_text_from_image': 2,
            'exa_ai_helper': 1,
            'web_search_deep_research_exa_ai': 1
        }
        tool_usage_count = {tool_name: 0 for tool_name in tool_usage_limits}
        
        # Detect if the question is text-only (file_name is empty/None)
        is_text_only_question = False
        original_question = ""
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'human':
                original_question = getattr(msg, 'content', "")
                break
        # Try to get file_name from trace or messages
        file_name = getattr(self, 'current_file_name', "")
        if not file_name:
            is_text_only_question = True
        
        for step in range(max_steps):
            response = None
            print(f"\n[Tool Loop] Step {step+1}/{max_steps} - Using LLM: {llm_type}")
            current_step_tool_results = []  # Reset for this step
            
            # Reset Mistral conversion flag for each step
            self._mistral_converted_this_step = False
            
            # ... existing code ...
            # Check if we've exceeded the maximum total tool calls
            if total_tool_calls >= max_total_tool_calls:
                print(f"[Tool Loop] Maximum total tool calls ({max_total_tool_calls}) reached. Calling _force_final_answer ().")
                # Let the LLM generate the final answer from tool results (or lack thereof)
                return self._force_final_answer(messages, tool_results_history, llm)
            
            # Check for excessive tool usage
            for tool_name, count in tool_usage_count.items():
                if count >= tool_usage_limits.get(tool_name, tool_usage_limits['default']):  # Use default limit for unknown tools
                    print(f"[Tool Loop] ⚠️ {tool_name} used {count} times (max: {tool_usage_limits.get(tool_name, tool_usage_limits['default'])}). Preventing further usage.")
                    # Add a message to discourage further use of this tool
                    if step > 2:  # Only add this message after a few steps
                        reminder = self._get_reminder_prompt(
                            reminder_type="tool_usage_issue",
                            tool_name=tool_name,
                            count=count
                        )
                        messages.append(HumanMessage(content=reminder))
            
            # Truncate messages to prevent token overflow
            messages = self._truncate_messages(messages, llm_type)
            
            # Check token limits and summarize if needed
            total_text = "".join(str(getattr(msg, 'content', '')) for msg in messages)
            estimated_tokens = self._estimate_tokens(total_text)
            token_limit = self._get_token_limit(llm_type)
            
            try:
                response = self._invoke_llm_provider(llm, messages)
            except Exception as e:
                # Check if this is a rate limit error that should be throttled
                error_str = str(e)
                if ("429" in error_str or "rate limit" in error_str.lower() or 
                    "service_tier_capacity_exceeded" in error_str or "3505" in error_str or
                    "invalid_request_message_order" in error_str or "3230" in error_str):
                    
                    # Try throttling and retrying instead of immediate fallback
                    if self._handle_rate_limit_throttling(e, llm_type, llm_type):
                        print(f"🔄 [Tool Loop] Retrying {llm_type} after rate limit throttling...")
                        # Continue the loop to retry the same LLM
                        continue
                    else:
                        print(f"⏰ [Tool Loop] Rate limit retries exhausted for {llm_type}, falling back to error handler...")
                        if llm_type == "mistral":
                            self._track_mistral_failure("rate_limit")
                
                # Check for Mistral AI specific message ordering error
                if llm_type == "mistral" and ("invalid_request_message_order" in error_str or "3230" in error_str):
                    return self._handle_mistral_message_ordering_error_in_tool_loop(e, llm_type, messages, llm, tool_results_history)
                else:
                    # Handle other errors normally
                    handled, result = self._handle_llm_error(e, llm_name=llm_type, llm_type=llm_type, phase="tool_loop",
                        messages=messages, llm=llm, tool_results_history=tool_results_history)
                    if handled:
                        return result
                    else:
                        raise

            # Check if response was truncated due to token limits
            if hasattr(response, 'response_metadata') and response.response_metadata:
                finish_reason = response.response_metadata.get('finish_reason')
                if finish_reason == 'length':
                    print(f"[Tool Loop] ❌ Hit token limit for {llm_type} LLM. Response was truncated. Cannot complete reasoning.")
                    # Check if chunking is enabled for this provider
                    config = self.LLM_CONFIG.get(llm_type, {})
                    enable_chunking = config.get("enable_chunking", True)  # Default to True for backward compatibility
                    
                    if enable_chunking:
                        # Handle response truncation using generic token limit error handler
                        print(f"[Tool Loop] Applying chunking mechanism for {llm_type} response truncation")
                        # Get the LLM name for proper logging
                        _, llm_name, _ = self._select_llm(llm_type, True)
                        return self._handle_token_limit_error(messages, llm, llm_name, Exception("Response truncated due to token limit"), llm_type)
                    else:
                        print(f"[Tool Loop] ⚠️ Chunking disabled for {llm_type}. Returning truncated response.")
                        return response

            # === DEBUG OUTPUT ===
            # Print LLM response using the new helper function
            print(f"[Tool Loop] Raw LLM response details:")
            self._print_message_components(response, "response")

            # Check for empty response
            if not hasattr(response, 'content') or not response.content:
                # Allow empty content if there are tool calls (this is normal for tool-calling responses)
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    print(f"[Tool Loop] Empty content but tool calls detected - proceeding with tool execution")
                else:
                    # If we have tool results but no content, force a final answer after 2 consecutive empty responses
                    if tool_results_history and consecutive_no_progress >= 1:
                        print(f"[Tool Loop] Empty content and we have {len(tool_results_history)} tool results for 2 consecutive steps. Forcing final answer.")
                        return self._force_final_answer(messages, tool_results_history, llm)
                    # Otherwise, increment no-progress counter and continue
                    consecutive_no_progress += 1
                    print(f"[Tool Loop] ❌ {llm_type} LLM returned empty response. Consecutive no-progress steps: {consecutive_no_progress}")
                    if consecutive_no_progress >= 2:
                        return AIMessage(content=f"Error: {llm_type} LLM returned empty response. Cannot complete reasoning.")
                    continue
            else:
                consecutive_no_progress = 0  # Reset counter on progress

            # Check for progress (new content or tool calls)
            current_content = getattr(response, 'content', '') or ''
            current_tool_calls = getattr(response, 'tool_calls', []) or []
            has_progress = (current_content != last_response_content or len(current_tool_calls) > 0)
            
            # Check if we have tool results but no final answer yet
            has_tool_results = len(tool_results_history) > 0
            has_final_answer = (hasattr(response, 'content') and response.content and 
                              self._has_final_answer_marker(response))
            
            if has_tool_results and not has_final_answer and step >= 2:  # Increased from 1 to 2 to give more time
                # We have information but no answer - provide explicit reminder to analyze tool results
                reminder = self._get_reminder_prompt(
                    reminder_type="final_answer_prompt",
                    messages=messages,
                    tools=self.tools,
                    tool_results_history=tool_results_history
                )
                messages.append(HumanMessage(content=reminder))
            
            if not has_progress:
                consecutive_no_progress += 1
                print(f"[Tool Loop] No progress detected. Consecutive no-progress steps: {consecutive_no_progress}")
                
                # Exit early if no progress for too many consecutive steps
                if consecutive_no_progress >= 3:  # Increased from 2 to 3
                    print(f"[Tool Loop] Exiting due to {consecutive_no_progress} consecutive steps without progress")
                    # If we have tool results, force a final answer before exiting
                    if tool_results_history:
                        print(f"[Tool Loop] Forcing final answer with {len(tool_results_history)} tool results before exit")
                        return self._force_final_answer(messages, tool_results_history, llm)
                    break
                elif consecutive_no_progress == 1:
                    # Add a gentle reminder to use tools
                    reminder = self._get_reminder_prompt(
                        reminder_type="final_answer_prompt",
                        tools=self.tools
                    )
                    messages.append(HumanMessage(content=reminder))
            else:
                consecutive_no_progress = 0  # Reset counter on progress
                
            last_response_content = current_content

            # If response has content and no tool calls, return
            if hasattr(response, 'content') and response.content and not getattr(response, 'tool_calls', None):
                
                # --- Check for 'FINAL ANSWER' marker ---
                if self._has_final_answer_marker(response):
                    print(f"[Tool Loop] Final answer detected: {response.content}")
                    # Track successful Mistral AI requests
                    if llm_type == "mistral":
                        self._track_mistral_success()
                    return response
                else:
                    # If we have tool results but no FINAL ANSWER marker, force processing
                    if tool_results_history:
                        print(f"[Tool Loop] Content without FINAL ANSWER marker but we have {len(tool_results_history)} tool results. Forcing final answer.")
                        # Track successful Mistral AI requests
                        if llm_type == "mistral":
                            self._track_mistral_success()
                        return self._force_final_answer(messages, tool_results_history, llm)
                    else:
                        # Lean fallback: if the model produced a clear answer without the marker,
                        # wrap it as a FINAL ANSWER to avoid unnecessary retries.
                        print("[Tool Loop] 'FINAL ANSWER' marker not found. Wrapping current content as final answer.")
                        final_text = self._extract_text_from_response(response).strip()
                        wrapped = AIMessage(content=f"FINAL ANSWER: {final_text}")
                        # Track successful Mistral AI requests
                        if llm_type == "mistral":
                            self._track_mistral_success()
                        return wrapped
            tool_calls = getattr(response, 'tool_calls', None)
            if tool_calls:
                print(f"[Tool Loop] Detected {len(tool_calls)} tool call(s)")
                
                # Add tool loop data to trace
                if call_id and self.question_trace:
                    self._add_tool_loop_data(llm_type, call_id, step + 1, tool_calls, consecutive_no_progress)
                
                # IMPORTANT: Preserve the assistant function call in history for providers
                # like GigaChat/OpenAI that require pairing before tool results.
                messages.append(response)

                # Limit the number of tool calls per step to prevent token overflow
                if len(tool_calls) > max_tool_calls_per_step:
                    print(f"[Tool Loop] Too many tool calls on a single step ({len(tool_calls)}). Limiting to first {max_tool_calls_per_step}.")
                    tool_calls = tool_calls[:max_tool_calls_per_step]
                
                # Simplified duplicate detection using new centralized methods
                new_tool_calls = []
                duplicate_count = 0
                for tool_call in tool_calls:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})
                    
                    # Check if tool usage limit exceeded FIRST (most restrictive check)
                    if tool_name in tool_usage_count and tool_usage_count[tool_name] >= tool_usage_limits.get(tool_name, tool_usage_limits['default']):
                        print(f"[Tool Loop] ⚠️ {tool_name} usage limit reached ({tool_usage_count[tool_name]}/{tool_usage_limits.get(tool_name, tool_usage_limits['default'])}). Skipping.")
                        duplicate_count += 1
                        continue
                    
                    # Check if this is a duplicate tool call (SECOND)
                    if self._is_duplicate_tool_call(tool_name, tool_args, called_tools):
                        duplicate_count += 1
                        print(f"[Tool Loop] Duplicate tool call detected: {tool_name} with args: {tool_args}")
                        reminder = self._get_reminder_prompt(
                            reminder_type="tool_usage_issue",
                            tool_name=tool_name,
                            tool_args=tool_args
                        )
                        messages.append(HumanMessage(content=reminder))
                        continue
                    
                    # New tool call - add it (LAST)
                    print(f"[Tool Loop] New tool call: {tool_name} with args: {tool_args}")
                    new_tool_calls.append(tool_call)
                    self._add_tool_call_to_history(tool_name, tool_args, called_tools)
                    
                    # Track tool usage
                    if tool_name in tool_usage_count:
                        tool_usage_count[tool_name] += 1
                        print(f"[Tool Loop] {tool_name} usage: {tool_usage_count[tool_name]}/{tool_usage_limits.get(tool_name, tool_usage_limits['default'])}")
                
                # Only force final answer if ALL tool calls were duplicates AND we have tool results
                if not new_tool_calls and tool_results_history:
                    print(f"[Tool Loop] All {len(tool_calls)} tool calls were duplicates and we have {len(tool_results_history)} tool results. Forcing final answer.")
                    result = self._force_final_answer(messages, tool_results_history, llm)
                    if result:
                        return result
                elif not new_tool_calls and not tool_results_history:
                    # No new tool calls and no previous results - this might be a stuck state
                    print(f"[Tool Loop] All tool calls were duplicates but no previous results. Adding reminder to use available tools.")
                    reminder = self._get_reminder_prompt(reminder_type="tool_usage_issue", tool_name=tool_name)
                    messages.append(HumanMessage(content=reminder))
                    continue
                
                # Execute only new tool calls
                for tool_call in new_tool_calls:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})
                    
                    # Execute tool using helper method with call_id for tracing
                    tool_result = self._execute_tool(tool_name, tool_args, tool_registry, call_id)
                    
                    # Store the raw result for this step
                    current_step_tool_results.append(tool_result)
                    tool_results_history.append(tool_result)
                    total_tool_calls += 1  # Increment total tool call counter
                    
                    # Report tool result
                    self._print_tool_result(tool_name, tool_result)
                    
                    # Add tool result to messages - let LangChain handle the formatting
                    messages.append(ToolMessage(content=tool_result, name=tool_name, tool_call_id=tool_call.get('id', tool_name)))
                
                # Convert messages for Mistral AI if needed (before next LLM call)
                if llm_type == "mistral" and not self._mistral_converted_this_step:
                    messages = self._convert_messages_for_mistral(messages)
                    self._mistral_converted_this_step = True
                
                continue  # Next LLM call
            # Gemini (and some LLMs) may use 'function_call' instead of 'tool_calls'
            function_call = getattr(response, 'function_call', None)
            if function_call:
                tool_name = function_call.get('name')
                tool_args = function_call.get('arguments', {})
                
                # Preserve assistant function call message in history
                messages.append(response)
                
                # Check if this is a duplicate function call
                if self._is_duplicate_tool_call(tool_name, tool_args, called_tools):
                    print(f"[Tool Loop] Duplicate function_call detected: {tool_name} with args: {tool_args}")
                    reminder = self._get_reminder_prompt(
                        reminder_type="tool_usage_issue",
                        tool_name=tool_name,
                        tool_args=tool_args
                    )
                    messages.append(HumanMessage(content=reminder))
                    
                    # Only force final answer if we have tool results
                    if tool_results_history:
                        print(f"[Tool Loop] Duplicate function_call with {len(tool_results_history)} tool results. Forcing final answer.")
                        result = self._force_final_answer(messages, tool_results_history, llm)
                        if result:
                            return result
                    else:
                        # No previous results - add reminder and continue
                        reminder = self._get_reminder_prompt(reminder_type="tool_usage_issue", tool_name=tool_name)
                        messages.append(HumanMessage(content=reminder))
                    continue
                
                # Check if tool usage limit exceeded
                if tool_name in tool_usage_count and tool_usage_count[tool_name] >= tool_usage_limits.get(tool_name, tool_usage_limits['default']):
                    print(f"[Tool Loop] ⚠️ {tool_name} usage limit reached ({tool_usage_count[tool_name]}/{tool_usage_limits.get(tool_name, tool_usage_limits['default'])}). Skipping.")
                    reminder = self._get_reminder_prompt(
                        reminder_type="tool_usage_issue",
                        tool_name=tool_name,
                        count=tool_usage_count[tool_name]
                    )
                    messages.append(HumanMessage(content=reminder))
                    continue
                
                # Add to history and track usage
                self._add_tool_call_to_history(tool_name, tool_args, called_tools)
                if tool_name in tool_usage_count:
                    tool_usage_count[tool_name] += 1
                
                # Execute tool using helper method with call_id for tracing
                tool_result = self._execute_tool(tool_name, tool_args, tool_registry, call_id)
                
                # Store the raw result for this step
                current_step_tool_results.append(tool_result)
                tool_results_history.append(tool_result)
                total_tool_calls += 1  # Increment total tool call counter
                
                # Report tool result (for function_call branch)
                self._print_tool_result(tool_name, tool_result)
                messages.append(ToolMessage(content=tool_result, name=tool_name, tool_call_id=tool_name))
                
                # Convert messages for Mistral AI if needed (after tool results are added)
                if llm_type == "mistral" and not self._mistral_converted_this_step:
                    messages = self._convert_messages_for_mistral(messages)
                    self._mistral_converted_this_step = True
                
                continue
            if hasattr(response, 'content') and response.content:
                return response
            print(f"[Tool Loop] No tool calls or final answer detected. Exiting loop.")
            
            # If we get here, the LLM didn't make tool calls or provide content
            # Add a reminder to use tools or provide an answer
            reminder = self._get_reminder_prompt(reminder_type="final_answer_prompt", tools=self.tools)
            messages.append(HumanMessage(content=reminder))
            continue
        
        # If we reach here, we've exhausted all steps or hit progress limits
        print(f"[Tool Loop] Exiting after {step+1} steps. Last response: {response}")
        
        # If we have tool results but no final answer, force one
        if tool_results_history and (not hasattr(response, 'content') or not response.content or not self._has_final_answer_marker(response)):
            print(f"[Tool Loop] Forcing final answer with {len(tool_results_history)} tool results at loop exit")
            return self._force_final_answer(messages, tool_results_history, llm)
        
        # Return the last response as-is, no partial answer extraction
        return response

    def _select_llm(self, llm_type, use_tools):
        # Updated to use arrays and provider names
        if llm_type not in self.LLM_CONFIG:
            raise ValueError(f"Invalid llm_type: {llm_type}")
        if llm_type not in self.llm_provider_names:
            raise ValueError(f"LLM {llm_type} not initialized")
        idx = self.llm_provider_names.index(llm_type)
        llm = self.llms_with_tools[idx] if use_tools else self.llms[idx]
        llm_name = self.LLM_CONFIG[llm_type]["name"]
        llm_type_str = self.LLM_CONFIG[llm_type]["type_str"]
        
        # Get the actual model name for rate limiting
        model_name = None
        if hasattr(llm, 'model_name'):
            model_name = llm.model_name
        elif hasattr(llm, 'model'):
            model_name = llm.model
        elif llm_type in self.active_model_config:
            # Get the first model from the active config
            models = self.active_model_config[llm_type].get("models", [])
            if models:
                model_name = models[0].get("model")
        
        # Set current model name for rate limiting
        self.current_model_name = model_name
        
        return llm, llm_name, llm_type_str

    @trace_prints_with_context("llm_call")
    def _make_llm_request(self, messages, use_tools=True, llm_type=None):
        """
        Make an LLM request with rate limiting.

        Args:
            messages: The messages to send to the LLM
            use_tools (bool): Whether to use tools (llm_with_tools vs llm)
            llm_type (str): Which LLM to use (mandatory)

        Returns:
            The LLM response

        Raises:
            Exception: If the LLM fails or if llm_type is not specified
        """

        if llm_type is None:
                raise Exception(
                    f"llm_type must be specified for _make_llm_request(). "
                    f"Please specify a valid llm_type from {list(self.LLM_CONFIG.keys())}"
                )
        # Estimate tokens for this request and set for _rate_limit
        total_text = "".join(str(getattr(msg, 'content', '')) for msg in messages)
        estimated_tokens = self._estimate_tokens(total_text)
        self._next_request_tokens = estimated_tokens
        # Start LLM trace
        call_id = self._trace_start_llm(llm_type)
        start_time = time.time()
        
        # Set the current LLM type for rate limiting
        self.current_llm_type = llm_type
        # ENFORCE: Never use tools for providers that do not support them
        if not self._provider_supports_tools(llm_type):
            use_tools = False
        
        # Add input to trace
        self._trace_add_llm_call_input(llm_type, call_id, messages, use_tools)
        
        llm, llm_name, llm_type_str = self._select_llm(llm_type, use_tools)
        if llm is None:
            raise Exception(f"{llm_name} LLM not available")
        
        try:
            self._rate_limit()
            print(f"🤖 Using {llm_name}")
            print(f"--- LLM Prompt/messages sent to {llm_name} ---")
            for i, msg in enumerate(messages):
                self._print_message_components(msg, i)
            tool_registry = {self._get_tool_name(tool): tool for tool in self.tools}
            if use_tools:
                response = self._run_tool_calling_loop(llm, messages, tool_registry, llm_type_str, call_id)
                if not hasattr(response, 'content') or not response.content:
                    print(f"⚠️ {llm_name} tool calling returned empty content, trying without tools...")
                    llm_no_tools, _, _ = self._select_llm(llm_type, False)
                    if llm_no_tools:
                        has_tool_messages = self._has_tool_messages(messages)
                        if has_tool_messages:
                            print(f"⚠️ Retrying {llm_name} without tools (tool results already in message history)")
                            response = llm_no_tools.invoke(messages)
                        else:
                            tool_results_history = []
                            for msg in messages:
                                if hasattr(msg, 'type') and msg.type == 'tool' and hasattr(msg, 'content'):
                                    tool_results_history.append(msg.content)
                            if tool_results_history:
                                print(f"⚠️ Retrying {llm_name} without tools with enhanced context")
                                print(f"📝 Tool results included: {len(tool_results_history)} tools")
                                reminder = self._get_reminder_prompt(
                                    reminder_type="final_answer_prompt",
                                    messages=messages,
                                    tools=self.tools,
                                    tool_results_history=tool_results_history
                                )
                                enhanced_messages = [self.system_prompt, HumanMessage(content=reminder)]
                                response = self._invoke_llm_provider(llm_no_tools, enhanced_messages)
                            else:
                                print(f"⚠️ Retrying {llm_name} without tools (no tool results found)")
                                response = self._invoke_llm_provider(llm_no_tools, messages)
                    if not hasattr(response, 'content') or not response.content:
                        print(f"⚠️ {llm_name} still returning empty content even without tools. This may be a token limit issue.")
                        return AIMessage(content=f"Error: {llm_name} failed due to token limits. Cannot complete reasoning.")
            else:
                response = self._invoke_llm_provider(llm, messages)
            print(f"--- Raw response from {llm_name} ---")
            
            # Add output to trace
            execution_time = time.time() - start_time
            self._trace_add_llm_call_output(llm_type, call_id, response, execution_time)
            
            # Track successful Mistral AI requests
            if llm_type == "mistral":
                self._track_mistral_success()
            
            return response
        except Exception as e:
            # Add error to trace
            execution_time = time.time() - start_time
            self._trace_add_llm_error(llm_type, call_id, e)
            
            # Check if this is a rate limit error that should be throttled
            error_str = str(e)
            if ("429" in error_str or "rate limit" in error_str.lower() or 
                "service_tier_capacity_exceeded" in error_str or "3505" in error_str or
                "invalid_request_message_order" in error_str or "3230" in error_str):
                
                # Try throttling and retrying instead of immediate fallback
                if self._handle_rate_limit_throttling(e, llm_name, llm_type):
                    print(f"🔄 Retrying {llm_name} after rate limit throttling...")
                    # Recursive retry with the same LLM
                    return self._make_llm_request(messages, use_tools, llm_type)
                else:
                    print(f"⏰ Rate limit retries exhausted for {llm_name}, falling back to error handler...")
                    if llm_type == "mistral":
                        self._track_mistral_failure("rate_limit")
            
            # Check for Mistral AI specific message ordering error
            if llm_type == "mistral" and ("invalid_request_message_order" in error_str or "3230" in error_str):
                return self._handle_mistral_message_ordering_error(e, llm_name, llm_type, messages, llm)
            else:
                # Handle other errors normally
                handled, result = self._handle_llm_error(e, llm_name, llm_type, phase="request", messages=messages, llm=llm)
                if handled:
                    return result
                else:
                    raise Exception(f"{llm_name} failed: {e}")

    

    def _handle_groq_token_limit_error(self, messages, llm, llm_name, original_error):
        """
        Handle Groq token limit errors by chunking tool results and processing them in intervals.
        """
        return self._handle_token_limit_error(messages, llm, llm_name, original_error, "groq")

    def _handle_token_limit_error(self, messages, llm, llm_name, original_error, llm_type="unknown"):
        """
        Generic token limit error handling that can be used for any LLM.
        """
        print(f"🔄 Handling token limit error for {llm_name} ({llm_type})")
        
        # Check if chunking is enabled for this provider
        config = self.LLM_CONFIG.get(llm_type, {})
        enable_chunking = config.get("enable_chunking", True)  # Default to True for backward compatibility
        
        if not enable_chunking:
            print(f"⚠️ Chunking disabled for {llm_type}. Cannot handle token limit error.")
            # Return a simple error message instead of chunking
            return AIMessage(content=f"Error: Token limit exceeded for {llm_name} and chunking is disabled. Please try with a shorter input or enable chunking for this provider.")
        
        # Extract tool results from messages
        tool_results = []
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'tool' and hasattr(msg, 'content'):
                tool_results.append(msg.content)
        
        # If no tool results, try to chunk the entire message content
        if not tool_results:
            print(f"📊 No tool results found, attempting to chunk entire message content")
            # Extract all message content
            all_content = []
            for msg in messages:
                if hasattr(msg, 'content') and msg.content:
                    all_content.append(str(msg.content))
            
            if not all_content:
                return AIMessage(content=f"Error: {llm_name} token limit exceeded but no content available to process.")
            
            # Create chunks from all content (use LLM-specific limits)
            token_limit = self._get_token_limit(llm_type)
            # Handle None token limits (like Gemini) by using a reasonable default
            if token_limit is None:
                token_limit = self.LLM_CONFIG["default"]["token_limit"]
            safe_tokens = int(token_limit * 0.60)
            chunks = self._create_token_chunks(all_content, safe_tokens)
            print(f"📦 Created {len(chunks)} chunks from message content")
        else:
            print(f"📊 Found {len(tool_results)} tool results to process in chunks")
            # Create chunks (use LLM-specific limits)
            token_limit = self._get_token_limit(llm_type)
            # Handle None token limits (like Gemini) by using a reasonable default
            if token_limit is None:
                token_limit = self.LLM_CONFIG["default"]["token_limit"]
            safe_tokens = int(token_limit * 0.60)
            chunks = self._create_token_chunks(tool_results, safe_tokens)
            print(f"📦 Created {len(chunks)} chunks from tool results")
        # Ensure original_question is always defined
        original_question = None
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'human' and getattr(msg, 'content', None):
                original_question = msg.content
                break
        if not original_question:
            original_question = '[No original question provided]'
        # Prepare LLM instances for chunking and synthesis
        llm_chunk = self._select_llm(llm_type, use_tools=False)[0]
        llm_final = self._select_llm(llm_type, use_tools=True)[0]
        all_responses = []
        wait_time = 60
        
        for i, chunk in enumerate(chunks):
            print(f"🔄 Processing chunk {i+1}/{len(chunks)}")
            
            # Wait between chunks (except first)
            if i > 0:
                print(f"⏳ Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            # Always use the same prompt for all chunks, now with original question
            chunk_prompt = f"Question: {original_question}\n\nAnalyze these results and provide key findings."
            chunk_content = "\n\n".join(chunk) if isinstance(chunk, list) else str(chunk)
            chunk_messages = [self.sys_msg, HumanMessage(content=chunk_prompt + "\n\n" + chunk_content)]
            try:
                response = llm_chunk.invoke(chunk_messages)
                if hasattr(response, 'content') and response.content:
                    all_responses.append(response.content)
                    print(f"✅ Chunk {i+1} processed")
            except Exception as e:
                print(f"❌ Chunk {i+1} failed: {e}")
                continue
        
        if not all_responses:
            return AIMessage(content=f"Error: Failed to process any chunks for {llm_name}")
        # Final synthesis step, now with original question and tools enabled
        final_prompt = (
            f"Question: {original_question}\n\nCombine these analyses into a final answer:\n\n"
            + "\n\n".join(all_responses)
            + "\n\nProvide your FINAL ANSWER based on all content, following the system prompt format."
        )
        final_messages = [self.sys_msg, HumanMessage(content=final_prompt)]
        try:
            final_response = llm_final.invoke(final_messages)
            return final_response
        except Exception as e:
            print(f"❌ Final synthesis failed: {e}")
            return AIMessage(content=f"OUTPUT {' '.join(all_responses)}")

    def _create_token_chunks(self, tool_results, max_tokens_per_chunk):
        """
        Create chunks of tool results that fit within the token limit.
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for result in tool_results:
            # Use tiktoken for accurate token counting
            result_tokens = self._estimate_tokens(result)
            if current_tokens + result_tokens > max_tokens_per_chunk and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [result]
                current_tokens = result_tokens
            else:
                current_chunk.append(result)
                current_tokens += result_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def _try_llm_sequence(self, messages, use_tools=True, reference=None, llm_sequence=None):
        """
        Try multiple LLMs in sequence, collect all results and their similarity scores, and pick the best one.
        Even if _vector_answers_match returns true, continue with the next models, 
        then choose the best one (highest similarity) or the first one with similar scores.
        Only one attempt per LLM, then move to the next.

        Args:
            messages (list): The messages to send to the LLM.
            use_tools (bool): Whether to use tools.
            reference (str, optional): Reference answer to compare against.
            llm_sequence (list, optional): List of LLM provider keys to use for this call.
        Returns:
            tuple: (answer, llm_used) where answer is the final answer and llm_used is the name of the LLM that succeeded.

        Raises:
            Exception: If all LLMs fail or none produce similar enough answers.
        """
        # Use provided llm_sequence or default
        llm_types_to_use = llm_sequence if llm_sequence is not None else self.DEFAULT_LLM_SEQUENCE
        available_llms = []
        for idx, llm_type in enumerate(self.llm_provider_names):
            # Only use LLMs that are in the provided llm_sequence (if any)
            if llm_type not in llm_types_to_use:
                continue
            # ENFORCE: Never use tools for providers that do not support them
            llm_use_tools = use_tools and self._provider_supports_tools(llm_type)
            llm, llm_name, _ = self._select_llm(llm_type, llm_use_tools)
            # Determine the actual model identifier for this LLM
            model_name = None
            if hasattr(llm, 'model_name'):
                model_name = llm.model_name
            elif hasattr(llm, 'model'):
                model_name = llm.model
            elif llm_type in self.active_model_config:
                models_cfg = self.active_model_config[llm_type].get("models", [])
                if models_cfg:
                    model_name = models_cfg[0].get("model")
            if llm:
                # Append provider type, human-readable provider name, tools flag, and actual model id
                available_llms.append((llm_type, llm_name, llm_use_tools, model_name))
            else:
                print(f"⚠️ {llm_name} not available, skipping...")
        if not available_llms:
            raise Exception("No LLMs are available. Please check your API keys and configuration.")
        print(f"🔄 Available LLMs: {[name for _, name, _, _ in available_llms]}")
        original_question = ""
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'human':
                original_question = msg.content
                break
        llm_results = []
        for llm_type, llm_name, llm_use_tools, model_name in available_llms:
            try:
                response = self._make_llm_request(messages, use_tools=llm_use_tools, llm_type=llm_type)
                answer = self._extract_final_answer(response)
                print(f"✅ {llm_name} answered: {answer}")
                print(f"✅ Reference: {reference}")
                
                # Capture stdout for this LLM attempt
                if hasattr(self, 'current_llm_call_id'):
                    self._trace_capture_llm_stdout(llm_type, self.current_llm_call_id)
                
                if reference is None:
                    print(f"✅ {llm_name} succeeded (no reference to compare)")
                    self._update_llm_tracking(llm_type, "success")
                    self._update_llm_tracking(llm_type, "submitted")  # Mark as submitted since it's the final answer
                    # Store actual model identifier for downstream display
                    llm_results.append((1.0, answer, model_name or llm_name, llm_type))
                    break
                is_match, similarity = self._vector_answers_match(answer, reference)
                if is_match:
                    print(f"✅ {llm_name} succeeded with similar answer to reference")
                else:
                    print(f"⚠️ {llm_name} succeeded but answer doesn't match reference")
                # Prefer actual model id; fall back to provider name if unavailable
                llm_results.append((similarity, answer, model_name or llm_name, llm_type))
                if similarity >= self.similarity_threshold:
                    self._update_llm_tracking(llm_type, "threshold_pass")
                if llm_type != available_llms[-1][0]:
                    print(f"🔄 Trying next LLM without reference...")
                else:
                    print(f"🔄 All LLMs tried, all failed")
            except Exception as e:
                print(f"❌ {llm_name} failed: {e}")
                
                # Capture stdout for this failed LLM attempt
                if hasattr(self, 'current_llm_call_id'):
                    self._trace_capture_llm_stdout(llm_type, self.current_llm_call_id)
                
                self._update_llm_tracking(llm_type, "failure")
                if llm_type == available_llms[-1][0]:
                    raise Exception(f"All available LLMs failed. Last error from {llm_name}: {e}")
                print(f"🔄 Trying next LLM...")
        # --- Finalist selection and stats update ---
        if llm_results:
            threshold = self.similarity_threshold
            for sim, ans, name, llm_type in llm_results:
                if sim >= threshold:
                    print(f"🎯 First answer above threshold: {ans} (LLM: {name}, similarity: {sim:.3f})")
                    self._update_llm_tracking(llm_type, "submitted")
                    return ans, name
            # If none above threshold, pick best similarity as low score submission
            best_similarity, best_answer, best_llm, best_llm_type = max(llm_results, key=lambda x: x[0])
            print(f"🔄 Returning best answer by similarity: {best_answer} (LLM: {best_llm}, similarity: {best_similarity:.3f})")
            self._update_llm_tracking(best_llm_type, "low_submit")
            return best_answer, best_llm
        raise Exception("All LLMs failed")

    def _get_reference_answer(self, question: str) -> Optional[str]:
        """
        Retrieve the reference answer for a question using the vector store manager.

        Args:
            question (str): The question text.

        Returns:
            str or None: The reference answer if found, else None.
        """
        return get_reference_answer(question)

    def _format_messages(self, question: str, reference: Optional[str] = None, chat_history: Optional[List[Dict[str, Any]]] = None) -> List[Any]:
        """
        Format the message list for the LLM, including system prompt, optional prior chat history,
        question, and optional reference answer.

        Args:
            question (str): The question to answer.
            reference (str, optional): The reference answer to include in context.
            chat_history (list, optional): Prior conversation turns as list of {role, content}.

        Returns:
            list: List of message objects for the LLM.
        """
        messages = [self.sys_msg]
        # Append prior chat history (user/assistant only) for continuity
        if chat_history and isinstance(chat_history, list):
            for turn in chat_history:
                try:
                    role = str(turn.get("role", "")).lower()
                    content = str(turn.get("content", ""))
                except Exception:
                    continue
                if not content:
                    continue
                if role in ("user", "human"):
                    messages.append(HumanMessage(content=content))
                elif role in ("assistant", "ai"):
                    messages.append(AIMessage(content=content))
        # Current question last
        messages.append(HumanMessage(content=question))
        if reference:
            messages.append(HumanMessage(content=f"Reference answer: {reference}"))
        return messages

    def _clean_final_answer_text(self, text: str) -> str:
        """
        Extracts and cleans the answer after 'FINAL ANSWER' marker 
        (case-insensitive, optional colon/space).
        Strips and normalizes whitespace.
        """
        # Handle None text gracefully
        if not text:
            return ""
        # Remove everything before and including 'final answer' (case-insensitive, optional colon/space)
        match = re.search(r'final answer\s*:?', text, flags=re.IGNORECASE)
        if match:
            text = text[match.end():]
        # Normalize whitespace and any JSON remainders
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lstrip('{[\'').rstrip(']]}"\'')
        return text.strip()

    def _get_tool_name(self, tool):
        if hasattr(tool, 'name'):
            return tool.name
        elif hasattr(tool, '__name__'):
            return tool.__name__
        else:
            return str(tool)

    def _convert_messages_for_mistral(self, messages: List) -> List:
        """
        Convert LangChain messages to Mistral AI compatible format.
        Mistral AI requires specific message formatting for tool calls.
        The key issue is that tool messages must immediately follow the assistant message
        that made the tool calls, not after user messages.
        
        Args:
            messages: List of LangChain message objects
            
        Returns:
            List of messages in Mistral AI compatible format
        """
        converted_messages = []
        i = 0
        orphaned_count = 0
        
        while i < len(messages):
            msg = messages[i]
            
            if hasattr(msg, 'type'):
                if msg.type == 'system':
                    converted_messages.append({
                        "role": "system",
                        "content": msg.content
                    })
                elif msg.type == 'human':
                    converted_messages.append({
                        "role": "user",
                        "content": msg.content
                    })
                elif msg.type == 'ai':
                    # For AI messages with tool calls, we need to handle them specially
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        # Add the assistant message with tool calls
                        converted_messages.append({
                            "role": "assistant",
                            "content": msg.content or "",
                            "tool_calls": msg.tool_calls
                        })
                        
                        # Look ahead for tool messages that should immediately follow
                        j = i + 1
                        while j < len(messages) and hasattr(messages[j], 'type') and messages[j].type == 'tool':
                            tool_msg = messages[j]
                            converted_messages.append({
                                "role": "tool",
                                "name": getattr(tool_msg, 'name', 'unknown'),
                                "content": tool_msg.content,
                                "tool_call_id": getattr(tool_msg, 'tool_call_id', getattr(tool_msg, 'name', 'unknown'))
                            })
                            j += 1
                        
                        # Skip the tool messages we've already processed
                        i = j - 1
                    else:
                        converted_messages.append({
                            "role": "assistant",
                            "content": msg.content or ""
                        })
                elif msg.type == 'tool':
                    # Tool messages should only appear after assistant messages with tool calls
                    # If we encounter one here, it might be orphaned - skip it
                    # Only log the first few orphaned messages to avoid spam
                    if orphaned_count < 2:
                        print(f"[Mistral Conversion] Warning: Orphaned tool message detected: {getattr(msg, 'name', 'unknown')}")
                        orphaned_count += 1
                    continue
            else:
                # Handle raw message objects (fallback)
                converted_messages.append(msg)
            
            i += 1
        
        return converted_messages

    def _invoke_llm_provider(self, llm, messages, llm_type=None):
        """
        Helper function to invoke LLM with automatic Mistral-specific message conversion.
        This centralizes the repetitive Mistral conversion logic and makes the code cleaner.
        Note: This function assumes messages are already in the correct format for Mistral AI.
        For tool loops where tool results are added after conversion, use manual conversion.
        
        Args:
            llm: The LLM instance to invoke
            messages: List of messages to send to the LLM
            llm_type: Type of LLM (if None, will be auto-detected)
            
        Returns:
            The LLM response
        """
        # Auto-detect LLM type if not provided
        if llm_type is None:
            llm_type = getattr(llm, 'llm_type', None) or getattr(llm, 'type_str', None) or ''
        
        # Convert messages for providers that require strict tool-call ordering (e.g., Mistral, GigaChat)
        if llm_type in ("mistral", "gigachat"):
            messages = self._convert_messages_for_mistral(messages)
        
        # Invoke the LLM
        return llm.invoke(messages)

        
    def _calculate_cosine_similarity(self, embedding1, embedding2) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Cosine similarity score (0.0 to 1.0)
        """
        return vector_store_manager.calculate_cosine_similarity(embedding1, embedding2)

    def _vector_answers_match(self, answer: str, reference: str):
        """
        Return (bool, similarity) where bool is if similarity >= threshold, and similarity is the float value.
        """
        return vector_answers_match(answer, reference, self.similarity_threshold)

    def get_llm_stats(self) -> dict:
        stats = {
            "total_questions": self.total_questions,
            "llm_stats": {},
            "summary": {}
        }
        used_models = {}
        for llm_type in self.llm_tracking.keys():
            model_id = None
            if llm_type in self.active_model_config:
                model_id = self.active_model_config[llm_type].get("model", "")
            used_models[llm_type] = model_id
        llm_types = list(self.llm_tracking.keys())
        total_submitted = 0
        total_low_submit = 0
        total_passed = 0
        total_failures = 0
        total_attempts = 0
        for llm_type in llm_types:
            llm_name = self.LLM_CONFIG[llm_type]["name"]
            model_id = used_models.get(llm_type, "")
            display_name = f"{llm_name} ({model_id})" if model_id else llm_name
            tracking = self.llm_tracking[llm_type]
            successes = tracking["successes"]
            failures = tracking["failures"]
            threshold_count = tracking["threshold_passes"]
            submitted = tracking["submitted"]
            low_submit = tracking["low_submit"]
            attempts = tracking["total_attempts"]
            total_submitted += submitted
            total_low_submit += low_submit
            total_passed += successes
            total_failures += failures
            total_attempts += attempts
            pass_rate = (successes / attempts * 100) if attempts > 0 else 0
            fail_rate = (failures / attempts * 100) if attempts > 0 else 0
            submit_rate = (submitted / self.total_questions * 100) if self.total_questions > 0 else 0
            stats["llm_stats"][display_name] = {
                "runs": attempts,
                "passed": successes,
                "pass_rate": f"{pass_rate:.1f}",
                "submitted": submitted,
                "submit_rate": f"{submit_rate:.1f}",
                "low_submit": low_submit,
                "failed": failures,
                "fail_rate": f"{fail_rate:.1f}",
                "threshold": threshold_count
            }
        overall_submit_rate = (total_submitted / self.total_questions * 100) if self.total_questions > 0 else 0
        stats["summary"] = {
            "total_questions": self.total_questions,
            "total_submitted": total_submitted,
            "total_low_submit": total_low_submit,
            "total_passed": total_passed,
            "total_failures": total_failures,
            "total_attempts": total_attempts,
            "overall_submit_rate": f"{overall_submit_rate:.1f}"
        }
        return stats

    def _format_llm_init_summary(self, as_str=True):
        """
        Return the LLM initialization summary as a formatted table string (for printing or saving).
        """
        if not hasattr(self, 'llm_init_results') or not self.llm_init_results:
            return ""
        provider_w = max(14, max(len(r['provider']) for r in self.llm_init_results) + 2)
        model_w = max(40, max(len(r['model']) for r in self.llm_init_results) + 2)
        plain_w = max(5, len('Plain'))
        tools_w = max(5, len('Tools (forced)'))
        error_w = max(20, len('Error (tools)'))
        header = (
            f"{'Provider':<{provider_w}}| "
            f"{'Model':<{model_w}}| "
            f"{'Plain':<{plain_w}}| "
            f"{'Tools':<{tools_w}}| "
            f"{'Error (tools)':<{error_w}}"
        )
        lines = ["===== LLM Initialization Summary =====", header, "-" * len(header)]
        for r in self.llm_init_results:
            plain = '✅' if r['plain_ok'] else '❌'
            config = self.LLM_CONFIG.get(r['llm_type'], {})
            model_force_tools = False
            for m in config.get('models', []):
                if m.get('model', '') == r['model']:
                    model_force_tools = config.get('force_tools', False) or m.get('force_tools', False)
                    break
            if r['tools_ok'] is None:
                tools = 'N/A'
            else:
                tools = '✅' if r['tools_ok'] else '❌'
            if model_force_tools:
                tools += ' (forced)'
            error_tools = ''
            if r['tools_ok'] is False and r['error_tools']:
                if '400' in r['error_tools']:
                    error_tools = '400'
                else:
                    error_tools = r['error_tools'][:18]
            lines.append(f"{r['provider']:<{provider_w}}| {r['model']:<{model_w}}| {plain:<{plain_w}}| {tools:<{tools_w}}| {error_tools:<{error_w}}")
        lines.append("=" * len(header))
        return "\n".join(lines) if as_str else lines

    def _get_llm_init_summary_json(self):
        """
        Return the LLM initialization summary as structured JSON data for dataset upload.
        """
        if not hasattr(self, 'llm_init_results') or not self.llm_init_results:
            return {}
        
        summary_data = {
            "results": []
        }
        
        for r in self.llm_init_results:
            config = self.LLM_CONFIG.get(r['llm_type'], {})
            model_force_tools = False
            for m in config.get('models', []):
                if m.get('model', '') == r['model']:
                    model_force_tools = config.get('force_tools', False) or m.get('force_tools', False)
                    break
            
            result_entry = {
                "provider": r['provider'],
                "model": r['model'],
                "llm_type": r['llm_type'],
                "plain_ok": r['plain_ok'],
                "tools_ok": r['tools_ok'],
                "force_tools": model_force_tools,
                "error_tools": r.get('error_tools', ''),
                "error_plain": r.get('error_plain', '')
            }
            summary_data["results"].append(result_entry)
        
        return summary_data

    def _format_llm_stats_table(self, as_str=True):
        """
        Return the LLM statistics as a formatted table string (for printing or saving).
        """
        stats = self.get_llm_stats()
        rows = []
        for name, data in stats["llm_stats"].items():
            # Show LLMs that have any activity (runs, submitted, low_submit, or any other activity)
            if (data["runs"] > 0 or data["submitted"] > 0 or data["low_submit"] > 0 or 
                data["passed"] > 0 or data["failed"] > 0 or data["threshold"] > 0):
                rows.append([
                    name,
                    data["runs"],
                    data["passed"],
                    data["pass_rate"],
                    data["submitted"],
                    data["submit_rate"],
                    data["low_submit"],
                    data["failed"],
                    data["fail_rate"],
                    data["threshold"]
                ])
        header = [
            "Model", "Runs", "Passed", "Pass %", "Submitted", "Submit %", "LowSubmit", "Failed", "Fail %", "Threshold"
        ]
        col_widths = [max(len(str(row[i])) for row in ([header] + rows)) for i in range(len(header))]
        def fmt_row(row):
            return " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
        lines = ["===== LLM Model Statistics =====", fmt_row(header), "-" * (sum(col_widths) + 3 * (len(header) - 1))]
        for row in rows:
            lines.append(fmt_row(row))
        # Add true totals row for numeric columns
        totals = ["TOTALS"]
        for i, col in enumerate(header[1:], 1):
            if col.endswith("%"):
                totals.append("")
            else:
                totals.append(sum(row[i] for row in rows if isinstance(row[i], (int, float))))
        lines.append(fmt_row(totals))
        lines.append("-" * (sum(col_widths) + 3 * (len(header) - 1)))
        s = stats["summary"]
        lines.append(f"Above Threshold Submissions: {s['total_submitted']} / {s['total_questions']} ({s['overall_submit_rate']}%)")
        lines.append("=" * (sum(col_widths) + 3 * (len(header) - 1)))
        return "\n".join(lines) if as_str else lines

    def _get_llm_stats_json(self):
        """
        Return the LLM statistics as structured JSON data for dataset upload.
        """
        stats = self.get_llm_stats()
        
        stats_data = {
            "llm_stats": {}
        }
        
        for name, data in stats["llm_stats"].items():
            # Include all LLMs that have any activity
            if (data["runs"] > 0 or data["submitted"] > 0 or data["low_submit"] > 0 or 
                data["passed"] > 0 or data["failed"] > 0 or data["threshold"] > 0):
                stats_data["llm_stats"][name] = {
                    "runs": data["runs"],
                    "passed": data["passed"],
                    "pass_rate": data["pass_rate"],
                    "submitted": data["submitted"],
                    "submit_rate": data["submit_rate"],
                    "low_submit": data["low_submit"],
                    "failed": data["failed"],
                    "fail_rate": data["fail_rate"],
                    "threshold": data["threshold"],
                    "successes": data.get("successes", 0),
                    "failures": data.get("failures", 0),
                    "total_attempts": data.get("total_attempts", 0),
                    "threshold_passes": data.get("threshold_passes", 0)
                }
        
        return stats_data

    def _print_llm_init_summary(self):
        summary = self._format_llm_init_summary(as_str=True)
        if summary:
            print("\n" + summary + "\n")

    def print_llm_stats_table(self):
        summary = self._format_llm_stats_table(as_str=True)
        if summary:
            print("\n" + summary + "\n")

    def _update_llm_tracking(self, llm_type: str, event_type: str, increment: int = 1):
        """
        Helper method to update LLM tracking statistics.
        
        Args:
            llm_type (str): The LLM type (e.g., 'gemini', 'groq')
            event_type (str): The type of event ('success', 'failure', 'threshold_pass', 'submitted', 'low_submit')
            increment (int): Amount to increment (default: 1)
        """
        if llm_type not in self.llm_tracking:
            return
        if event_type == "success":
            self.llm_tracking[llm_type]["successes"] += increment
            self.llm_tracking[llm_type]["total_attempts"] += increment
        elif event_type == "failure":
            self.llm_tracking[llm_type]["failures"] += increment
            self.llm_tracking[llm_type]["total_attempts"] += increment
        elif event_type == "threshold_pass":
            self.llm_tracking[llm_type]["threshold_passes"] += increment
        elif event_type == "submitted":
            self.llm_tracking[llm_type]["submitted"] += increment
            # Ensure total_attempts is incremented for submitted events if not already counted
            if self.llm_tracking[llm_type]["total_attempts"] == 0:
                self.llm_tracking[llm_type]["total_attempts"] += increment
        elif event_type == "low_submit":
            self.llm_tracking[llm_type]["low_submit"] += increment
            # Ensure total_attempts is incremented for low_submit events if not already counted
            if self.llm_tracking[llm_type]["total_attempts"] == 0:
                self.llm_tracking[llm_type]["total_attempts"] += increment

    @trace_prints_with_context("question")
    def __call__(self, question: str, file_data: str = None, file_name: str = None, llm_sequence: list = None, chat_history: Optional[List[Dict[str, Any]]] = None) -> dict:
        """
        Run the agent on a single question, using step-by-step reasoning and tools.

        Args:
            question (str): The question to answer.
            file_data (str, optional): Base64 encoded file data if a file is attached.
            file_name (str, optional): Name of the attached file.
            llm_sequence (list, optional): List of LLM provider keys to use for this call.
            chat_history (list, optional): Prior conversation to maintain continuity.
        Returns:
            dict: Dictionary containing:
                - answer: The agent's final answer, formatted per system_prompt
                - similarity_score: Similarity score against reference (0.0-1.0)
                - llm_used: Name of the LLM that provided the answer
                - reference: Reference answer used for comparison, or "Reference answer not found"
                - question: Original question text
                - file_name: Name of attached file (if any)
                - error: Error message (if any error occurred)

        Workflow:
            1. Store file data for use by tools.
            2. Retrieve similar Q/A for context using the retriever.
            3. Use LLM sequence with similarity checking against reference.
            4. If no similar answer found, fall back to reference answer.
        """
        # Initialize trace for this question
        self._trace_init_question(question, file_data, file_name)
        
        print(f"\n🔎 Processing question: {question}\n")
        
        # Increment total questions counter
        self.total_questions += 1
        
        # Store the original question for reuse throughout the process
        self.original_question = question
        
        # Store file data for use by tools
        self.current_file_data = file_data
        self.current_file_name = file_name
        
        if file_data and file_name:
            print(f"📁 File attached: {file_name} ({len(file_data)} chars base64)")
        
        # 1. Retrieve similar Q/A for context
        reference = self._get_reference_answer(question)
        
        # 2. Step-by-step reasoning with LLM sequence and similarity checking
        messages = self._format_messages(question, reference=reference, chat_history=chat_history)
        try:
            answer, llm_used = self._try_llm_sequence(messages, use_tools=True, reference=reference, llm_sequence=llm_sequence)
            print(f"🎯 Final answer from {llm_used}")
            
            # Calculate similarity score if reference exists
            similarity_score = 0.0
            if reference:
                is_match, similarity_score = self._vector_answers_match(answer, reference)
            else:
                similarity_score = 1.0  # No reference to compare against
                
            # Display comprehensive stats
            self.print_llm_stats_table()
            
            # # Return structured result
            # Use helper function to ensure valid answer
            final_answer = {
                "submitted_answer": ensure_valid_answer(answer),  # Consistent field name
                "similarity_score": similarity_score,
                "llm_used": llm_used,
                "reference": reference if reference else "Reference answer not found",
                "question": question
            }
            
            # Finalize trace with success result
            self._trace_finalize_question(final_answer)
            
            result = self._trace_get_full()
            return result
            
        except Exception as e:
            print(f"❌ All LLMs failed: {e}")
            self.print_llm_stats_table()
            
            # Return error result
            error_result = {
                "submitted_answer": f"Error: {e}",  # Consistent field name - never None
                "similarity_score": 0.0,
                "llm_used": "none",
                "reference": reference if reference else "Reference answer not found",
                "question": question,
                "error": str(e)
            }
            
            # Finalize trace with error result
            self._trace_finalize_question(error_result)
            
            # Add trace to the result
            error_result = self._trace_get_full()
            
            return error_result

    def _extract_text_from_response(self, response: Any) -> str:
        """
        Helper method to extract text content from various response object types.
        
        Args:
            response (Any): The response object (could be LLM response, dict, or string)
            
        Returns:
            str: The text content from the response
        """
        # Handle None responses gracefully
        if not response:
            return ""
            
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, dict) and 'content' in response:
            return response['content']
        else:
            return str(response)

    def _has_final_answer_marker(self, response: Any) -> bool:
        """
        Check if the LLM response contains a "FINAL ANSWER:" marker.
        This is used in the tool calling loop to determine if the response is a final answer.

        Args:
            response (Any): The LLM response object.

        Returns:
            bool: True if the response contains "FINAL ANSWER:" marker, False otherwise.
        """
        text = self._extract_text_from_response(response)
        # Check if any line contains 'final answer' (case-insensitive, optional colon/space)
        for line in text.splitlines():
            if re.search(r'final answer\s*:?', line, flags=re.IGNORECASE):
                return True
        return False



    def _extract_final_answer(self, response: Any) -> str:
        """
        Extract the final answer from the LLM response, removing the "FINAL ANSWER:" prefix.
        The LLM is responsible for following the system prompt formatting rules.
        This method is used for validation against reference answers and submission.

        Args:
            response (Any): The LLM response object.

        Returns:
            str: The extracted final answer string with "FINAL ANSWER:" prefix removed, or default string if not found.
        """
        # Extract text from response
        text = self._extract_text_from_response(response)
        # If marker exists, clean after marker; otherwise, use stripped text
        if self._has_final_answer_marker(text):
            cleaned_answer = self._clean_final_answer_text(text)
        else:
            cleaned_answer = (text or "").strip()
        
        # Use helper function to ensure valid answer
        return ensure_valid_answer(cleaned_answer)

    def _llm_answers_match(self, answer: str, reference: str) -> bool:
        """
        Use the LLM to validate whether the agent's answer matches the reference answer according to the system prompt rules.
        This method is kept for compatibility but should be avoided due to rate limiting.

        Args:
            answer (str): The agent's answer.
            reference (str): The reference answer.

        Returns:
            bool: True if the LLM determines the answers match, False otherwise.
        """
        validation_prompt = (
            f"Agent's answer:\n{answer}\n\n"
            f"Reference answer:\n{reference}\n\n"
            "Question: Does the agent's answer match the reference answer exactly, following the system prompt's answer formatting and constraints? "
            "Reply with only 'true' or 'false'."
        )
        validation_msg = [SystemMessage (content=self.system_prompt), HumanMessage(content=validation_prompt)]
        try:
            response = self._try_llm_sequence(validation_msg, use_tools=False)
            result = self._extract_text_from_response(response).strip().lower()
            return result.startswith('true')
        except Exception as e:
            # Fallback: conservative, treat as not matching if validation fails
            print(f"LLM validation error in _llm_answers_match: {e}")
            return False

    def _gather_tools(self) -> List[Any]:
        """
        Gather all callable tools from tools.py for LLM tool binding.

        Returns:
            list: List of tool functions.
        """
       
        # Get all attributes from the tools module
        tool_list = []
        for name, obj in tools.__dict__.items():
            # Only include actual tool objects (decorated with @tool) or callable functions
            # that are not classes, modules, or builtins
            if (callable(obj) and 
                not name.startswith("_") and 
                not isinstance(obj, type) and  # Exclude classes
                hasattr(obj, '__module__') and  # Must have __module__ attribute
                obj.__module__ == 'tools' and  # Must be from tools module
                name not in ["GaiaAgent", "CodeInterpreter"]):  # Exclude specific classes
                
                # Check if it's a proper tool object (has the tool attributes)
                if hasattr(obj, 'name') and hasattr(obj, 'description'):
                    # This is a proper @tool decorated function
                    tool_list.append(obj)
                elif callable(obj) and not name.startswith("_"):
                    # This is a regular function that might be a tool
                    # Only include if it's not an internal function
                    if not name.startswith("_") and name not in [
                        "_convert_chess_move_internal", 
                        "_get_best_chess_move_internal", 
                        "_get_chess_board_fen_internal",
                        "_expand_fen_rank",
                        "_compress_fen_rank", 
                        "_invert_mirror_fen",
                        "_add_fen_game_state"
                    ]:
                        tool_list.append(obj)
        
        # Add specific tools that might be missed
        specific_tools = [
            # List of specific tool names to ensure inclusion (grouped by category for clarity)
                # Math tools
                'multiply', 'add', 'subtract', 'divide', 'modulus', 'power', 'square_root',
                # File and data tools
                'save_and_read_file', 'download_file_from_url', 'get_task_file',
                # Image and media tools
                'extract_text_from_image', 'analyze_csv_file', 'analyze_excel_file',
                'analyze_image', 'transform_image', 'draw_on_image', 'generate_simple_image', 'combine_images',
                'understand_video', 'understand_audio',
                # Chess tools
                'convert_chess_move', 'get_best_chess_move', 'get_chess_board_fen', 'solve_chess_position',
                # Code execution
                'execute_code_multilang',
                # Research and search tools
                'web_search_deep_research_exa_ai', 'exa_ai_helper', 
                'wiki_search', 'arxiv_search', 'web_search',
                # Comindware Platform tools
                'edit_or_create_text_attribute', 'get_text_attribute'
        ]
        
        # Build a set of tool names for deduplication (handle both __name__ and .name attributes)
        tool_names = set(self._get_tool_name(tool) for tool in tool_list)
        
        # Ensure all specific tools are included
        for tool_name in specific_tools:
            if hasattr(tools, tool_name):
                tool_obj = getattr(tools, tool_name)
                name_val = self._get_tool_name(tool_obj)
                if name_val not in tool_names:
                    tool_list.append(tool_obj)
                    tool_names.add(name_val)
        
        # Add retriever tool from vector store if available
        retriever_tool = get_retriever_tool()
        if retriever_tool:
            tool_list.append(retriever_tool)
            print("✅ Added retriever tool from vector store")
        else:
            print("ℹ️ No retriever tool available (vector store disabled)")
        
        # Filter out any tools that don't have proper tool attributes
        final_tool_list = []
        for tool in tool_list:
            if hasattr(tool, 'name') and hasattr(tool, 'description'):
                # This is a proper tool object
                final_tool_list.append(tool)
            elif callable(tool) and not self._get_tool_name(tool).startswith("_"):
                # This is a callable function that should be a tool
                final_tool_list.append(tool)
        
        print(f"✅ Gathered {len(final_tool_list)} tools: {[self._get_tool_name(tool) for tool in final_tool_list]}")
        return final_tool_list

    def _filter_tools_for_llm(self, tools_list: List[Any], llm_type: str) -> List[Any]:
        """
        Filter tools to avoid provider-specific JSON Schema issues.

        - For providers like 'gigachat' that are strict about JSON schema, drop tools
          whose argument schemas contain open-ended Dict/Any objects without defined properties
          (e.g., parameters named 'params' typed as Dict[str, Any]).

        Args:
            tools_list: Tools gathered for binding
            llm_type: Provider type string

        Returns:
            List[Any]: Filtered list of tools
        """
        if not tools_list:
            return tools_list
        if llm_type not in ("gigachat",):
            return tools_list

        filtered = []
        for t in tools_list:
            try:
                # LangChain tools expose args schema via .args or .args_schema, fall back to __signature__
                schema = None
                if hasattr(t, "args") and isinstance(getattr(t, "args"), dict):
                    schema = getattr(t, "args")
                elif hasattr(t, "args_schema") and getattr(t, "args_schema") is not None:
                    try:
                        schema = getattr(t, "args_schema").schema().get("properties", {})
                    except Exception:
                        schema = None
                if schema is None and hasattr(t, "__signature__") and getattr(t, "__signature__") is not None:
                    # Build a simple schema-like view from signature annotations
                    sig = getattr(t, "__signature__")
                    pseudo = {}
                    for p in sig.parameters.values():
                        pseudo[p.name] = str(p.annotation) if p.annotation is not p.empty else "Any"
                    schema = pseudo

                # Decide if tool is safe: reject if any param looks like an unstructured dict
                unsafe = False
                if isinstance(schema, dict):
                    for k, v in schema.items():
                        v_str = str(v).lower()
                        if k in ("params", "options", "kwargs") or ("dict" in v_str and ("any" in v_str or "typing.any" in v_str)):
                            unsafe = True
                            break
                if not unsafe:
                    filtered.append(t)
                else:
                    print(f"ℹ️ Skipping tool for {llm_type} due to open dict param: {self._get_tool_name(t)}")
            except Exception as e:
                # If inspection fails, keep the tool to avoid accidental removal
                print(f"ℹ️ Tool inspection failed for {self._get_tool_name(t)}: {e}. Keeping tool.")
                filtered.append(t)

        if not filtered:
            # Fallback: if filtering removed everything, return original list
            return tools_list
        return filtered

    def _inject_file_data_to_tool_args(self, tool_name: str, tool_args: dict) -> dict:
        """
        Automatically inject file data and system prompt into tool arguments if needed.
        
        Args:
            tool_name (str): Name of the tool being called
            tool_args (dict): Original tool arguments
            
        Returns:
            dict: Modified tool arguments with file data and system prompt if needed
        """
        # Tools that need file data
        file_tools = {
            'understand_audio': 'file_path',
            'analyze_image': 'image_base64', 
            'transform_image': 'image_base64',
            'draw_on_image': 'image_base64',
            'combine_images': 'images_base64',
            'extract_text_from_image': 'image_path',
            'analyze_csv_file': 'file_path',
            'analyze_excel_file': 'file_path',
            'get_chess_board_fen': 'image_path',
            'solve_chess_position': 'image_path',
            'execute_code_multilang': 'code'  # Add support for code injection
        }
        
        # Tools that need system prompt for better formatting
        system_prompt_tools = ['understand_video', 'understand_audio']
        
        # Inject system prompt for video and audio understanding tools
        if tool_name in system_prompt_tools and 'system_prompt' not in tool_args:
            tool_args['system_prompt'] = self.system_prompt
            print(f"[Tool Loop] Injected system prompt for {tool_name}")
        
        if tool_name in file_tools and self.current_file_data and self.current_file_name:
            param_name = file_tools[tool_name]
            
            # For image tools, use base64 directly
            if 'image' in param_name:
                tool_args[param_name] = self.current_file_data
                print(f"[Tool Loop] Injected base64 image data for {tool_name}")
            # For file path tools, create a temporary file
            elif 'file_path' in param_name:
                # Decode base64 and create temporary file
                file_data = base64.b64decode(self.current_file_data)
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(self.current_file_name)[1], delete=False) as temp_file:
                    temp_file.write(file_data)
                    temp_file_path = temp_file.name
                tool_args[param_name] = temp_file_path
                print(f"[Tool Loop] Created temporary file {temp_file_path} for {tool_name}")
            # For code tools, decode and inject the code content
            elif param_name == 'code':
                try:
                    # Get file extension
                    temp_ext = os.path.splitext(self.current_file_name)[1].lower()
                    code_str = tool_args.get('code', '')
                    orig_file_name = self.current_file_name
                    file_data = base64.b64decode(self.current_file_data)
                    # List of code file extensions
                    code_exts = ['.py', '.js', '.cpp', '.c', '.java', '.rb', '.go', '.ts', '.sh', '.php', '.rs']
                    if temp_ext in code_exts:
                        # If it's a code file, decode as UTF-8 and inject as code
                        code_content = file_data.decode('utf-8')
                        tool_args[param_name] = code_content
                        print(f"[Tool Loop] Injected code from attached file for {tool_name}: {len(code_content)} characters")
                    else:
                        # Otherwise, treat as data file: create temp file and patch code string
                        with tempfile.NamedTemporaryFile(suffix=temp_ext, delete=False) as temp_file:
                            temp_file.write(file_data)
                            temp_file_path = temp_file.name
                        print(f"[Tool Loop] Created temporary file {temp_file_path} for code execution")
                        # Replace all occurrences of the original file name in the code string with the temp file path
                        patched_code = code_str.replace(orig_file_name, temp_file_path)
                        tool_args[param_name] = patched_code
                        print(f"[Tool Loop] Patched code to use temp file path for {tool_name}")
                except Exception as e:
                    print(f"[Tool Loop] Failed to patch code for code injection: {e}")
        
        return tool_args

    def _init_gemini_llm(self, config, model_config):
        return ChatGoogleGenerativeAI(
            model=model_config["model"],
            temperature=model_config["temperature"],
            google_api_key=os.environ.get(config["api_key_env"]),
            max_tokens=model_config["max_tokens"]
        )

    def _init_groq_llm(self, config, model_config):
        if not os.environ.get(config["api_key_env"]):
            print(f"⚠️ {config['api_key_env']} not found in environment variables. Skipping Groq...")
            return None
        return ChatGroq(
            model=model_config["model"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"]
        )

    def _init_huggingface_llm(self, config, model_config):
        # Convert model to repo_id for HuggingFace
        model_config_with_repo = model_config.copy()
        model_config_with_repo['repo_id'] = model_config['model']
        del model_config_with_repo['model']
        
        allowed_fields = {'repo_id', 'task', 'max_new_tokens', 'do_sample', 'temperature'}
        filtered_config = {k: v for k, v in model_config_with_repo.items() if k in allowed_fields}
        try:
            endpoint = HuggingFaceEndpoint(**filtered_config)
            return ChatHuggingFace(
                llm=endpoint,
                verbose=True,
            )
        except Exception as e:
            if "402" in str(e) or "payment required" in str(e).lower():
                print(f"\u26a0\ufe0f HuggingFace Payment Required (402) error: {e}")
                print("💡 You have exceeded your HuggingFace credits. Skipping HuggingFace LLM initialization.")
                return None
            raise

    def _init_openrouter_llm(self, config, model_config):
        api_key = os.environ.get(config["api_key_env"])
        api_base = os.environ.get(config["api_base_env"])
        if not api_key or not api_base:
            print(f"⚠️ {config['api_key_env']} or {config['api_base_env']} not found in environment variables. Skipping OpenRouter...")
            return None
        return ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base=api_base,
            model_name=model_config["model"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"]
        )

    def _init_mistral_llm(self, config, model_config):
        api_key = os.environ.get(config["api_key_env"])
        if not api_key:
            print(f"⚠️ {config['api_key_env']} not found in environment variables. Skipping Mistral AI...")
            return None
        return ChatMistralAI(
            model=model_config["model"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"]
        )

    def _init_gigachat_llm(self, config, model_config):
        try:
            from langchain_gigachat.chat_models import GigaChat as LC_GigaChat
        except Exception as e:
            print(f"⚠️ langchain-gigachat not installed or failed to import: {e}. Skipping GigaChat...")
            return None
        api_key = os.environ.get(config["api_key_env"]) or os.environ.get("GIGACHAT_API_KEY")
        if not api_key:
            print(f"⚠️ {config['api_key_env']} not found in environment variables. Skipping GigaChat...")
            return None
        scope = os.environ.get(config.get("scope_env", "GIGACHAT_SCOPE"))
        verify_ssl_env = os.environ.get(config.get("verify_ssl_env", "GIGACHAT_VERIFY_SSL"), "false")
        verify_ssl = str(verify_ssl_env).strip().lower() in ("1", "true", "yes", "y")
        # Initialize LangChain GigaChat client
        return LC_GigaChat(
            credentials=api_key,
            model=model_config["model"],
            verify_ssl_certs=verify_ssl is True and True or False,
            scope=scope
        )

    def _ping_llm(self, llm_name: str, llm_type: str, use_tools: bool = False, llm_instance=None) -> bool:
        """
        Test an LLM with a simple "Hello" message to verify it's working, using the unified LLM request method.
        Includes the system message for realistic testing.
        Args:
            llm_name: Name of the LLM for logging purposes
            llm_type: The LLM type string (e.g., 'gemini', 'groq', etc.)
            use_tools: Whether to use tools (default: False)
            llm_instance: If provided, use this LLM instance directly for testing
        Returns:
            bool: True if test passes, False otherwise
        """
        # Use the provided llm_instance if given, otherwise use the lookup logic
        if llm_instance is not None:
            llm = llm_instance
        else:
            if llm_type is None:
                print(f"❌ {llm_name} llm_type not provided - cannot test")
                return False
            try:
                llm, _, _ = self._select_llm(llm_type, use_tools)
            except Exception as e:
                print(f"❌ {llm_name} test failed: {e}")
                return False
        try:
            test_message = [self.sys_msg, HumanMessage(content="What is the main question in the whole Galaxy and all. Max 150 words (250 tokens)")]
            print(f"🧪 Testing {llm_name} with 'Hello' message...")
            start_time = time.time()
            test_response = self._invoke_llm_provider(llm, test_message)
            end_time = time.time()
            if test_response and hasattr(test_response, 'content') and test_response.content:
                print(f"✅ {llm_name} test successful!")
                print(f"   Response time: {end_time - start_time:.2f}s")
                print(f"   Test message details:")
                self._print_message_components(test_message[0], "test_input")
                print(f"   Test response details:")
                self._print_message_components(test_response, "test")
                return True
            else:
                print(f"❌ {llm_name} returned empty response")
                return False
        except Exception as e:
            print(f"❌ {llm_name} test failed: {e}")
            return False

    def _is_duplicate_tool_call(self, tool_name: str, tool_args: dict, called_tools: list) -> bool:
        """
        Check if a tool call is a duplicate based on tool name and vector similarity of arguments.
        
        Args:
            tool_name: Name of the tool
            tool_args: Arguments for the tool
            called_tools: List of previously called tool dictionaries
            
        Returns:
            bool: True if this is a duplicate tool call
        """
        return vector_store_manager.is_duplicate_tool_call(tool_name, tool_args, called_tools, self.tool_calls_similarity_threshold)

    def _add_tool_call_to_history(self, tool_name: str, tool_args: dict, called_tools: list) -> None:
        """
        Add a tool call to the history of called tools.
        
        Args:
            tool_name: Name of the tool
            tool_args: Arguments for the tool
            called_tools: List of previously called tool dictionaries
        """
        vector_store_manager.add_tool_call_to_history(tool_name, tool_args, called_tools)

    def _trim_for_print(self, obj, max_len=None):
        """
        Helper to trim any object (string, dict, etc.) for debug printing only.
        Converts to string, trims to max_len (default: self.MAX_PRINT_LEN), and adds suffix with original length if needed.
        """
        if max_len is None:
            max_len = self.MAX_PRINT_LEN
        s = str(obj)
        orig_len = len(s)
        
        if orig_len > max_len:
            return f"Truncated. Original length: {orig_len}\n{s[:max_len]}"
        return s

    def _format_value_for_print(self, value):
        """
        Smart value formatter that handles JSON serialization, fallback, and trimming.
        ENHANCED: Now uses _deep_trim_dict_max_length() for dicts/lists for consistent base64 and length handling.
        Returns a formatted string ready for printing.
        """
        if isinstance(value, str):
            return self._trim_for_print(value)
        elif isinstance(value, (dict, list)):
            # Use _deep_trim_dict_max_length() for print statements with both base64 and length truncation
            trimmed = self._deep_trim_dict_max_length(value)
            try:
                # Convert back to JSON string for display
                return json.dumps(trimmed, indent=2, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                # Fallback to string representation
                return str(trimmed)
        else:
            return self._trim_for_print(str(value))

    def _print_meaningful_attributes(self, msg, attributes, separator, printed_attrs=None):
        """
        Generic helper to check and print meaningful attributes from a message object.
        
        Args:
            msg: The message object to inspect
            attributes: List of attribute names to check
            separator: String separator to print before each attribute
            printed_attrs: Set of already printed attributes (optional, for tracking)
        """
        if printed_attrs is None:
            printed_attrs = set()
            
        for attr in attributes:
            if hasattr(msg, attr):
                value = getattr(msg, attr)
                if value is not None and value != "" and value != [] and value != {}:
                    print(separator)
                    print(f"  {attr}: {self._format_value_for_print(value)}")
                    printed_attrs.add(attr)
        
        return printed_attrs

    def _print_message_components(self, msg, msg_index):
        """
        Smart, agnostic message component printer that dynamically discovers and prints all relevant attributes.
        Uses introspection, JSON-like handling, and smart filtering for optimal output.
        """
        separator = "------------------------------------------------\n"
        print(separator) 
        print(f"Message {msg_index}:")
        
        # Get message type dynamically
        msg_type = getattr(msg, 'type', 'unknown')
        print(f"  type: {msg_type}")
        
        # Define priority attributes to check first (most important)
        priority_attrs = ['content', 'tool_calls', 'function_call', 'name', 'tool_call_id']
        
        # Define secondary attributes to check if they exist and have meaningful values
        secondary_attrs = ['additional_kwargs', 'response_metadata', 'id', 'timestamp', 'metadata']
        
        # Smart attribute discovery and printing
        printed_attrs = set()
        
        # Check priority attributes first
        printed_attrs = self._print_meaningful_attributes(msg, priority_attrs, separator, printed_attrs)
        
        # Check secondary attributes if they exist and haven't been printed
        self._print_meaningful_attributes(msg, secondary_attrs, separator, printed_attrs)
        
        # Dynamic discovery: check for any other non-private attributes we might have missed
        dynamic_attrs = []
        for attr_name in dir(msg):
            if (not attr_name.startswith('_') and 
                attr_name not in printed_attrs and 
                attr_name not in secondary_attrs and
                attr_name not in ['type'] and  # Already printed
                not callable(getattr(msg, attr_name))):  # Skip methods
                dynamic_attrs.append(attr_name)
        
        # Print any dynamically discovered meaningful attributes
        self._print_meaningful_attributes(msg, dynamic_attrs, separator, printed_attrs)
        
        print(separator)

    def _is_base64_data(self, data: str) -> bool:
        """
        Check if string is likely base64 data using Python's built-in validation.
        Fast and reliable detection for logging purposes.
        """
        if len(data) < 50:  # Too short to be meaningful base64
            return False
        try:
            # Check if it's valid base64 by attempting to decode first 100 chars
            base64.b64decode(data[:100])
            # Additional check for base64 character pattern
            if re.match(r'^[A-Za-z0-9+/=]+$', data):
                return True
        except Exception:
            return False
        return False

    def _deep_trim_dict_base64(self, obj, max_len=None):
        """
        Recursively traverse JSON structure and ONLY truncate base64 data.
        Keep all other text fields intact for complete trace visibility.
        """
        if max_len is None:
            max_len = 100  # Shorter for base64 placeholders
        
        if isinstance(obj, dict):
            return {k: self._deep_trim_dict_base64(v, max_len) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_trim_dict_base64(v, max_len) for v in obj]
        elif isinstance(obj, str):
            # ONLY check for base64, leave everything else intact
            if self._is_base64_data(obj):
                return f"[BASE64_DATA] Length: {len(obj)} chars"
            return obj  # ← Keep all non-base64 text intact
        else:
            return obj

    def _deep_trim_dict_max_length(self, obj, max_len=None):
        """
        First truncate base64 data, then check remaining text for max length.
        This ensures base64 is always handled properly before length checks.
        """
        if max_len is None:
            max_len = self.MAX_PRINT_LEN
        
        # Step 1: Handle base64 first
        obj = self._deep_trim_dict_base64(obj)
        
        # Step 2: Now check remaining text for max length
        if isinstance(obj, dict):
            return {k: self._deep_trim_dict_max_length(v, max_len) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_trim_dict_max_length(v, max_len) for v in obj]
        elif isinstance(obj, str):
            # Base64 is already handled, now check length
            if len(obj) > max_len:
                return f"Truncated. Original length: {len(obj)}\n{obj[:max_len]}"
            return obj
        else:
            return obj

    def _print_tool_result(self, tool_name, tool_result):
        """
        Print tool results in a readable format with deep recursive trimming for all dicts/lists.
        For dict/list results, deeply trim all string fields. For other types, use _trim_for_print.
        """
        if isinstance(tool_result, (dict, list)):
            trimmed = self._deep_trim_dict_max_length(tool_result)
            print(f"[Tool Loop] Tool result for '{tool_name}': {trimmed}")
        else:
            print(f"[Tool Loop] Tool result for '{tool_name}': {self._trim_for_print(tool_result)}")
        print()

    def _extract_main_text_from_tool_result(self, tool_result):
        """
        Extract the main text from a tool result dict (e.g., wiki_results, web_results, arxiv_results, etc.).
        """
        if isinstance(tool_result, dict):
            for key in ("wiki_results", "web_results", "arxiv_results", "result", "text", "content"):
                if key in tool_result and isinstance(tool_result[key], str):
                    return tool_result[key]
            # Fallback: join all string values
            return " ".join(str(v) for v in tool_result.values() if isinstance(v, str))
        return str(tool_result)

    def _retry_with_final_answer_reminder(self, messages, use_tools, llm_type):
        """
        Injects a final answer reminder, retries the LLM request, and extracts the answer.
        Returns (answer, response)
        """
        # Find the original question from the message history
        original_question = None
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'human':
                original_question = msg.content
                break
        
        # Build the prompt message (slim, direct)
        prompt = (
            "TASK: Extract the FINAL answer from the given LLM response. "
            "If a **question** is present, extract the most likely FINAL ANSWER according to the system prompt's answer formatting rules. "
            "Return only the most likely final answer, formatted exactly as required by the system prompt.\n\n"
            "FOCUS: Focus on the most relevant facts, numbers, and names, related to the question if present.\n\n"
            "PURPOSE: Extract the FINAL ANSWER per the system prompt.\n\n"
            "INSTRUCTIONS: Do not use tools.\n\n"
        )
        if original_question:
            prompt += f"QUESTION: {original_question}\n\n"
        prompt += "RESPONSE TO ANALYZE:\nAnalyze the previous response and provide your FINAL ANSWER."
        
        # Inject the message into the queue
        messages.append(HumanMessage(content=prompt))
        
        # Make the LLM call and extract the answer
        response = self._make_llm_request(messages, use_tools=use_tools, llm_type=llm_type)
        answer = self._extract_final_answer(response)
        return answer, response

    def _get_reminder_prompt(
        self,
        reminder_type: str,
        messages=None,
        tools=None,
        tool_results_history=None,
        tool_name=None,
        count=None,
        tool_args=None,
        question=None
    ) -> str:
        """
        Get standardized reminder prompts based on type. Extracts tool_names, tool_count, and original_question as needed.
        
        Args:
            reminder_type: Type of reminder needed
            messages: Message history (for extracting question)
            tools: List of tool objects (for tool names)
            tool_results_history: List of tool results (for count)
            tool_name: Name of the tool (for tool-specific reminders)
            count: Usage count (for tool-specific reminders)
            tool_args: Arguments for the tool (for duplicate reminders)
            question: Optional question override
            
        Returns:
            str: The reminder prompt
        """
        # Extract tool_names if needed
        tool_names = None
        if tools is not None:
            tool_names = ', '.join([self._get_tool_name(tool) for tool in tools])
            
        # Extract tool_count if needed
        tool_count = None
        if tool_results_history is not None:
            tool_count = len(tool_results_history)
            
        # Extract original_question if needed
        original_question = None
        if messages is not None:
            for msg in messages:
                if hasattr(msg, 'type') and msg.type == 'human':
                    original_question = msg.content
                    break
        if not original_question:
            original_question = question or '[Original question not found]'
            
        reminders = {
            "final_answer_prompt": (
                    "Analyse existing tool results, then provide your FINAL ANSWER.\n"
                + (
                    "Use VARIOUS tools to gather missing information, then provide your FINAL ANSWER.\n"
                    f"Available tools include: {tool_names or 'various tools'}.\n"
                    if not tool_count or tool_count == 0 else ""
                  )
                + (
                    f"\n\nIMPORTANT: You have gathered information from {tool_count} tool calls.\n"
                    "The tool results are available in the conversation.\n"
                    "Carefully analyze tool results and provide your FINAL ANSWER to the ORIGINAL QUESTION.\n"
                    "Follow the system prompt.\n"
                    "Do not call any more tools - analyze the existing results and provide your answer now.\n"
                    if tool_count and tool_count > 0 else ""
                  )
                + "\n\nPlease answer the following question in the required format:\n\n"
                + f"ORIGINAL QUESTION:\n{original_question}\n\n"
                + "Your answer must start with 'FINAL ANSWER:' and follow the system prompt.\n"
            ),
            "tool_usage_issue": (
                "Call a DIFFERENT TOOL.\n"
                + (
                    f"You have already called '{tool_name or 'this tool'}'"
                    + (f" {count} times" if count is not None else "")
                    + (f" with arguments {tool_args}" if tool_args is not None else "")
                    + ". "
                    if (tool_name or count is not None or tool_args is not None) else ""
                )
                + "Do not call the tools repeately with the same arguments.\n"
                + "Consider any results you have.\n"
                + f"ORIGINAL QUESTION:\n{original_question}\n\n"
                + "Provide your FINAL ANSWER based on the information you have or call OTHER TOOLS.\n"
            ),
        }
        return reminders.get(reminder_type, "Please analyse the tool results and provide your FINAL ANSWER.")

    def _create_simple_chunk_prompt(self, messages, chunk_results, chunk_num, total_chunks):
        """Create a simple prompt for processing a chunk."""
        # Find original question
        original_question = ""
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'human':
                original_question = msg.content
                break
        
        # Determine if this is tool results or general content
        is_tool_results = any('tool' in str(result).lower() or 'result' in str(result).lower() for result in chunk_results)
        
        if is_tool_results:
            prompt = f"Question: {original_question}\n\nTool Results (Part {chunk_num}/{total_chunks}):\n"
            for i, result in enumerate(chunk_results, 1):
                prompt += f"{i}. {result}\n\n"
        else:
            prompt = f"Question: {original_question}\n\nContent Analysis (Part {chunk_num}/{total_chunks}):\n"
            for i, result in enumerate(chunk_results, 1):
                prompt += f"{i}. {result}\n\n"
        
        if chunk_num < total_chunks:
            prompt += "Analyze these results and provide key findings."
        else:
            prompt += "Provide your FINAL ANSWER based on all content, when you receive it, following the system prompt format."
        
        return prompt

    def _is_token_limit_error(self, error, llm_type="unknown") -> bool:
        """
        Check if the error is a token limit error or router error using vector similarity.
        
        Args:
            error: The exception object
            llm_type: Type of LLM for specific error patterns
            
        Returns:
            bool: True if it's a token limit error or router error
        """
        error_str = str(error).lower()
        
        # Token limit and router error patterns for vector similarity
        error_patterns = [
            "Error code: 413 - {'error': {'message': 'Request too large for model `qwen-qwq-32b` in organization `org_01jyfgv54ge5ste08j9248st66` service tier `on_demand` on tokens per minute (TPM): Limit 6000, Requested 9681, please reduce your message size and try again. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}",
            "500 Server Error: Internal Server Error for url: https://router.huggingface.co/hyperbolic/v1/chat/completions (Request ID: Root=1-6861ed33-7dd4232d49939c6f65f6e83d;164205eb-e591-4b20-8b35-5745a13f05aa)",
            "Error code: 429 - {'error': {'message': 'Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '50', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1756771200000'}, 'provider_name': None}}, 'user_id': 'user_2zGr0yIHMzRxIJYcW8N0my40LIs'}"
        ]
        
        # Direct substring checks for efficiency
        if any(term in error_str for term in ["413", "429", "token", "limit", "tokens per minute", "truncated", "tpm", "router.huggingface.co", "402", "payment required", "rate limit", "rate_limit"]):
            return True
        
        # Check if error matches any pattern using vector similarity
        for pattern in error_patterns:
            if self._vector_answers_match(error_str, pattern):
                return True
        
        return False

    def _get_token_limit(self, provider: str) -> int:
        """
        Get the token limit for a given provider, using the active model config, with fallback to default.
        """
        try:
            if provider in self.active_model_config:
                return self.active_model_config[provider].get("token_limit", self.LLM_CONFIG["default"]["token_limit"])
            else:
                return self.LLM_CONFIG["default"]["token_limit"]
        except Exception:
            return self.LLM_CONFIG["default"]["token_limit"]

    def _provider_supports_tools(self, llm_type: str) -> bool:
        """
        Returns True if the provider supports tool-calling, based on LLM_CONFIG.
        """
        config = self.LLM_CONFIG.get(llm_type, {})
        return config.get("tool_support", False)

    def _handle_llm_error(self, e, llm_name, llm_type, phase, **kwargs):
        """
        Centralized error handler for LLM errors (init, runtime, tool loop, request, etc.).
        For phase="init": returns (ok: bool, error_str: str).
        For phase="runtime"/"tool_loop"/"request": returns (handled: bool, result: Optional[Any]).
        All logging and comments are preserved from original call sites.
        """
        # --- INIT PHASE ---
        if phase == "init":
            if self._is_token_limit_error(e, llm_type) or "429" in str(e):
                print(f"⛔ {llm_name} initialization failed due to rate limit/quota (429) [{phase}]: {e}")
                return False, str(e)
            raise
        # --- RUNTIME/TOOL LOOP PHASE ---
        # Enhanced Groq token limit error handling
        if llm_type == "groq" and self._is_token_limit_error(e):
            print(f"⚠️ Groq token limit error detected: {e}")
            return True, self._handle_groq_token_limit_error(kwargs.get('messages'), kwargs.get('llm'), llm_name, e)
        # Special handling for HuggingFace router errors
        if llm_type == "huggingface" and self._is_token_limit_error(e):
            # Check if chunking is enabled for HuggingFace
            config = self.LLM_CONFIG.get(llm_type, {})
            enable_chunking = config.get("enable_chunking", True)
            
            if enable_chunking:
                print(f"⚠️ HuggingFace router error detected, applying chunking: {e}")
                return True, self._handle_token_limit_error(kwargs.get('messages'), kwargs.get('llm'), llm_name, e, llm_type)
            else:
                print(f"⚠️ HuggingFace router error detected, but chunking disabled: {e}")
                raise e
        if llm_type == "huggingface" and "500 Server Error" in str(e) and "router.huggingface.co" in str(e):
            error_msg = f"HuggingFace router service error (500): {e}"
            print(f"⚠️ {error_msg}")
            print("💡 This is a known issue with HuggingFace's router service. Consider using Google Gemini or Groq instead.")
            raise Exception(error_msg)
        if llm_type == "huggingface" and "timeout" in str(e).lower():
            error_msg = f"HuggingFace timeout error: {e}"
            print(f"⚠️ {error_msg}")
            print("💡 HuggingFace models may be slow or overloaded. Consider using Google Gemini or Groq instead.")
            raise Exception(error_msg)
        # Special handling for Groq network errors
        if llm_type == "groq" and ("no healthy upstream" in str(e).lower() or "network" in str(e).lower() or "connection" in str(e).lower()):
            error_msg = f"Groq network connectivity error: {e}"
            print(f"⚠️ {error_msg}")
            print("💡 This is a network connectivity issue with Groq's servers. The service may be temporarily unavailable.")
            raise Exception(error_msg)
        # Enhanced token limit error handling for all LLMs (tool loop context)
        if phase in ("tool_loop", "runtime", "request") and self._is_token_limit_error(e, llm_type):
            # Check if chunking is enabled for this provider
            config = self.LLM_CONFIG.get(llm_type, {})
            enable_chunking = config.get("enable_chunking", True)
            
            if enable_chunking:
                print(f"[Tool Loop] Token limit error detected for {llm_type} in tool calling loop")
                _, llm_name, _ = self._select_llm(llm_type, True)
                return True, self._handle_token_limit_error(kwargs.get('messages'), kwargs.get('llm'), llm_name, e, llm_type)
            else:
                print(f"[Tool Loop] Token limit error detected for {llm_type}, but chunking disabled")
                raise e
        # Handle HuggingFace router errors with chunking (tool loop context)
        if phase in ("tool_loop", "runtime", "request") and llm_type == "huggingface" and self._is_token_limit_error(e):
            # Check if chunking is enabled for HuggingFace
            config = self.LLM_CONFIG.get(llm_type, {})
            enable_chunking = config.get("enable_chunking", True)
            
            if enable_chunking:
                print(f"⚠️ HuggingFace router error detected, applying chunking: {e}")
                return True, self._handle_token_limit_error(kwargs.get('messages'), kwargs.get('llm'), llm_name, e, llm_type)
            else:
                print(f"⚠️ HuggingFace router error detected, but chunking disabled: {e}")
                raise e
        # Check for general token limit errors specifically (tool loop context)
        if phase in ("tool_loop", "runtime", "request") and ("413" in str(e) or "token" in str(e).lower() or "limit" in str(e).lower()):
            print(f"[Tool Loop] Token limit error detected. Forcing final answer with available information.")
            tool_results_history = kwargs.get('tool_results_history')
            if tool_results_history:
                return True, self._force_final_answer(kwargs.get('messages'), tool_results_history, kwargs.get('llm'))
            else:
                return True, AIMessage(content=f"Error: Token limit exceeded for {llm_type} LLM. Cannot complete reasoning.")
        # Generic fallback for tool loop
        if phase in ("tool_loop", "runtime", "request"):
            return True, AIMessage(content=f"Error during LLM processing: {str(e)}")
        # Fallback: not handled here
        return False, None

    def _get_available_models(self) -> Dict:
        """
        Get list of available models and their status.
        
        Returns:
            Dict: Available models with their status
        """
        available_models = {}
        for llm_type, config in self.LLM_CONFIG.items():
            if llm_type == "default":
                continue
            available_models[llm_type] = {
                "name": config.get("name", llm_type),
                "models": config.get("models", []),
                "tool_support": config.get("tool_support", False),
                "max_history": config.get("max_history", self.LLM_CONFIG["default"]["max_history"])
            }
        return available_models

    def _get_tool_support_status(self) -> Dict:
        """
        Get tool support status for each LLM type.
        
        Returns:
            Dict: Tool support status for each LLM
        """
        tool_status = {}
        for llm_type, config in self.LLM_CONFIG.items():
            if llm_type == "default":
                continue
            tool_status[llm_type] = {
                "tool_support": config.get("tool_support", False),
                "force_tools": config.get("force_tools", False)
            }
        return tool_status

    # ===== TRACING SYSTEM METHODS =====
    
    def _trace_init_question(self, question: str, file_data: str = None, file_name: str = None):
        """
        Initialize trace for a new question.
        
        Args:
            question: The question being processed
            file_data: Base64 file data if attached
            file_name: Name of attached file
        """
        self.question_trace = {
            "question": question,
            "file_name": file_name if file_name is not None else "N/A",
            "file_size": len(file_data) if file_data else 0,
            "start_time": datetime.datetime.now().isoformat(),
            "llm_traces": {},
            "logs": [],
            "final_result": None,
            "per_llm_stdout": []  # Array to store stdout for each LLM attempt
        }
        self.current_llm_call_id = None
        self.current_llm_stdout_buffer = None  # Buffer for current LLM's stdout
        print(f"🔍 Initialized trace for question: {question[:100]}...")

    def _get_llm_name(self, llm_type: str) -> str:
        """
        Get the LLM name for a given LLM type.
        
        Args:
            llm_type: Type of LLM
            
        Returns:
            str: LLM name (model ID if available, otherwise provider name)
        """
        model_id = ""
        if llm_type in self.active_model_config:
            model_id = self.active_model_config[llm_type].get("model", "")
        
        return f"{model_id}" if model_id else self.LLM_CONFIG[llm_type]["name"]

    def _trace_start_llm(self, llm_type: str) -> str:
        """
        Start a new LLM call trace and return call_id.
        
        Args:
            llm_type: Type of LLM being called
            
        Returns:
            str: Unique call ID for this LLM call
        """
        if not self.question_trace:
            return None
            
        call_id = f"{llm_type}_call_{len(self.question_trace['llm_traces'].get(llm_type, [])) + 1}"
        self.current_llm_call_id = call_id
        
        # Initialize LLM trace if not exists
        if llm_type not in self.question_trace["llm_traces"]:
            self.question_trace["llm_traces"][llm_type] = []
        
        # Create descriptive LLM name with model info
        llm_name = self._get_llm_name(llm_type)
        
        # Create new call trace
        call_trace = {
            "call_id": call_id,
            "llm_name": llm_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "input": {},
            "output": {},
            "tool_executions": [],
            "tool_loop_data": [],
            "execution_time": None,
            "total_tokens": None,
            "error": None
        }
        
        self.question_trace["llm_traces"][llm_type].append(call_trace)
        
        # Start new stdout buffer for this LLM attempt
        self.current_llm_stdout_buffer = StringIO()
        
        print(f"🤖 Started LLM trace: {call_id} ({llm_type})")
        return call_id

    def _trace_capture_llm_stdout(self, llm_type: str, call_id: str):
        """
        Capture stdout for the current LLM attempt and add it to the trace.
        This should be called when an LLM attempt is complete.
        
        Args:
            llm_type: Type of LLM that just completed
            call_id: Call ID of the completed LLM attempt
        """
        if not self.question_trace or not self.current_llm_stdout_buffer:
            return
            
        # Get the captured stdout
        stdout_content = self.current_llm_stdout_buffer.getvalue()
        
        # Create descriptive LLM name with model info
        llm_name = self._get_llm_name(llm_type)
        
        # Add to per-LLM stdout array
        llm_stdout_entry = {
            "llm_type": llm_type,
            "llm_name": llm_name,
            "call_id": call_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "stdout": stdout_content
        }
        
        self.question_trace["per_llm_stdout"].append(llm_stdout_entry)
        
        # Clear the buffer for next LLM
        self.current_llm_stdout_buffer = None
        
        print(f"📝 Captured stdout for {llm_type} ({call_id}): {len(stdout_content)} chars")

    def _trace_add_llm_call_input(self, llm_type: str, call_id: str, messages: List, use_tools: bool):
        """
        Add input data to current LLM call trace.
        
        Args:
            llm_type: Type of LLM
            call_id: Call ID
            messages: Input messages
            use_tools: Whether tools are being used
        """
        if not self.question_trace or not call_id:
            return
            
        # Find the call trace
        for call_trace in self.question_trace["llm_traces"].get(llm_type, []):
            if call_trace["call_id"] == call_id:
                # Use _deep_trim_dict_base64 to preserve all text data in trace JSON
                trimmed_messages = self._deep_trim_dict_base64(messages)
                call_trace["input"] = {
                    "messages": trimmed_messages,
                    "use_tools": use_tools,
                    "llm_type": llm_type
                }
                break

    def _trace_add_llm_call_output(self, llm_type: str, call_id: str, response: Any, execution_time: float):
        """
        Add output data to current LLM call trace.
        
        Args:
            llm_type: Type of LLM
            call_id: Call ID
            response: LLM response
            execution_time: Time taken for the call
        """
        if not self.question_trace or not call_id:
            return
            
        # Find the call trace
        for call_trace in self.question_trace["llm_traces"].get(llm_type, []):
            if call_trace["call_id"] == call_id:
                # Use _deep_trim_dict_base64 to preserve all text data in trace JSON
                trimmed_response = self._deep_trim_dict_base64(response)
                call_trace["output"] = {
                    "content": getattr(response, 'content', None),
                    "tool_calls": getattr(response, 'tool_calls', None),
                    "response_metadata": getattr(response, 'response_metadata', None),
                    "raw_response": trimmed_response
                }
                call_trace["execution_time"] = execution_time
                
                # Extract and accumulate token usage
                token_data = self._extract_token_usage(response, llm_type)
                if token_data:
                    # Initialize token usage if not exists
                    if "token_usage" not in call_trace:
                        call_trace["token_usage"] = {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "call_count": 0,
                            "calls": []
                        }
                    
                    # Add current call data
                    call_data = {
                        "call_id": call_id,
                        "timestamp": datetime.datetime.now().isoformat(),
                        **token_data
                    }
                    call_trace["token_usage"]["calls"].append(call_data)
                    
                    # Accumulate totals
                    call_trace["token_usage"]["prompt_tokens"] += token_data.get("prompt_tokens", 0)
                    call_trace["token_usage"]["completion_tokens"] += token_data.get("completion_tokens", 0)
                    call_trace["token_usage"]["total_tokens"] += token_data.get("total_tokens", 0)
                    call_trace["token_usage"]["call_count"] += 1
                
                # Fallback to estimated tokens if no token data available
                if not token_data or not any([token_data.get("prompt_tokens"), token_data.get("completion_tokens"), token_data.get("total_tokens")]):
                    call_trace["total_tokens"] = self._estimate_tokens(str(response)) if response else None
                
                break

    def _add_tool_execution_trace(self, llm_type: str, call_id: str, tool_name: str, tool_args: dict, tool_result: str, execution_time: float):
        """
        Add tool execution trace to current LLM call.
        
        Args:
            llm_type: Type of LLM
            call_id: Call ID
            tool_name: Name of the tool
            tool_args: Tool arguments
            tool_result: Tool result
            execution_time: Time taken for tool execution
        """
        if not self.question_trace or not call_id:
            return
            
        # Find the call trace
        for call_trace in self.question_trace["llm_traces"].get(llm_type, []):
            if call_trace["call_id"] == call_id:
                # Use _deep_trim_dict_base64 to preserve all text data in trace JSON
                trimmed_args = self._deep_trim_dict_base64(tool_args)
                trimmed_result = self._deep_trim_dict_base64(tool_result)
                
                tool_execution = {
                    "tool_name": tool_name,
                    "args": trimmed_args,
                    "result": trimmed_result,
                    "execution_time": execution_time,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                call_trace["tool_executions"].append(tool_execution)
                break

    def _add_tool_loop_data(self, llm_type: str, call_id: str, step: int, tool_calls: List, consecutive_no_progress: int):
        """
        Add tool loop data to current LLM call trace.
        
        Args:
            llm_type: Type of LLM
            call_id: Call ID
            step: Current step number
            tool_calls: List of tool calls detected
            consecutive_no_progress: Number of consecutive steps without progress
        """
        if not self.question_trace or not call_id:
            return
            
        # Find the call trace
        for call_trace in self.question_trace["llm_traces"].get(llm_type, []):
            if call_trace["call_id"] == call_id:
                loop_data = {
                    "step": step,
                    "tool_calls_detected": len(tool_calls) if tool_calls else 0,
                    "consecutive_no_progress": consecutive_no_progress,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                call_trace["tool_loop_data"].append(loop_data)
                break

    def _trace_add_llm_error(self, llm_type: str, call_id: str, error: Exception):
        """
        Add error information to current LLM call trace.
        
        Args:
            llm_type: Type of LLM
            call_id: Call ID
            error: Exception that occurred
        """
        if not self.question_trace or not call_id:
            return
            
        # Find the call trace
        for call_trace in self.question_trace["llm_traces"].get(llm_type, []):
            if call_trace["call_id"] == call_id:
                call_trace["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "timestamp": datetime.datetime.now().isoformat()
                }
                break

    def _trace_finalize_question(self, final_result: dict):
        """
        Finalize the question trace with final results.
        
        Args:
            final_result: Final result dictionary
        """
        if not self.question_trace:
            return
            
        self.question_trace["final_result"] = final_result
        self.question_trace["end_time"] = datetime.datetime.now().isoformat()
        
        # Calculate total execution time
        start_time = datetime.datetime.fromisoformat(self.question_trace["start_time"])
        end_time = datetime.datetime.fromisoformat(self.question_trace["end_time"])
        total_time = (end_time - start_time).total_seconds()
        self.question_trace["total_execution_time"] = total_time
        
        # Calculate total tokens across all LLM calls
        total_tokens = 0
        for llm_type, calls in self.question_trace["llm_traces"].items():
            for call in calls:
                if "token_usage" in call:
                    total_tokens += call["token_usage"].get("total_tokens", 0)
        
        self.question_trace["tokens_total"] = total_tokens
        
        # Capture any remaining stdout from current LLM attempt
        if hasattr(self, 'current_llm_stdout_buffer') and self.current_llm_stdout_buffer:
            self._trace_capture_llm_stdout(self.current_llm_type, self.current_llm_call_id)
        
        # Capture all debug output as comprehensive text
        debug_output = self._capture_all_debug_output()
        self.question_trace["debug_output"] = debug_output
        
        print(f"📊 Question trace finalized. Total execution time: {total_time:.2f}s")
        print(f"📝 Captured stdout for {len(self.question_trace.get('per_llm_stdout', []))} LLM attempts")
        print(f"🔢 Total tokens used: {total_tokens}")
        print(f"📄 Debug output captured: {len(debug_output)} characters")

    def _capture_all_debug_output(self) -> str:
        """
        Capture all debug output as comprehensive text, including:
        - All logs from the question trace
        - All LLM traces with their details
        - All tool executions
        - All stdout captures
        - Error information
        - Performance metrics
        
        Returns:
            str: Comprehensive debug output as text
        """
        if not self.question_trace:
            return "No trace available"
        
        debug_lines = []
        debug_lines.append("=" * 80)
        debug_lines.append("COMPREHENSIVE DEBUG OUTPUT")
        debug_lines.append("=" * 80)
        
        # Question metadata
        debug_lines.append(f"Question: {self.question_trace.get('question', 'N/A')}")
        debug_lines.append(f"File: {self.question_trace.get('file_name', 'N/A')}")
        debug_lines.append(f"File Size: {self.question_trace.get('file_size', 0)} chars")
        debug_lines.append(f"Start Time: {self.question_trace.get('start_time', 'N/A')}")
        debug_lines.append(f"End Time: {self.question_trace.get('end_time', 'N/A')}")
        debug_lines.append(f"Total Execution Time: {self.question_trace.get('total_execution_time', 0):.2f}s")
        debug_lines.append(f"Total Tokens: {self.question_trace.get('tokens_total', 0)}")
        debug_lines.append("")
        
        # Final result
        debug_lines.append("-" * 40)
        final_result = self.question_trace.get('final_result', {})
        if final_result:
            debug_lines.append("FINAL RESULT:")
            debug_lines.append("-" * 40)
            for key, value in final_result.items():
                debug_lines.append(f"{key}: {value}")
            debug_lines.append("")
        
        
        # Per-LLM stdout captures
        debug_lines.append("-" * 40)
        per_llm_stdout = self.question_trace.get('per_llm_stdout', [])
        if per_llm_stdout:
            debug_lines.append("PER-LLM STDOUT CAPTURES:")
            for i, stdout_entry in enumerate(per_llm_stdout, 1):
                debug_lines.append("-" * 40)
                debug_lines.append(f"LLM Attempt {i}:")
                debug_lines.append("-" * 40)
                debug_lines.append(f"  LLM Type: {stdout_entry.get('llm_type', 'N/A')}")
                debug_lines.append(f"  LLM Name: {stdout_entry.get('llm_name', 'N/A')}")
                debug_lines.append(f"  Call ID: {stdout_entry.get('call_id', 'N/A')}")
                debug_lines.append(f"  Timestamp: {stdout_entry.get('timestamp', 'N/A')}")
                stdout_content = stdout_entry.get('stdout', '')
                debug_lines.append(f"  Stdout Length: {len(stdout_content)} characters")
                if stdout_content:
                    debug_lines.append(f"  Stdout: {stdout_content}")
                    # CAN BE SHORTENED debug_lines.append(f"  Stdout Preview: {stdout_content[:self.MAX_PRINT_LEN]}...")
                debug_lines.append("")
        
        # All logs
        debug_lines.append("-" * 40)
        logs = self.question_trace.get('logs', [])
        if logs:
            debug_lines.append("GENERAL LOGS:")
            debug_lines.append("-" * 40)
            for log in logs:
                timestamp = log.get('timestamp', 'N/A')
                message = log.get('message', 'N/A')
                function = log.get('function', 'N/A')
                debug_lines.append(f"[{timestamp}] [{function}] {message}")
            debug_lines.append("")
        
        # LLM traces
        debug_lines.append("-" * 40)
        llm_traces = self.question_trace.get('llm_traces', {})
        if llm_traces:
            debug_lines.append("LLM TRACES:")
            debug_lines.append("-" * 40)
            for llm_type, calls in llm_traces.items():
                debug_lines.append(f"LLM Type: {llm_type}")
                debug_lines.append("-" * 30)
                for i, call in enumerate(calls, 1):
                    debug_lines.append(f"  Call {i}: {call.get('call_id', 'N/A')}")
                    debug_lines.append(f"    LLM Name: {call.get('llm_name', 'N/A')}")
                    debug_lines.append(f"    Timestamp: {call.get('timestamp', 'N/A')}")
                    debug_lines.append(f"    Execution Time: {call.get('execution_time', 'N/A')}")
                    
                    # Input details
                    input_data = call.get('input', {})
                    if input_data:
                        debug_lines.append(f"    Input Messages: {len(input_data.get('messages', []))}")
                        debug_lines.append(f"    Use Tools: {input_data.get('use_tools', False)}")
                    
                    # Output details
                    output_data = call.get('output', {})
                    if output_data:
                        content = output_data.get('content', '')
                        if content:
                            debug_lines.append(f"    Output Content: {content[:200]}...")
                        tool_calls = output_data.get('tool_calls', [])
                        if tool_calls:
                            debug_lines.append(f"    Tool Calls: {len(tool_calls)}")
                    
                    # Token usage
                    token_usage = call.get('token_usage', {})
                    if token_usage:
                        debug_lines.append(f"    Tokens: {token_usage.get('total_tokens', 0)}")
                    
                    # Tool executions
                    tool_executions = call.get('tool_executions', [])
                    if tool_executions:
                        debug_lines.append(f"    Tool Executions: {len(tool_executions)}")
                        for j, tool_exec in enumerate(tool_executions, 1):
                            tool_name = tool_exec.get('tool_name', 'N/A')
                            exec_time = tool_exec.get('execution_time', 0)
                            debug_lines.append(f"      Tool {j}: {tool_name} ({exec_time:.2f}s)")
                    
                    # Tool loop data
                    tool_loop_data = call.get('tool_loop_data', [])
                    if tool_loop_data:
                        debug_lines.append(f"    Tool Loop Steps: {len(tool_loop_data)}")
                    
                    # Error information
                    error = call.get('error', {})
                    if error:
                        debug_lines.append(f"    Error: {error.get('type', 'N/A')} - {error.get('message', 'N/A')}")
                    
                    # Call-specific logs
                    call_logs = call.get('logs', [])
                    if call_logs:
                        debug_lines.append(f"    Logs: {len(call_logs)} entries")
                    
                    debug_lines.append("")
                debug_lines.append("")
        
        debug_lines.append("=" * 80)
        debug_lines.append("END DEBUG OUTPUT")
        debug_lines.append("=" * 80)
        
        return "\n".join(debug_lines)

    def _trace_get_full(self) -> dict:
        """
        Get the complete trace for the current question.
        
        Returns:
            dict: Complete trace data or None if no trace exists
        """
        if not self.question_trace:
            return None
            
        # Serialize the trace data to ensure it's JSON-serializable
        return self._serialize_trace_data(self.question_trace)

    def _serialize_trace_data(self, obj):
        """
        Recursively serialize trace data, converting LangChain message objects and other
        non-JSON-serializable objects to dictionaries.
        
        Args:
            obj: Object to serialize
            
        Returns:
            Serialized object that can be JSON serialized
        """
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, list):
            return [self._serialize_trace_data(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._serialize_trace_data(value) for key, value in obj.items()}
        elif hasattr(obj, 'type') and hasattr(obj, 'content'):
            # This is likely a LangChain message object
            return {
                "type": getattr(obj, 'type', 'unknown'),
                "content": self._serialize_trace_data(getattr(obj, 'content', '')),
                "additional_kwargs": self._serialize_trace_data(getattr(obj, 'additional_kwargs', {})),
                "response_metadata": self._serialize_trace_data(getattr(obj, 'response_metadata', {})),
                "tool_calls": self._serialize_trace_data(getattr(obj, 'tool_calls', [])),
                "function_call": self._serialize_trace_data(getattr(obj, 'function_call', None)),
                "name": getattr(obj, 'name', None),
                "tool_call_id": getattr(obj, 'tool_call_id', None),
                "id": getattr(obj, 'id', None),
                "timestamp": getattr(obj, 'timestamp', None),
                "metadata": self._serialize_trace_data(getattr(obj, 'metadata', {}))
            }
        else:
            # For any other object, try to convert to string
            try:
                return str(obj)
            except:
                return f"<non-serializable object of type {type(obj).__name__}>"

    def _trace_clear(self):
        """
        Clear the current question trace.
        """
        self.question_trace = None
        self.current_llm_call_id = None
        self.current_llm_stdout_buffer = None

    def _add_log_to_context(self, message: str, function: str):
        """
        Add log to the appropriate context based on current execution.
        
        Args:
            message: The log message
            function: The function name that generated the log
        """
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "message": message,
            "function": function
        }
        
        if not self.question_trace:
            return
        
        context = getattr(self, '_current_trace_context', None)
        
        if context == "llm_call" and self.current_llm_call_id:
            # Add to current LLM call
            self._add_log_to_llm_call(log_entry)
        elif context == "tool_execution":
            # Add to current tool execution
            self._add_log_to_tool_execution(log_entry)
        elif context == "tool_loop":
            # Add to current tool loop step
            self._add_log_to_tool_loop(log_entry)
        elif context == "final_answer":
            # Add to current LLM call's final answer enforcement
            self._add_log_to_llm_call(log_entry)
        else:
            # Add to question-level logs
            self.question_trace.setdefault("logs", []).append(log_entry)

    def _add_log_to_llm_call(self, log_entry: dict):
        """
        Add log entry to the current LLM call.
        
        Args:
            log_entry: The log entry to add
        """
        if not self.question_trace or not self.current_llm_call_id:
            return
            
        llm_type = self.current_llm_type
        call_id = self.current_llm_call_id
        
        # Find the call trace
        for call_trace in self.question_trace["llm_traces"].get(llm_type, []):
            if call_trace["call_id"] == call_id:
                # Check if this is a final answer enforcement log
                if log_entry.get("function") == "_force_final_answer":
                    call_trace.setdefault("final_answer_enforcement", []).append(log_entry)
                else:
                    call_trace.setdefault("logs", []).append(log_entry)
                break

    def _add_log_to_tool_execution(self, log_entry: dict):
        """
        Add log entry to the current tool execution.
        
        Args:
            log_entry: The log entry to add
        """
        if not self.question_trace or not self.current_llm_call_id:
            return
            
        llm_type = self.current_llm_type
        call_id = self.current_llm_call_id
        
        # Find the call trace and add to the last tool execution
        for call_trace in self.question_trace["llm_traces"].get(llm_type, []):
            if call_trace["call_id"] == call_id:
                tool_executions = call_trace.get("tool_executions", [])
                if tool_executions:
                    tool_executions[-1].setdefault("logs", []).append(log_entry)
                break

    def _add_log_to_tool_loop(self, log_entry: dict):
        """
        Add log entry to the current tool loop step.
        
        Args:
            log_entry: The log entry to add
        """
        if not self.question_trace or not self.current_llm_call_id:
            return
            
        llm_type = self.current_llm_type
        call_id = self.current_llm_call_id
        
        # Find the call trace and add to the last tool loop step
        for call_trace in self.question_trace["llm_traces"].get(llm_type, []):
            if call_trace["call_id"] == call_id:
                tool_loop_data = call_trace.get("tool_loop_data", [])
                if tool_loop_data:
                    tool_loop_data[-1].setdefault("logs", []).append(log_entry)
                break

    def _extract_token_usage(self, response, llm_type: str) -> dict:
        """
        Extract token usage data from LLM response.
        
        Args:
            response: The LLM response object
            llm_type: Type of LLM provider
            
        Returns:
            dict: Token usage data with available fields
        """
        token_data = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "finish_reason": None,
            "system_fingerprint": None,
            "input_token_details": {},
            "output_token_details": {}
        }
        
        try:
            # Extract from response_metadata (OpenRouter, HuggingFace)
            if hasattr(response, 'response_metadata') and response.response_metadata:
                metadata = response.response_metadata
                if 'token_usage' in metadata:
                    usage = metadata['token_usage']
                    token_data.update({
                        "prompt_tokens": usage.get('prompt_tokens'),
                        "completion_tokens": usage.get('completion_tokens'),
                        "total_tokens": usage.get('total_tokens')
                    })
                
                token_data["finish_reason"] = metadata.get('finish_reason')
                token_data["system_fingerprint"] = metadata.get('system_fingerprint')
            
            # Extract from usage_metadata (Groq, some others)
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                token_data.update({
                    "prompt_tokens": usage.get('input_tokens'),
                    "completion_tokens": usage.get('output_tokens'),
                    "total_tokens": usage.get('total_tokens')
                })
                
                # Extract detailed token breakdowns
                token_data["input_token_details"] = usage.get('input_token_details', {})
                token_data["output_token_details"] = usage.get('output_token_details', {})
            
            # Clean up None values
            token_data = {k: v for k, v in token_data.items() if v is not None}
            
        except Exception as e:
            self._add_log_to_context(f"Error extracting token usage: {str(e)}", "_extract_token_usage")
        
        return token_data

    def get_available_model_choices(self):
        """
        Return a flat list of available models in 'provider: model' format, only for successfully initialized models.
        """
        choices = ["ALL"]
        for provider, model_config in self.active_model_config.items():
            model_name = model_config.get("model")
            if model_name:
                choices.append(f"{provider}: {model_name}")
        return choices

    def _track_mistral_failure(self, error_type: str):
        """
        Track Mistral AI failures and automatically disable tool calling if it consistently fails.
        
        Args:
            error_type: Type of error ("message_ordering", "rate_limit", "other")
        """
        if not hasattr(self, '_mistral_failure_tracker'):
            self._mistral_failure_tracker = {
                'message_ordering': 0,
                'rate_limit': 0,
                'other': 0,
                'total_failures': 0,
                'last_failure_time': None,
                'consecutive_failures': 0,
                'last_success_time': None
            }
        
        self._mistral_failure_tracker[error_type] += 1
        self._mistral_failure_tracker['total_failures'] += 1
        self._mistral_failure_tracker['consecutive_failures'] += 1
        self._mistral_failure_tracker['last_failure_time'] = time.time()
        
        # Reset consecutive failures if we had a recent success (within last 5 minutes)
        if (self._mistral_failure_tracker.get('last_success_time') and 
            time.time() - self._mistral_failure_tracker['last_success_time'] < 300):
            self._mistral_failure_tracker['consecutive_failures'] = 0
        
        # If we have too many consecutive message ordering failures, disable tool calling for Mistral
        if self._mistral_failure_tracker['message_ordering'] >= 3:
            if self.LLM_CONFIG.get('mistral', {}).get('force_tools', True):
                print(f"⚠️ Mistral AI has had {self._mistral_failure_tracker['message_ordering']} message ordering failures. Disabling tool calling.")
                self.LLM_CONFIG['mistral']['force_tools'] = False
                self.LLM_CONFIG['mistral']['tool_support'] = False
        
        # If we have too many consecutive failures overall, disable tool calling
        elif self._mistral_failure_tracker['consecutive_failures'] >= 5:
            if self.LLM_CONFIG.get('mistral', {}).get('force_tools', True):
                print(f"⚠️ Mistral AI has had {self._mistral_failure_tracker['consecutive_failures']} consecutive failures. Disabling tool calling.")
                self.LLM_CONFIG['mistral']['force_tools'] = False
                self.LLM_CONFIG['mistral']['tool_support'] = False
        
        # If we have too many total failures, consider disabling Mistral entirely
        if self._mistral_failure_tracker['total_failures'] >= 15:
            print(f"⚠️ Mistral AI has had {self._mistral_failure_tracker['total_failures']} total failures. Consider using alternative providers.")

    def _track_mistral_success(self):
        """
        Track successful Mistral AI requests to reset failure counters.
        """
        if hasattr(self, '_mistral_failure_tracker'):
            self._mistral_failure_tracker['consecutive_failures'] = 0
            self._mistral_failure_tracker['last_success_time'] = time.time()
            
            # If tools were disabled due to failures, consider re-enabling them after success
            if (self.LLM_CONFIG.get('mistral', {}).get('force_tools', False) == False and 
                self._mistral_failure_tracker['message_ordering'] < 2):
                print(f"✅ Mistral AI success - considering re-enabling tool calling")
                self.LLM_CONFIG['mistral']['force_tools'] = True
                self.LLM_CONFIG['mistral']['tool_support'] = True

    def _handle_mistral_message_ordering_error(self, error, llm_name, llm_type, messages, llm):
        """
        Handle Mistral AI message ordering errors by reconstructing the conversation.
        
        Args:
            error: The original error
            llm_name: Name of the LLM for logging
            llm_type: Type of LLM
            messages: Original message history
            llm: LLM instance
            
        Returns:
            Response from the retry attempt or raises exception
        """
        error_str = str(error)
        print(f"❌ Mistral AI message ordering error detected: {error_str}")
        self._track_mistral_failure("message_ordering")
        print(f"🔄 Attempting to fix message ordering and retry...")
        
        # Try to fix the message ordering by reconstructing the conversation
        try:
            # Extract the original question and tool results
            original_question = None
            tool_results = []
            
            for msg in messages:
                if hasattr(msg, 'type'):
                    if msg.type == 'human' and not any('reminder' in str(msg.content).lower() for reminder in ['use tools', 'final answer', 'analyze']):
                        original_question = msg.content
                    elif msg.type == 'tool':
                        tool_results.append({
                            'name': getattr(msg, 'name', 'unknown'),
                            'content': msg.content,
                            'tool_call_id': getattr(msg, 'tool_call_id', getattr(msg, 'name', 'unknown'))
                        })
            
            if original_question and tool_results:
                # Reconstruct a clean conversation for Mistral
                clean_messages = []
                
                # Add system message if present
                for msg in messages:
                    if hasattr(msg, 'type') and msg.type == 'system':
                        clean_messages.append(msg)
                        break
                
                # Add the original question
                clean_messages.append(HumanMessage(content=original_question))
                
                # Add tool results as a single user message
                if tool_results:
                    tool_summary = "Tool results:\n" + "\n".join([f"{result['name']}: {result['content']}" for result in tool_results])
                    clean_messages.append(HumanMessage(content=f"Please analyze these results and provide the final answer:\n{tool_summary}"))
                
                # Try the clean conversation
                response = self._invoke_llm_provider(llm, clean_messages)
                print(f"✅ Successfully retried with clean message ordering")
                return response
            else:
                raise Exception("Could not reconstruct clean conversation for Mistral AI")
                
        except Exception as retry_error:
            print(f"❌ Failed to fix Mistral AI message ordering: {retry_error}")
            # Fall back to error handler
            handled, result = self._handle_llm_error(error, llm_name, llm_type, phase="request", messages=messages, llm=llm)
            if handled:
                return result
            else:
                raise Exception(f"{llm_name} failed: {error}")

    def _handle_mistral_message_ordering_error_in_tool_loop(self, error, llm_type, messages, llm, tool_results_history):
        """
        Handle Mistral AI message ordering errors specifically in the tool loop context.
        
        Args:
            error: The original error
            llm_type: Type of LLM
            messages: Original message history
            llm: LLM instance
            tool_results_history: History of tool results
            
        Returns:
            Response from the retry attempt or raises exception
        """
        error_str = str(error)
        print(f"❌ [Tool Loop] Mistral AI message ordering error detected: {error_str}")
        try:
            response = self._handle_mistral_message_ordering_error(error, llm_type, llm_type, messages, llm)
            print(f"✅ [Tool Loop] Successfully retried with clean message ordering")
            return response
        except Exception as retry_error:
            print(f"❌ [Tool Loop] Failed to fix Mistral AI message ordering: {retry_error}")
            # Fall back to error handler
            handled, result = self._handle_llm_error(error, llm_name=llm_type, llm_type=llm_type, phase="tool_loop",
                messages=messages, llm=llm, tool_results_history=tool_results_history)
            if handled:
                return result
            else:
                raise
