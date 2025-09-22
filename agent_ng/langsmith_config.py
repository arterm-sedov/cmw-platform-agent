"""
LangSmith Configuration Module
=============================

Lean LangSmith observability configuration for the CMW Platform Agent.
Provides minimal setup for tracing LLM calls and application flows.

Based on LangSmith's official quickstart guide:
https://docs.langchain.com/langsmith/observability-quickstart
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LangSmithConfig:
    """Lean LangSmith configuration manager"""
    
    def __init__(self):
        self.tracing_enabled = self._is_tracing_enabled()
        self.api_key = self._get_api_key()
        self.workspace_id = self._get_workspace_id()
        self.project_name = self._get_project_name()
        
    def _is_tracing_enabled(self) -> bool:
        """Check if LangSmith tracing is enabled"""
        return os.getenv("LANGSMITH_TRACING", "false").lower() in ("true", "1", "yes")
    
    def _get_api_key(self) -> Optional[str]:
        """Get LangSmith API key"""
        return os.getenv("LANGSMITH_API_KEY")
    
    def _get_workspace_id(self) -> Optional[str]:
        """Get LangSmith workspace ID"""
        return os.getenv("LANGSMITH_WORKSPACE_ID")
    
    def _get_project_name(self) -> str:
        """Get project name for traces"""
        return os.getenv("LANGSMITH_PROJECT", "cmw-platform-agent")
    
    def is_configured(self) -> bool:
        """Check if LangSmith is properly configured"""
        return self.tracing_enabled and self.api_key is not None
    
    def get_config_dict(self) -> dict:
        """Get configuration as dictionary"""
        return {
            "tracing_enabled": self.tracing_enabled,
            "api_key": "***" if self.api_key else None,
            "workspace_id": self.workspace_id,
            "project_name": self.project_name,
            "is_configured": self.is_configured()
        }


def get_langsmith_config() -> LangSmithConfig:
    """Get LangSmith configuration instance"""
    return LangSmithConfig()


def setup_langsmith_environment():
    """Setup LangSmith environment variables if not already set"""
    config = get_langsmith_config()
    
    if config.tracing_enabled and config.api_key:
        # Set environment variables for LangSmith
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = config.api_key
        
        if config.workspace_id:
            os.environ["LANGSMITH_WORKSPACE_ID"] = config.workspace_id
        
        print(f"✅ LangSmith tracing enabled for project: {config.project_name}")
        return True
    else:
        print("ℹ️ LangSmith tracing disabled (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY)")
        return False


def get_traceable_decorator():
    """Get the traceable decorator if LangSmith is configured"""
    config = get_langsmith_config()
    
    if config.is_configured():
        try:
            from langsmith import traceable
            return traceable
        except ImportError:
            print("⚠️ LangSmith not installed. Install with: pip install langsmith")
            return None
    else:
        return None


def get_openai_wrapper():
    """Get OpenAI wrapper for tracing if LangSmith is configured"""
    config = get_langsmith_config()
    
    if config.is_configured():
        try:
            from langsmith.wrappers import wrap_openai
            return wrap_openai
        except ImportError:
            print("⚠️ LangSmith not installed. Install with: pip install langsmith")
            return None
    else:
        return None


# Initialize configuration on import
_config = get_langsmith_config()
setup_langsmith_environment()
