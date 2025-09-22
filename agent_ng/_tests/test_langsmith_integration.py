"""
LangSmith Integration Tests
==========================

Tests for LangSmith observability integration in the CMW Platform Agent.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Test LangSmith configuration
def test_langsmith_config_creation():
    """Test LangSmith configuration creation"""
    from agent_ng.langsmith_config import LangSmithConfig
    
    config = LangSmithConfig()
    assert hasattr(config, 'tracing_enabled')
    assert hasattr(config, 'api_key')
    assert hasattr(config, 'workspace_id')
    assert hasattr(config, 'project_name')

def test_langsmith_config_with_env_vars():
    """Test LangSmith configuration with environment variables"""
    with patch.dict(os.environ, {
        'LANGSMITH_TRACING': 'true',
        'LANGSMITH_API_KEY': 'test_key',
        'LANGSMITH_WORKSPACE_ID': 'test_workspace',
        'LANGSMITH_PROJECT': 'test_project'
    }):
        from agent_ng.langsmith_config import LangSmithConfig
        
        config = LangSmithConfig()
        assert config.tracing_enabled is True
        assert config.api_key == 'test_key'
        assert config.workspace_id == 'test_workspace'
        assert config.project_name == 'test_project'
        assert config.is_configured() is True

def test_langsmith_config_without_env_vars():
    """Test LangSmith configuration without environment variables"""
    with patch.dict(os.environ, {}, clear=True):
        from agent_ng.langsmith_config import LangSmithConfig
        
        config = LangSmithConfig()
        assert config.tracing_enabled is False
        assert config.api_key is None
        assert config.workspace_id is None
        assert config.project_name == 'cmw-platform-agent'
        assert config.is_configured() is False

def test_get_traceable_decorator():
    """Test getting traceable decorator"""
    from agent_ng.langsmith_config import get_traceable_decorator
    
    # Test without LangSmith configured
    with patch.dict(os.environ, {}, clear=True):
        decorator = get_traceable_decorator()
        assert decorator is None

def test_get_openai_wrapper():
    """Test getting OpenAI wrapper"""
    from agent_ng.langsmith_config import get_openai_wrapper
    
    # Test without LangSmith configured
    with patch.dict(os.environ, {}, clear=True):
        wrapper = get_openai_wrapper()
        assert wrapper is None

def test_llm_manager_langsmith_integration():
    """Test LLM manager LangSmith integration"""
    from agent_ng.llm_manager import LLMManager
    
    manager = LLMManager()
    
    # Test wrapping method exists
    assert hasattr(manager, '_wrap_llm_with_langsmith')
    
    # Test with mock LLM
    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    
    # Test without LangSmith configured
    with patch.dict(os.environ, {}, clear=True):
        wrapped_llm = manager._wrap_llm_with_langsmith(mock_llm, "openrouter")
        assert wrapped_llm is mock_llm  # Should return original LLM

def test_app_langsmith_integration():
    """Test app LangSmith integration"""
    from agent_ng.app_ng_modular import NextGenApp
    
    # Test with LangSmith disabled
    with patch.dict(os.environ, {}, clear=True):
        app = NextGenApp()
        assert hasattr(app, '_get_traceable_decorator')
        
        decorator = app._get_traceable_decorator()
        # Should return a no-op decorator when LangSmith is not configured
        assert callable(decorator)

def test_langsmith_environment_setup():
    """Test LangSmith environment setup"""
    from agent_ng.langsmith_config import setup_langsmith_environment
    
    # Test with LangSmith disabled
    with patch.dict(os.environ, {}, clear=True):
        result = setup_langsmith_environment()
        assert result is False
    
    # Test with LangSmith enabled
    with patch.dict(os.environ, {
        'LANGSMITH_TRACING': 'true',
        'LANGSMITH_API_KEY': 'test_key'
    }):
        result = setup_langsmith_environment()
        assert result is True

if __name__ == "__main__":
    pytest.main([__file__])
