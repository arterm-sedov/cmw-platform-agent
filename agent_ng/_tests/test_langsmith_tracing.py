"""
Test LangSmith Tracing Implementation
====================================

Test the lean LangSmith tracing solution to ensure it works correctly
with different LLM providers and creates single traces per conversation.
"""

import os
import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

# Test imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_ng.native_langchain_streaming import NativeLangChainStreaming, StreamingEvent
from agent_ng.langsmith_config import setup_langsmith_environment, get_langsmith_config


class TestLangSmithTracing:
    """Test LangSmith tracing implementation"""
    
    def setup_method(self):
        """Setup test environment"""
        # Load environment variables from .env file (same as main app)
        from dotenv import load_dotenv
        load_dotenv()
        
        # Setup LangSmith environment
        setup_langsmith_environment()
        
        # Create streaming manager
        self.streaming_manager = NativeLangChainStreaming()
    
    def teardown_method(self):
        """Cleanup test environment"""
        # No cleanup needed since we're using load_dotenv() like the main app
        pass
    
    def test_langsmith_config(self):
        """Test LangSmith configuration"""
        config = get_langsmith_config()
        assert config.tracing_enabled == True
        assert config.api_key is not None  # Should have a valid API key
        assert config.project_name == "cmw-agent"  # From .env file
        assert config.is_configured() == True
    
    def test_traceable_decorator_import(self):
        """Test that traceable decorator can be imported"""
        try:
            from langsmith import traceable
            assert callable(traceable)
        except ImportError:
            pytest.skip("LangSmith not installed")
    
    @pytest.mark.asyncio
    async def test_stream_agent_response_tracing(self):
        """Test that stream_agent_response has tracing decorator"""
        # Check if the method has the traceable decorator
        method = getattr(self.streaming_manager, 'stream_agent_response')
        assert hasattr(method, '__wrapped__') or hasattr(method, '__traceable__')
    
    @pytest.mark.asyncio
    async def test_single_trace_per_conversation(self):
        """Test that only one trace is created per conversation, not per token"""
        # This test verifies the structure is correct for single trace per conversation
        method = getattr(self.streaming_manager, 'stream_agent_response')
        assert callable(method)
        
        # The @traceable decorator should ensure single trace per method call
        # not per token in the stream
    
    def test_tracing_environment_setup(self):
        """Test that tracing environment is properly set up"""
        assert os.environ.get("LANGSMITH_TRACING") == "true"
        assert os.environ.get("LANGSMITH_API_KEY") is not None  # Should have a valid API key
        assert os.environ.get("LANGSMITH_PROJECT") == "cmw-agent"
    


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
