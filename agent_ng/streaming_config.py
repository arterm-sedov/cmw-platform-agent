"""
Streaming Configuration
======================

Centralized configuration for streaming parameters.
"""

import os
from typing import Optional


class StreamingConfig:
    """Centralized streaming configuration"""
    
    # Default values
    DEFAULT_MAX_TOOL_CALL_ITERATIONS = 25
    
    def __init__(self):
        self.max_tool_call_iterations = self._get_max_tool_call_iterations()
    
    def _get_max_tool_call_iterations(self) -> int:
        """Get max tool call iterations from environment or use default"""
        try:
            return int(os.getenv('LANGCHAIN_MAX_TOOL_CALL_ITERATIONS', self.DEFAULT_MAX_TOOL_CALL_ITERATIONS))
        except (ValueError, TypeError):
            return self.DEFAULT_MAX_TOOL_CALL_ITERATIONS
    
    def get_max_tool_call_iterations(self) -> int:
        """Get the configured max tool call iterations"""
        return self.max_tool_call_iterations
    
    def set_max_tool_call_iterations(self, value: int) -> None:
        """Set max tool call iterations (for testing or runtime changes)"""
        self.max_tool_call_iterations = value
    
    # Backward compatibility aliases
    def get_max_iterations(self) -> int:
        """Get the configured max tool call iterations (backward compatibility)"""
        return self.get_max_tool_call_iterations()
    
    def set_max_iterations(self, value: int) -> None:
        """Set max tool call iterations (backward compatibility)"""
        self.set_max_tool_call_iterations(value)


# Global configuration instance
_config_instance: Optional[StreamingConfig] = None


def get_streaming_config() -> StreamingConfig:
    """Get the global streaming configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = StreamingConfig()
    return _config_instance


def reset_streaming_config() -> None:
    """Reset the global configuration (useful for testing)"""
    global _config_instance
    _config_instance = None
