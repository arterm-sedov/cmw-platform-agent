"""
Agent Configuration
==================

Central configuration file for the CMW Platform Agent.
Contains all configurable settings including refresh intervals, timeouts, and other parameters.
"""

from dataclasses import dataclass
from typing import Dict, Any
import os

@dataclass
class RefreshIntervals:
    """Auto-refresh intervals for different UI components"""
    status: float = 2.0      # Status pane refresh interval (seconds)
    logs: float = 3.0        # Logs pane refresh interval (seconds)
    stats: float = 4.0       # Stats pane refresh interval (seconds)
    progress: float = 3.0    # Progress pane refresh interval (seconds) - reduced to prevent UI blocking

@dataclass
class AgentSettings:
    """Main agent configuration settings"""
    # Language settings
    default_language: str = "ru"
    supported_languages: list = None
    
    # Port settings
    default_port: int = 7860
    auto_port_range: int = 10  # Number of ports to try when auto-finding
    
    # UI settings
    refresh_intervals: RefreshIntervals = None
    
    # Agent settings
    max_conversation_history: int = 50
    max_tokens_per_request: int = 4000
    request_timeout: float = 30.0
    
    # Debug settings
    debug_mode: bool = False
    verbose_logging: bool = False
    
    def __post_init__(self):
        """Initialize default values after dataclass creation"""
        if self.supported_languages is None:
            self.supported_languages = ["en", "ru"]
        
        if self.refresh_intervals is None:
            self.refresh_intervals = RefreshIntervals()

class AgentConfig:
    """Central configuration manager for the CMW Platform Agent"""
    
    def __init__(self):
        self.settings = AgentSettings()
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # Language settings
        if os.getenv('CMW_DEFAULT_LANGUAGE'):
            self.settings.default_language = os.getenv('CMW_DEFAULT_LANGUAGE')
        
        # Port settings
        if os.getenv('CMW_DEFAULT_PORT'):
            try:
                self.settings.default_port = int(os.getenv('CMW_DEFAULT_PORT'))
            except ValueError:
                pass
        
        # Debug settings
        if os.getenv('CMW_DEBUG_MODE', '').lower() in ['true', '1', 'yes']:
            self.settings.debug_mode = True
        
        if os.getenv('CMW_VERBOSE_LOGGING', '').lower() in ['true', '1', 'yes']:
            self.settings.verbose_logging = True
        
        # Refresh intervals from environment
        self._load_refresh_intervals_from_env()
    
    def _load_refresh_intervals_from_env(self):
        """Load refresh intervals from environment variables"""
        # Note: All refresh intervals are internal application parameters
        # They should be modified in the code, not via environment variables
        # This method is kept for future extensibility but currently does nothing
        pass
    
    def get_refresh_intervals(self) -> RefreshIntervals:
        """Get the current refresh intervals configuration"""
        return self.settings.refresh_intervals
    
    def get_language_settings(self) -> Dict[str, Any]:
        """Get language-related settings"""
        return {
            'default_language': self.settings.default_language,
            'supported_languages': self.settings.supported_languages
        }
    
    def get_port_settings(self) -> Dict[str, Any]:
        """Get port-related settings"""
        return {
            'default_port': self.settings.default_port,
            'auto_port_range': self.settings.auto_port_range
        }
    
    def get_agent_settings(self) -> Dict[str, Any]:
        """Get agent-related settings"""
        return {
            'max_conversation_history': self.settings.max_conversation_history,
            'max_tokens_per_request': self.settings.max_tokens_per_request,
            'request_timeout': self.settings.request_timeout
        }
    
    def get_debug_settings(self) -> Dict[str, Any]:
        """Get debug-related settings"""
        return {
            'debug_mode': self.settings.debug_mode,
            'verbose_logging': self.settings.verbose_logging
        }
    
    def update_setting(self, category: str, key: str, value: Any):
        """Update a specific setting"""
        if category == 'refresh_intervals':
            if hasattr(self.settings.refresh_intervals, key):
                setattr(self.settings.refresh_intervals, key, value)
        elif hasattr(self.settings, key):
            setattr(self.settings, key, value)
    
    def print_config(self):
        """Print current configuration"""
        print("🔧 Agent Configuration:")
        print(f"  Language: {self.settings.default_language}")
        print(f"  Port: {self.settings.default_port}")
        print(f"  Debug Mode: {self.settings.debug_mode}")
        print(f"  Refresh Intervals:")
        print(f"    Status: {self.settings.refresh_intervals.status}s")
        print(f"    Logs: {self.settings.refresh_intervals.logs}s")
        print(f"    Stats: {self.settings.refresh_intervals.stats}s")
        print(f"    Progress: {self.settings.refresh_intervals.progress}s")

# Global configuration instance
config = AgentConfig()

# Convenience functions for easy access
def get_refresh_intervals() -> RefreshIntervals:
    """Get refresh intervals configuration"""
    return config.get_refresh_intervals()

def get_language_settings() -> Dict[str, Any]:
    """Get language settings"""
    return config.get_language_settings()

def get_port_settings() -> Dict[str, Any]:
    """Get port settings"""
    return config.get_port_settings()

def get_agent_settings() -> Dict[str, Any]:
    """Get agent settings"""
    return config.get_agent_settings()

def get_debug_settings() -> Dict[str, Any]:
    """Get debug settings"""
    return config.get_debug_settings()
