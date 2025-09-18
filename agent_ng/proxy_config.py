"""
Proxy Configuration Module
=========================

Clean, modular proxy configuration system with separate configs for:
- Agent requests and tools
- LLM providers

Features:
- Environment variable loading via dotenv
- Separate proxy configs for different components
- Abstract and DRY implementation
- Type-safe configuration
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Any
from dotenv import load_dotenv

# Load environment variables from .env file
print("🔍 [DEBUG] Loading .env file...")
env_loaded = load_dotenv()
print(f"🔍 [DEBUG] .env file loaded: {env_loaded}")

# Check if .env file exists
import os
env_file = ".env"
if os.path.exists(env_file):
    print(f"✅ [DEBUG] .env file found at: {os.path.abspath(env_file)}")
    with open(env_file, 'r') as f:
        content = f.read()
        print(f"🔍 [DEBUG] .env content preview:")
        for line in content.split('\n')[:10]:  # Show first 10 lines
            if line.strip():
                print(f"  {line}")
else:
    print(f"❌ [DEBUG] .env file NOT found at: {os.path.abspath(env_file)}")
    print(f"🔍 [DEBUG] Current working directory: {os.getcwd()}")
    print(f"🔍 [DEBUG] Files in current directory: {os.listdir('.')}")


@dataclass
class ProxyConfig:
    """Base proxy configuration"""
    enabled: bool = False
    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None
    no_proxy: Optional[str] = None
    verify_ssl: bool = True
    timeout: float = 30.0

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for requests library"""
        if not self.enabled:
            return {}
        
        proxy_dict = {}
        if self.http_proxy:
            proxy_dict['http'] = self.http_proxy
        if self.https_proxy:
            proxy_dict['https'] = self.https_proxy
        return proxy_dict

    def to_env_vars(self) -> Dict[str, str]:
        """Convert to environment variables for subprocess calls"""
        if not self.enabled:
            return {}
        
        env_vars = {}
        if self.http_proxy:
            env_vars['HTTP_PROXY'] = self.http_proxy
            env_vars['http_proxy'] = self.http_proxy
        if self.https_proxy:
            env_vars['HTTPS_PROXY'] = self.https_proxy
            env_vars['https_proxy'] = self.https_proxy
        if self.no_proxy:
            env_vars['NO_PROXY'] = self.no_proxy
            env_vars['no_proxy'] = self.no_proxy
        
        if not self.verify_ssl:
            env_vars['CURL_CA_BUNDLE'] = ''
            env_vars['REQUESTS_CA_BUNDLE'] = ''
        
        return env_vars


class ProxyConfigManager:
    """Manages proxy configurations for different components"""
    
    def __init__(self):
        self._agent_config: Optional[ProxyConfig] = None
        self._llm_config: Optional[ProxyConfig] = None
        self._agent_cleared_env_vars = False  # Track if we've cleared agent env vars
    
    def get_agent_proxy_config(self) -> ProxyConfig:
        """Get proxy configuration for agent requests and tools"""
        if self._agent_config is None:
            # Clear any existing standard proxy environment variables first
            if not self._agent_cleared_env_vars:
                print(f"🧹 [DEBUG] Pre-clearing standard proxy environment variables for agent")
                for key in ['HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy', 'NO_PROXY', 'no_proxy']:
                    if key in os.environ:
                        del os.environ[key]
                        print(f"  - Pre-cleared {key}")
                self._agent_cleared_env_vars = True
            
            self._agent_config = self._load_proxy_config(
                prefix="AGENT_PROXY",
                fallback_prefix="PROXY"
            )
        return self._agent_config
    
    def get_llm_proxy_config(self) -> ProxyConfig:
        """Get proxy configuration for LLM providers"""
        if self._llm_config is None:
            self._llm_config = self._load_proxy_config(
                prefix="LLM_PROXY",
                fallback_prefix="PROXY"
            )
        return self._llm_config
    
    def reset_configs(self):
        """Reset all proxy configurations (useful when switching LLM or refreshing)"""
        print(f"🔄 [DEBUG] Resetting proxy configurations")
        self._agent_config = None
        self._llm_config = None
        self._agent_cleared_env_vars = False
        # Re-clear environment variables
        for key in ['HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy', 'NO_PROXY', 'no_proxy']:
            if key in os.environ:
                del os.environ[key]
                print(f"  - Reset cleared {key}")
    
    def _load_proxy_config(self, prefix: str, fallback_prefix: str = None) -> ProxyConfig:
        """Load proxy configuration from environment variables"""
        config = ProxyConfig()
        
        print(f"🔍 [DEBUG] Loading proxy config for prefix: {prefix}")
        print(f"🔍 [DEBUG] Fallback prefix: {fallback_prefix}")
        
        # Check if proxy is enabled
        enabled_key = f"{prefix}_ENABLED"
        enabled_value = os.getenv(enabled_key, '')
        print(f"🔍 [DEBUG] {enabled_key} = '{enabled_value}'")
        
        if enabled_value.lower() in ['true', '1', 'yes']:
            config.enabled = True
            print(f"✅ [DEBUG] {prefix} proxy ENABLED")
        else:
            print(f"❌ [DEBUG] {prefix} proxy DISABLED")
            # Clear standard proxy environment variables when disabled
            if prefix == "AGENT_PROXY":
                print(f"🧹 [DEBUG] Clearing standard proxy environment variables for agent")
                for key in ['HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy', 'NO_PROXY', 'no_proxy']:
                    if key in os.environ:
                        del os.environ[key]
                        print(f"  - Cleared {key}")
        
        # Load proxy URLs with fallback to generic variables
        http_key = f"{prefix}_HTTP_PROXY"
        https_key = f"{prefix}_HTTPS_PROXY"
        no_proxy_key = f"{prefix}_NO_PROXY"
        
        # Try specific prefix first, then fallback
        http_value = os.getenv(http_key) or (os.getenv(f"{fallback_prefix}_HTTP_PROXY") if fallback_prefix else None) or os.getenv('HTTP_PROXY')
        https_value = os.getenv(https_key) or (os.getenv(f"{fallback_prefix}_HTTPS_PROXY") if fallback_prefix else None) or os.getenv('HTTPS_PROXY')
        no_proxy_value = os.getenv(no_proxy_key) or (os.getenv(f"{fallback_prefix}_NO_PROXY") if fallback_prefix else None) or os.getenv('NO_PROXY')
        
        config.http_proxy = http_value
        config.https_proxy = https_value
        config.no_proxy = no_proxy_value
        
        print(f"🔍 [DEBUG] {http_key} = '{http_value}'")
        print(f"🔍 [DEBUG] {https_key} = '{https_value}'")
        print(f"🔍 [DEBUG] {no_proxy_key} = '{no_proxy_value}'")
        print(f"🔍 [DEBUG] Standard HTTP_PROXY = '{os.getenv('HTTP_PROXY')}'")
        print(f"🔍 [DEBUG] Standard HTTPS_PROXY = '{os.getenv('HTTPS_PROXY')}'")
        print(f"🔍 [DEBUG] Standard NO_PROXY = '{os.getenv('NO_PROXY')}'")
        
        # SSL verification
        ssl_key = f"{prefix}_VERIFY_SSL"
        if os.getenv(ssl_key, '').lower() in ['false', '0', 'no']:
            config.verify_ssl = False
        
        # Timeout
        timeout_key = f"{prefix}_TIMEOUT"
        try:
            timeout_value = os.getenv(timeout_key)
            if timeout_value:
                config.timeout = float(timeout_value)
        except ValueError:
            pass
        
        print(f"🔍 [DEBUG] Final config for {prefix}:")
        print(f"  - enabled: {config.enabled}")
        print(f"  - http_proxy: {config.http_proxy}")
        print(f"  - https_proxy: {config.https_proxy}")
        print(f"  - no_proxy: {config.no_proxy}")
        print(f"  - verify_ssl: {config.verify_ssl}")
        print(f"  - timeout: {config.timeout}")
        print(f"  - to_dict(): {config.to_dict()}")
        
        return config
    
    def apply_llm_proxy_environment(self):
        """Apply LLM proxy configuration to environment variables"""
        llm_config = self.get_llm_proxy_config()
        if llm_config.enabled:
            env_vars = llm_config.to_env_vars()
            for key, value in env_vars.items():
                os.environ[key] = value


# Global proxy manager instance
_proxy_manager = None

def get_proxy_manager() -> ProxyConfigManager:
    """Get global proxy manager instance"""
    global _proxy_manager
    if _proxy_manager is None:
        _proxy_manager = ProxyConfigManager()
    return _proxy_manager

def get_agent_proxy_config() -> ProxyConfig:
    """Get agent proxy configuration"""
    return get_proxy_manager().get_agent_proxy_config()

def get_llm_proxy_config() -> ProxyConfig:
    """Get LLM proxy configuration"""
    return get_proxy_manager().get_llm_proxy_config()

def apply_llm_proxy_environment():
    """Apply LLM proxy configuration to environment"""
    get_proxy_manager().apply_llm_proxy_environment()

def reset_proxy_configs():
    """Reset all proxy configurations (useful when switching LLM or refreshing)"""
    get_proxy_manager().reset_configs()
