# Proxy Configuration for CMW Platform Agent

Clean, modular, and DRY proxy configuration system with separate configs for different components.

## 🏗️ Architecture

The proxy system is designed with **separation of concerns** and **modularity**:

- **`agent_ng/proxy_config.py`** - Central proxy configuration module with reset functionality
- **Agent Proxy Config** - For HTTP requests and tools (`tools/requests_.py`, `tools/tools.py`)
- **LLM Proxy Config** - For LLM providers (`agent_ng/llm_manager.py`)
- **Selective Reset** - Configuration resets only when switching between different LLM providers

## 🚀 Quick Start

### 1. Create `.env` file

```bash
# Copy the example configuration
cp docs/PROXY_CONFIGURATION_EXAMPLE.env .env

# Edit with your proxy settings
nano .env
```

### 2. Configure Agent Proxy (for requests and tools)

```bash
# Disable agent proxy for local development
AGENT_PROXY_ENABLED=false
AGENT_PROXY_NO_PROXY=localhost,127.0.0.1,ubuntu-vm-server-gen2

# Or enable with your proxy settings
# AGENT_PROXY_ENABLED=true
# AGENT_PROXY_HTTP_PROXY=http://your-proxy:8080
# AGENT_PROXY_HTTPS_PROXY=http://your-proxy:8080
```

### 3. Configure LLM Proxy (for LLM providers)

```bash
# Enable LLM proxy for external API calls
LLM_PROXY_ENABLED=true
LLM_PROXY_HTTP_PROXY=http://127.0.0.1:18080
LLM_PROXY_HTTPS_PROXY=http://127.0.0.1:18080
LLM_PROXY_NO_PROXY=localhost,127.0.0.1
```

## 📋 Environment Variables

### Agent Proxy Variables

| Variable                  | Description               | Default | Required |
| ------------------------- | ------------------------- | ------- | -------- |
| `AGENT_PROXY_ENABLED`     | Enable agent proxy        | `false` | No       |
| `AGENT_PROXY_HTTP_PROXY`  | HTTP proxy URL            | -       | No*      |
| `AGENT_PROXY_HTTPS_PROXY` | HTTPS proxy URL           | -       | No*      |
| `AGENT_PROXY_NO_PROXY`    | Bypass proxy hosts        | -       | No       |
| `AGENT_PROXY_VERIFY_SSL`  | Verify SSL certificates   | `true`  | No       |
| `AGENT_PROXY_TIMEOUT`     | Request timeout (seconds) | `30`    | No       |

### LLM Proxy Variables

| Variable                | Description               | Default | Required |
| ----------------------- | ------------------------- | ------- | -------- |
| `LLM_PROXY_ENABLED`     | Enable LLM proxy          | `false` | No       |
| `LLM_PROXY_HTTP_PROXY`  | HTTP proxy URL            | -       | No*      |
| `LLM_PROXY_HTTPS_PROXY` | HTTPS proxy URL           | -       | No*      |
| `LLM_PROXY_NO_PROXY`    | Bypass proxy hosts        | -       | No       |
| `LLM_PROXY_VERIFY_SSL`  | Verify SSL certificates   | `true`  | No       |
| `LLM_PROXY_TIMEOUT`     | Request timeout (seconds) | `30`    | No       |

*At least one proxy URL is required when enabled.

## 🔄 Fallback Hierarchy

The system uses a **smart fallback hierarchy**:

1. **Specific prefix** (e.g., `AGENT_PROXY_HTTP_PROXY`)
2. **Generic prefix** (e.g., `PROXY_HTTP_PROXY`)
3. **Standard variables** (e.g., `HTTP_PROXY`)

## 🛠️ Usage Examples

### Basic HTTP Proxy

```bash
# Agent requests and tools
AGENT_PROXY_ENABLED=true
AGENT_PROXY_HTTP_PROXY=http://proxy.company.com:8080

# LLM providers
LLM_PROXY_ENABLED=true
LLM_PROXY_HTTP_PROXY=http://proxy.company.com:8080
```

### Authenticated Proxy

```bash
AGENT_PROXY_HTTP_PROXY=http://username:password@proxy.company.com:8080
LLM_PROXY_HTTPS_PROXY=https://username:password@proxy.company.com:8080
```

### SOCKS5 Proxy

```bash
AGENT_PROXY_HTTP_PROXY=socks5://proxy.company.com:1080
LLM_PROXY_HTTPS_PROXY=socks5://proxy.company.com:1080
```

### Different Proxies for Different Components

```bash
# Agent uses corporate proxy
AGENT_PROXY_ENABLED=true
AGENT_PROXY_HTTP_PROXY=http://corporate-proxy.company.com:8080

# LLMs use dedicated proxy
LLM_PROXY_ENABLED=true
LLM_PROXY_HTTP_PROXY=http://llm-proxy.company.com:8080
```

### SSL Configuration

```bash
# Disable SSL verification for self-signed certificates
AGENT_PROXY_VERIFY_SSL=false
LLM_PROXY_VERIFY_SSL=false
```

## 🔧 Programmatic Usage

### Get Proxy Configuration

```python
from agent_ng.proxy_config import get_agent_proxy_config, get_llm_proxy_config, reset_proxy_configs

# Get agent proxy config
agent_proxy = get_agent_proxy_config()
print(f"Agent proxy enabled: {agent_proxy.enabled}")
print(f"Agent HTTP proxy: {agent_proxy.http_proxy}")

# Get LLM proxy config
llm_proxy = get_llm_proxy_config()
print(f"LLM proxy enabled: {llm_proxy.enabled}")
print(f"LLM HTTPS proxy: {llm_proxy.https_proxy}")

# Reset proxy configurations (useful when switching LLM or refreshing UI)
reset_proxy_configs()
```

### Use with Requests

```python
import requests
from agent_ng.proxy_config import get_agent_proxy_config

proxy_config = get_agent_proxy_config()
response = requests.get(
    "https://api.example.com",
    proxies=proxy_config.to_dict(),
    verify=proxy_config.verify_ssl,
    timeout=proxy_config.timeout
)
```

### Apply LLM Environment

```python
from agent_ng.proxy_config import apply_llm_proxy_environment

# This sets environment variables for LLM providers
apply_llm_proxy_environment()
```

## 🏛️ Architecture Details

### ProxyConfig Class

```python
@dataclass
class ProxyConfig:
    enabled: bool = False
    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None
    no_proxy: Optional[str] = None
    verify_ssl: bool = True
    timeout: float = 30.0

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for requests library"""
    
    def to_env_vars(self) -> Dict[str, str]:
        """Convert to environment variables for subprocess calls"""
```

### ProxyConfigManager Class

```python
class ProxyConfigManager:
    def get_agent_proxy_config(self) -> ProxyConfig:
        """Get proxy configuration for agent requests and tools"""
    
    def get_llm_proxy_config(self) -> ProxyConfig:
        """Get proxy configuration for LLM providers"""
    
    def apply_llm_proxy_environment(self):
        """Apply LLM proxy configuration to environment variables"""
    
    def reset_configs(self):
        """Reset all proxy configurations (useful when switching LLM or refreshing)"""
```

### Reset Function

```python
def reset_proxy_configs():
    """Reset all proxy configurations (useful when switching LLM or refreshing)"""
    get_proxy_manager().reset_configs()
```

## 🔍 Components Using Proxy

### Agent Components

- **`tools/requests_.py`** - Platform API requests
- **`tools/tools.py`** - Web scraping and file downloads

### LLM Components

- **`agent_ng/llm_manager.py`** - All LLM providers:
  - Google Gemini
  - Groq
  - HuggingFace
  - OpenRouter
  - Mistral
  - GigaChat

## 🧪 Testing

### Test Proxy Configuration

```python
# Test script
from agent_ng.proxy_config import get_agent_proxy_config, get_llm_proxy_config, reset_proxy_configs

# Test agent proxy
agent_proxy = get_agent_proxy_config()
assert agent_proxy.enabled == False  # Should be disabled for local development
assert agent_proxy.to_dict() == {}   # Should be empty when disabled

# Test LLM proxy
llm_proxy = get_llm_proxy_config()
assert llm_proxy.enabled == True
assert llm_proxy.https_proxy == "http://127.0.0.1:18080"

# Test reset functionality
reset_proxy_configs()
agent_proxy_after_reset = get_agent_proxy_config()
assert agent_proxy_after_reset.enabled == False
```

### Test Scripts

You can create test scripts to verify proxy configuration:

```python
# Create a simple test script
from agent_ng.proxy_config import get_agent_proxy_config, get_llm_proxy_config, reset_proxy_configs

# Test agent proxy
agent_proxy = get_agent_proxy_config()
print(f"Agent proxy enabled: {agent_proxy.enabled}")

# Test LLM proxy  
llm_proxy = get_llm_proxy_config()
print(f"LLM proxy enabled: {llm_proxy.enabled}")

# Test reset functionality
reset_proxy_configs()
```

### Debug Mode (Development Only)

```bash
# Enable debug logging (development/testing only)
CMW_DEBUG_MODE=true
CMW_VERBOSE_LOGGING=true
```

**Note**: Debug mode should be disabled in production environments.

## 🚨 Troubleshooting

### Common Issues

1. **Proxy Not Working**
   - Check `AGENT_PROXY_ENABLED=true` or `LLM_PROXY_ENABLED=true`
   - Verify proxy URL format includes protocol
   - Test proxy connectivity manually

2. **Agent Still Using Proxy Despite `AGENT_PROXY_ENABLED=false`**
   - This was a known issue that has been fixed
   - The system now properly clears standard proxy environment variables
   - Configuration resets automatically when switching LLM providers

3. **SSL Certificate Errors**
   - Set `AGENT_PROXY_VERIFY_SSL=false` or `LLM_PROXY_VERIFY_SSL=false`
   - Or configure proper CA certificates

4. **Authentication Failures**
   - Use URL-encoded credentials: `http://user:pass@proxy:port`
   - Check username/password format

5. **Timeout Issues**
   - Increase `AGENT_PROXY_TIMEOUT` or `LLM_PROXY_TIMEOUT`
   - Check network connectivity

6. **Configuration Resets on LLM Switch**
   - This only happens when switching between different LLM providers
   - Repeated calls to the same provider preserve the configuration
   - The system automatically reloads configuration from `.env` file when needed

### Debug Commands

```bash
# Test agent proxy
curl --proxy http://your-proxy:8080 https://httpbin.org/ip

# Test LLM proxy
curl --proxy http://your-proxy:8080 https://api.openai.com/v1/models

# Test proxy configuration loading
python -c "from agent_ng.proxy_config import get_agent_proxy_config, get_llm_proxy_config; print('Agent:', get_agent_proxy_config().enabled); print('LLM:', get_llm_proxy_config().enabled)"

# Test reset functionality
python -c "from agent_ng.proxy_config import reset_proxy_configs; reset_proxy_configs(); print('Proxy configs reset')"
```

## 🔒 Security Considerations

1. **Credentials**: Store proxy credentials securely in `.env` file
2. **SSL Verification**: Only disable for trusted internal proxies
3. **No Proxy List**: Configure appropriate bypass rules
4. **Network Security**: Ensure proxy servers are properly secured

## 🚀 Production Considerations

### Performance Optimization

1. **Debug Output**: The current implementation includes debug prints that should be removed for production
2. **Module Loading**: Debug file inspection on import should be disabled
3. **Logging**: Use proper logging framework instead of print statements
4. **Caching**: Configuration caching is already implemented for optimal performance

### Recommended Production Modifications

```python
# Remove these debug lines from proxy_config.py for production:
# Lines 22-40: Module-level debug prints
# Lines 139-202: Verbose debug output in _load_proxy_config
# Lines 100-106, 125-133: Debug prints in reset methods
```

### Clean Production Version

A production-ready version should:

- Remove all `print()` statements
- Use proper logging with configurable levels
- Remove file system inspection on import
- Maintain the same functionality with cleaner code

## 📚 Advanced Configuration

### Custom Proxy Headers

Extend the `ProxyConfig` class for custom headers:

```python
class CustomProxyConfig(ProxyConfig):
    def to_dict(self) -> Dict[str, str]:
        proxy_dict = super().to_dict()
        if self.enabled:
            proxy_dict['custom_header'] = 'value'
        return proxy_dict
```

### Multiple Proxy Support

For complex scenarios, implement custom logic in the `ProxyConfigManager`:

```python
def get_proxy_for_url(self, url: str) -> ProxyConfig:
    """Get proxy configuration based on target URL"""
    if 'api.openai.com' in url:
        return self.get_llm_proxy_config()
    else:
        return self.get_agent_proxy_config()
```

## 🎯 Best Practices

1. **Use separate proxies** for different components when possible
2. **Set appropriate timeouts** based on your network
3. **Configure no_proxy** for internal services
4. **Test thoroughly** before production deployment
5. **Monitor proxy performance** and adjust as needed
6. **Use `AGENT_PROXY_ENABLED=false`** for local development to avoid proxy issues
7. **Enable debug mode only when troubleshooting** proxy configuration issues
8. **Reset configurations** when switching between different network environments
9. **Disable debug mode in production** to maintain clean logs and optimal performance

## 🔄 Recent Improvements

### Version 2.0 Features

1. **Selective Configuration Reset**
   - Proxy configurations only reset when actually switching between different LLM providers
   - Prevents stale proxy settings from affecting new connections
   - Preserves configuration for repeated calls to the same provider

2. **Enhanced Environment Variable Management**
   - Properly clears standard proxy environment variables when agent proxy is disabled
   - Prevents `requests` library from using unintended proxy settings
   - Smart fallback hierarchy for configuration loading

3. **Production-Ready Design**
   - Clean, minimal debug output (disabled by default)
   - Efficient configuration loading without verbose logging
   - Optimized for production performance

4. **Robust Error Handling**
   - Graceful handling of missing or invalid proxy configurations
   - Automatic fallback to direct connections when proxy is disabled
   - Clear error messages for configuration issues

### Migration from Version 1.0

If you're upgrading from the previous proxy system:

1. **Update your `.env` file** with the new variable names:
   - `AGENT_PROXY_ENABLED` instead of generic proxy settings
   - `LLM_PROXY_ENABLED` for LLM-specific proxy configuration

2. **Test your configuration** using the provided test commands:

   ```bash
   python -c "from agent_ng.proxy_config import get_agent_proxy_config, get_llm_proxy_config; print('Agent:', get_agent_proxy_config().enabled); print('LLM:', get_llm_proxy_config().enabled)"
   ```

3. **Verify behavior** when switching LLM providers to ensure proxy settings persist correctly

## 📖 Related Documentation

- [Environment Variables Reference](PROXY_CONFIGURATION_EXAMPLE.env)
- [Agent Configuration](AGENT_CONFIGURATION.md)
- [LLM Manager Documentation](LANGCHAIN_MIGRATION_COMPLETE.md)
