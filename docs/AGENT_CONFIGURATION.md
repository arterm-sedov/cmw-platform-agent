# Agent Configuration

## Overview

The CMW Platform Agent uses a central configuration system that allows you to customize various settings including refresh intervals, language preferences, port settings, and debug options.

## Configuration File

The main configuration is located in `agent_ng/agent_config.py` and provides:

- **Centralized Settings**: All configurable parameters in one place
- **Environment Variable Support**: Override settings via environment variables
- **Type Safety**: Uses dataclasses for type-safe configuration
- **Easy Access**: Simple functions to get configuration values

## Configuration Categories

### 1. Refresh Intervals

Controls how often different UI components are updated:

```python
@dataclass
class RefreshIntervals:
    status: float = 2.0      # Status pane refresh interval (seconds)
    logs: float = 3.0        # Logs pane refresh interval (seconds)
    stats: float = 4.0       # Stats pane refresh interval (seconds)
    progress: float = 2.0    # Progress pane refresh interval (seconds)
```

### 2. Language Settings

```python
default_language: str = "ru"  # Default language (ru/en)
supported_languages: list = ["en", "ru"]  # Supported languages
```

### 3. Port Settings

```python
default_port: int = 7860      # Default port
auto_port_range: int = 10     # Number of ports to try when auto-finding
```

### 4. Agent Settings

```python
max_conversation_history: int = 50    # Max conversation history
max_tokens_per_request: int = 4000   # Max tokens per request
request_timeout: float = 30.0        # Request timeout in seconds
```

### 5. Debug Settings

```python
debug_mode: bool = False      # Enable debug mode
verbose_logging: bool = False # Enable verbose logging
```

## Environment Variables

You can override any configuration using environment variables:

### Language Settings
```bash
# Set default language
export CMW_DEFAULT_LANGUAGE="en"  # or "ru"
```

### Port Settings
```bash
# Set default port
export CMW_DEFAULT_PORT="7861"
```

### Refresh Intervals
```bash
# Note: All refresh intervals are internal application parameters
# and should be modified in the code, not via environment variables
# 
# To modify refresh intervals, edit agent_ng/agent_config.py:
# - Status: 2.0s (internal)
# - Logs: 3.0s (internal)  
# - Stats: 4.0s (internal)
# - Progress: 2.0s (internal)
```

### Debug Settings
```bash
# Enable debug mode
export CMW_DEBUG_MODE="true"
export CMW_VERBOSE_LOGGING="true"
```

## Command Line Usage

### View Current Configuration
```bash
python agent_ng/app_ng_modular.py --config
```

### Override Settings
```bash
# Override port
python agent_ng/app_ng_modular.py --en -p 7861

# Use auto port finding
python agent_ng/app_ng_modular.py --ru --auto-port
```

## Programmatic Usage

### Get Configuration Values
```python
from agent_ng.agent_config import get_refresh_intervals, get_language_settings

# Get refresh intervals
intervals = get_refresh_intervals()
print(f"Status updates every {intervals.status} seconds")

# Get language settings
lang_settings = get_language_settings()
print(f"Default language: {lang_settings['default_language']}")
```

### Update Configuration
```python
from agent_ng.agent_config import config

# Update refresh intervals
config.update_setting('refresh_intervals', 'status', 1.0)  # 1 second
config.update_setting('refresh_intervals', 'logs', 2.0)    # 2 seconds

# Update other settings
config.update_setting('agent_settings', 'max_tokens_per_request', 8000)
```

## Default Values

| Setting | Default Value | Description | Environment Configurable |
|---------|---------------|-------------|-------------------------|
| `status` | 2.0s | Status pane refresh interval | ❌ Internal only |
| `logs` | 3.0s | Logs pane refresh interval | ❌ Internal only |
| `stats` | 4.0s | Stats pane refresh interval | ❌ Internal only |
| `progress` | 2.0s | Progress pane refresh interval | ❌ Internal only |
| `default_language` | "ru" | Default language | ✅ `CMW_DEFAULT_LANGUAGE` |
| `default_port` | 7860 | Default port | ✅ `CMW_DEFAULT_PORT` |
| `auto_port_range` | 10 | Port range for auto-finding | ❌ Internal only |
| `max_conversation_history` | 50 | Max conversation history | ❌ Internal only |
| `max_tokens_per_request` | 4000 | Max tokens per request | ❌ Internal only |
| `request_timeout` | 30.0s | Request timeout | ❌ Internal only |
| `debug_mode` | False | Debug mode | ✅ `CMW_DEBUG_MODE` |
| `verbose_logging` | False | Verbose logging | ✅ `CMW_VERBOSE_LOGGING` |

## Best Practices

1. **Use Environment Variables**: For deployment and different environments
2. **Keep Defaults Sensible**: Default values should work for most use cases
3. **Document Changes**: When modifying defaults, update documentation
4. **Test Configuration**: Always test configuration changes
5. **Use Type Hints**: Configuration uses type hints for better IDE support

## Troubleshooting

### Configuration Not Loading
- Check that `agent_ng/agent_config.py` is properly imported
- Verify environment variable names are correct
- Check for typos in configuration keys

### Refresh Intervals Not Working
- Ensure the UI Manager is using `get_refresh_intervals()`
- Check that timer components are properly created
- Verify event handlers are connected

### Port Issues
- Check if port is already in use
- Use `--auto-port` flag for automatic port finding
- Verify port range settings

## Future Enhancements

- **Configuration File**: Support for YAML/JSON configuration files
- **Hot Reload**: Reload configuration without restarting
- **Validation**: Configuration validation and error reporting
- **Profiles**: Different configuration profiles for different environments
- **UI Configuration**: Web-based configuration interface
