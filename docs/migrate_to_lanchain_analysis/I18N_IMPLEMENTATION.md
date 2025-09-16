# Internationalization (i18n) Implementation

## Overview

The CMW Platform Agent now supports both English and Russian languages through a custom translation system. The implementation provides comprehensive translation coverage with command-line language selection and environment variable configuration.

## Features

- **Single App Architecture**: One app supports both languages dynamically
- **Custom Translation System**: Direct translation key lookup for better control
- **Comprehensive Translation**: All UI text, messages, and status indicators are translated
- **Command-Line Language Selection**: Simple command-line interface for language selection
- **Environment Variable Support**: Configurable default language via environment variables
- **Russian Default**: Russian is the default language
- **Fallback Support**: Graceful fallback to English if translations are missing

## Usage

### Command Line Interface

```bash
# Russian (default)
python agent_ng/app_ng_modular.py

# English
python agent_ng/app_ng_modular.py --en

# Russian explicitly
python agent_ng/app_ng_modular.py --ru

# English with specific port
python agent_ng/app_ng_modular.py --en -p 7861

# Russian with auto port finding
python agent_ng/app_ng_modular.py --ru --auto-port

# Show help
python agent_ng/app_ng_modular.py --help
```

### Environment Variable Configuration

```bash
# Windows PowerShell
$env:CMW_DEFAULT_LANGUAGE="en"  # Set English as default
$env:CMW_DEFAULT_LANGUAGE="ru"  # Set Russian as default

# Windows Command Prompt
set CMW_DEFAULT_LANGUAGE=en
set CMW_DEFAULT_LANGUAGE=ru

# Linux/Mac
export CMW_DEFAULT_LANGUAGE=en
export CMW_DEFAULT_LANGUAGE=ru
```

### Programmatic Usage

```python
from agent_ng.app_ng_modular import NextGenAppWithLanguageDetection

# Create app with language detection
app = NextGenAppWithLanguageDetection(language="ru")  # Russian default
demo = app.create_interface()

# Launch app
demo.launch(server_port=7860)
```

## Architecture

### Translation System

The translation system is built around three main components:

1. **Translation Dictionary** (`agent_ng/i18n_translations.py`):
   - Contains all UI text in both English and Russian
   - Provides helper functions for translation retrieval and formatting
   - Supports variable substitution in translated strings
   - Direct key lookup without Gradio i18n dependency

2. **UI Manager** (`agent_ng/ui_manager.py`):
   - Simplified UI creation without language selector
   - Uses direct translation key lookup
   - Clean interface focused on core functionality

3. **Tab Modules** (`agent_ng/tabs/`):
   - All tab modules (Chat, Logs, Stats) support i18n
   - Accept language parameter for translation context
   - Use direct translation key lookup for all user-facing text

### Key Files

- `agent_ng/i18n_translations.py` - Translation dictionaries and helper functions
- `agent_ng/app_ng_modular.py` - Main app with language support and command-line interface
- `agent_ng/ui_manager.py` - UI manager with simplified i18n support
- `agent_ng/tabs/*.py` - Tab modules with translation support
- `docs/LANGUAGE_CONFIGURATION.md` - Complete configuration guide

## Translation Keys

The translation system uses a comprehensive set of keys covering all UI elements:

### App Structure
- `app_title` - Application title
- `hero_title` - Main header title
- `tab_chat`, `tab_logs`, `tab_stats` - Tab labels

### Chat Interface
- `welcome_title`, `welcome_description` - Welcome section
- `try_asking_title`, `try_asking_examples` - Example prompts
- `quick_actions_title` - Quick actions section
- `chat_label`, `message_label`, `message_placeholder` - Chat components
- `send_button`, `clear_button` - Action buttons

### Status and Progress
- `status_title`, `status_initializing`, `status_ready` - Status indicators
- `progress_title`, `progress_ready` - Progress indicators

### Error Messages
- `error_processing`, `error_streaming` - Error messages
- `agent_not_ready`, `agent_initializing` - Agent status messages

### Statistics and Logs
- `stats_title`, `logs_title` - Tab content headers
- `agent_status_section`, `conversation_section` - Stats sections
- `token_usage_section`, `tools_section` - Detailed stats

## Adding New Translations

To add new translations:

1. **Add to Translation Dictionary**:
   ```python
   # In agent_ng/i18n_translations.py
   RUSSIAN_TRANSLATIONS["new_key"] = "Russian translation"
   ENGLISH_TRANSLATIONS["new_key"] = "English translation"
   ```

2. **Use in Code**:
   ```python
   # In UI components
   text = self._get_translation("new_key")
   
   # With variable substitution
   message = format_translation("error_message", self.language, error=error_text)
   ```

3. **Test the Translation**:
   ```python
   from agent_ng.i18n_translations import get_translation_key
   print(get_translation_key("new_key", "ru"))
   print(get_translation_key("new_key", "en"))
   ```

## Testing

Run the comprehensive test suite:

```bash
python misc_files/test_i18n_app.py
```

This tests:
- Translation dictionary functionality
- App creation in both languages
- UI component creation with translations
- Formatted translation with variables

## Port Configuration

- **Default port**: 7860
- **Auto port finding**: `--auto-port` flag finds available port automatically
- **Custom port**: `-p PORT` or `--port PORT` for specific port

## Language Priority

The language selection follows this priority order:

1. **Command line arguments** (`--ru` or `--en`) - Highest priority
2. **Environment variable** (`CMW_DEFAULT_LANGUAGE`) - Medium priority  
3. **Default** (Russian) - Lowest priority

## Current Limitations

- **No UI Language Switcher**: Language can only be changed via command line
- **No Runtime Switching**: Language is set at startup and cannot be changed during runtime
- **No Browser Detection**: Language is not automatically detected from browser settings

## Future Enhancements

- **UI Language Switcher**: Add a dropdown in the interface to switch languages
- **Browser Detection**: Automatic language detection based on browser settings
- **Session-based Language Persistence**: Remember user's language preference
- **Additional Languages**: Easy to add more languages by extending the translation dictionaries

## Best Practices

1. **Always use translation keys** instead of hardcoded text
2. **Test both languages** when adding new features
3. **Use descriptive key names** that indicate the context
4. **Support variable substitution** for dynamic content
5. **Provide fallback translations** for missing keys


## Troubleshooting

### Common Issues

1. **KeyError: 20 in Gradio Queue**:
   - This error occurs when there are orphaned event handlers
   - Solution: Restart the application to clear the queue
   - Ensure all event handlers are properly connected

2. **Missing Translation Key**:
   - Check if the key exists in both language dictionaries
   - Verify the key name spelling
   - Use `get_translation_key()` to test

3. **Translation Not Appearing**:
   - Ensure the component is using `_get_translation()`
   - Check that the language parameter is passed correctly
   - Verify the translation key exists

4. **App Creation Fails**:
   - Check that all required modules are imported
   - Verify the language parameter is valid ("en" or "ru")
   - Ensure all dependencies are installed

### Debug Mode

Enable debug logging to troubleshoot translation issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- **UI Language Switcher**: Add a dropdown in the interface to switch languages
- **Browser Detection**: Automatic language detection based on browser settings
- **Session-based Language Persistence**: Remember user's language preference
- **Additional Languages**: Easy to add more languages by extending the translation dictionaries
- **RTL Support**: Right-to-left language support for Arabic, Hebrew, etc.
- **Translation Management**: External translation management system integration
