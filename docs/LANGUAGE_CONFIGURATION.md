# Language Configuration

## Default Language Settings

The CMW Platform Agent now defaults to **Russian** language. You can configure the default language in several ways:

### 1. Environment Variable (Recommended)

Set the `CMW_DEFAULT_LANGUAGE` environment variable:

```bash
# Windows PowerShell
$env:CMW_DEFAULT_LANGUAGE="ru"  # Russian
$env:CMW_DEFAULT_LANGUAGE="en"  # English

# Windows Command Prompt
set CMW_DEFAULT_LANGUAGE=ru
set CMW_DEFAULT_LANGUAGE=en

# Linux/Mac
export CMW_DEFAULT_LANGUAGE=ru
export CMW_DEFAULT_LANGUAGE=en
```

### 2. Command Line Arguments (Override Environment)

```bash
# Force English (overrides environment)
python agent_ng/app_ng_modular.py --en

# Force Russian (overrides environment)  
python agent_ng/app_ng_modular.py --ru

# Russian with auto port finding
python agent_ng/app_ng_modular.py --ru --auto-port
```

### 3. Priority Order

1. **Command line arguments** (`--ru` or `--en`)
2. **Environment variable** (`CMW_DEFAULT_LANGUAGE`)
3. **Default** (Russian)

### 4. Examples

```bash
# Default behavior (Russian)
python agent_ng/app_ng_modular.py

# English via environment variable
$env:CMW_DEFAULT_LANGUAGE="en"
python agent_ng/app_ng_modular.py

# Override environment with command line
$env:CMW_DEFAULT_LANGUAGE="en"
python agent_ng/app_ng_modular.py --ru  # Will start in Russian

# Auto port finding
python agent_ng/app_ng_modular.py --ru --auto-port
```

## Runtime Language Switching

Once the application is running, users can switch languages using the language selector in the UI, regardless of the startup language.
