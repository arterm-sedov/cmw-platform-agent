# Language Switching Mechanics (CMW Platform Agent)

## Overview

This report documents how the application selects and switches UI language across modules in `agent_ng`. It covers configuration priority, runtime behavior, translation sources, and current limitations.

## Configuration and Priority

- Command-line flags: `--en` / `--ru` override all other sources.
- Environment variable: `CMW_DEFAULT_LANGUAGE` sets the default when CLI flags are not provided.
- Config default: `agent_ng/agent_config.py` initializes `default_language` (currently "ru").
- Gradio env: `GRADIO_DEFAULT_LANGUAGE` is consulted by the runtime detection helper to align with Gradio's i18n selection.

Priority as implemented at startup:

1) CLI flags (`--ru` / `--en`)
2) `GRADIO_DEFAULT_LANGUAGE` (via runtime detection)
3) `CMW_DEFAULT_LANGUAGE` (via config)
4) Code default (ru)

## Startup and Detection

- `NextGenAppWithLanguageDetection.get_current_language()` reads `GRADIO_DEFAULT_LANGUAGE` first; if unavailable, it queries a minimal Gradio `I18n` instance; on failure falls back to "ru".
- `main()` computes the effective language by combining config defaults, detection, and CLI flags, and logs the final selection.

Relevant code references:

```1318:1346:agent_ng/app_ng_modular.py
    def get_current_language(self, request: gr.Request = None) -> str:
        """
        Get current language using Gradio's I18n system with GRADIO_DEFAULT_LANGUAGE as primary source.
        """
        try:
            # Primary: Use GRADIO_DEFAULT_LANGUAGE environment variable
            import os
            from dotenv import load_dotenv
            load_dotenv()  # Load .env file
            gradio_lang = os.getenv("GRADIO_DEFAULT_LANGUAGE", "").lower()
            if gradio_lang is not None:
                return gradio_lang
            # Secondary: Use Gradio's I18n system for detection
            i18n = gr.I18n(en={"language": "en"}, ru={"language": "ru"})
            detected_language = i18n("language")
            if detected_language is not None:
                return detected_language
        except Exception:
            return "ru"
```

```1493:1514:agent_ng/app_ng_modular.py
    language = language_settings["default_language"]
    try:
        temp_app = NextGenAppWithLanguageDetection()
        detected_language = temp_app.get_current_language()
        if detected_language in ["en", "ru"]:
            language = detected_language
    except Exception as e:
        ...
    if args.russian:
        language = "ru"
    elif args.english:
        language = "en"
```

```1466:1471:agent_ng/app_ng_modular.py
    parser.add_argument(
        "-en", "--english", action="store_true", help="Start in English"
    )
    parser.add_argument(
        "-ru", "--russian", action="store_true", help="Start in Russian"
    )
```

```66:67:agent_ng/agent_config.py
        if os.getenv('CMW_DEFAULT_LANGUAGE'):
            self.settings.default_language = os.getenv('CMW_DEFAULT_LANGUAGE')
```

## Translations and Resolution

- Dictionaries live in `agent_ng/i18n_translations.py` for RU/EN. A `gr.I18n` instance can be constructed via `create_i18n_instance()`.
- UI components call `_get_translation(key)` which prefers the Gradio `I18n` instance (if provided) and falls back to direct dictionary lookup via `get_translation_key(key, language)`.
- `UIManager` maintains per-language singletons via `get_ui_manager(language=...)`.

```622:631:agent_ng/i18n_translations.py
def create_i18n_instance() -> gr.I18n:
    return gr.I18n(en=ENGLISH_TRANSLATIONS, ru=RUSSIAN_TRANSLATIONS)
```

```241:246:agent_ng/ui_manager.py
        return get_translation_key(key, self.language)

# Global instances for different languages
_ui_manager_en = None
_ui_manager_ru = None
```

```247:267:agent_ng/ui_manager.py
def get_ui_manager(language: str = "en", i18n_instance: gr.I18n | None = None) -> UIManager:
    global _ui_manager_en, _ui_manager_ru
    if language.lower() == "ru":
        if _ui_manager_ru is None:
            _ui_manager_ru = UIManager(language="ru", i18n_instance=i18n_instance)
        return _ui_manager_ru
    else:
        if _ui_manager_en is None:
            _ui_manager_en = UIManager(language="en", i18n_instance=i18n_instance)
        return _ui_manager_en
```

```97:110:agent_ng/tabs/home_tab.py
    def _get_translation(self, key: str, **kwargs: Any) -> str:
        if self.i18n:
            try:
                return self.i18n.t(key, **kwargs)
            except Exception as e:
                ...
        if get_translation_key:
            return get_translation_key(key, self.language)
        return f"[{key}]"
```

## Runtime Switching Status

- There is currently no in-UI language selector (no dropdown bound to language). Language is effectively chosen at startup based on the priority above.
- Documentation note suggests a language selector could be added; present implementation does not expose it yet.

Evidence:

```186:195:docs/migrate_to_lanchain_analysis/I18N_IMPLEMENTATION.md
- **No UI Language Switcher**: Language can only be changed via command line
- **No Runtime Switching**: Language is set at startup and cannot be changed during runtime
- **No Browser Detection**: Language is not automatically detected from browser settings
```

## How to Change Language

- CLI: `python agent_ng/app_ng_modular.py --en` or `--ru`
- Env: `CMW_DEFAULT_LANGUAGE=en` (or `ru`) before launch
- Gradio env: `GRADIO_DEFAULT_LANGUAGE=en` (or `ru`) to influence detection

## Recommendations

- Add a language dropdown to persist selection per session and trigger a UI rebuild with the chosen locale.
- Store selection in session state and propagate to `UIManager` for deterministic translations.
- Keep CLI/env precedence for headless deployments; prefer explicit UI control for interactive usage.
