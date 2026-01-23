"""
Dead Code: Language Detection Functions
======================================

These language detection functions were defined but never called anywhere in the codebase.
They can be safely removed unless needed for future language detection functionality.

Extracted from:
- agent_ng/app_ng_modular.py

Functions:
- detect_language_from_url() - Never called
- detect_language() - Never called  
- get_demo() - Only called once (legacy support)
- create_safe_demo() - Only called once (legacy support)
"""

import gradio as gr
import logging

_logger = logging.getLogger(__name__)

def detect_language_from_url(self):
    """Detect language from URL parameters using environment variables or sys.argv"""
    try:
        # Check if we're running with Gradio and can access URL parameters
        import os
        import sys

        # Check for language parameter in command line arguments
        for i, arg in enumerate(sys.argv):
            if arg == '--lang' and i + 1 < len(sys.argv):
                lang = sys.argv[i + 1].lower()
                if lang in self.supported_languages:
                    print(f"🌐 Language detected from command line: {lang}")
                    return lang
            elif arg.startswith('--lang='):
                lang = arg.split('=')[1].lower()
                if lang in self.supported_languages:
                    print(f"🌐 Language detected from command line: {lang}")
                    return lang

        # Check environment variable (can be set by Gradio)
        lang_env = os.environ.get('GRADIO_LANG', '').lower()
        if lang_env in self.supported_languages:
            print(f"🌐 Language detected from environment: {lang_env}")
            return lang_env

        # Check for URL parameter in environment (Gradio might set this)
        url_lang = os.environ.get('LANG_PARAM', '').lower()
        if url_lang in self.supported_languages:
            print(f"🌐 Language detected from URL parameter: {url_lang}")
            return url_lang

        return None

    except Exception as e:
        print(f"⚠️ URL language detection failed: {e}")
        return None

def detect_language(self, request=None):
    """Detect language from URL parameters, headers, or browser settings"""
    # Default to Russian
    detected_lang = "ru"

    try:
        # Check URL parameters first
        if request and hasattr(request, 'quick_params'):
            lang_param = request.quick_params.get('lang', '').lower()
            if lang_param in self.supported_languages:
                detected_lang = lang_param
                print(f"🌐 Language detected from URL parameter: {detected_lang}")
                return detected_lang

        # Check Accept-Language header
        if request and hasattr(request, 'headers'):
            accept_lang = request.headers.get('Accept-Language', '')
            if 'ru' in accept_lang.lower():
                detected_lang = "ru"
                print(f"🌐 Language detected from browser: {detected_lang}")
                return detected_lang

        # Fallback to default
        print(f"🌐 Using default language: {detected_lang}")
        return detected_lang

    except Exception as e:
        print(f"⚠️ Language detection failed: {e}, using default: {detected_lang}")
        return detected_lang

def get_demo(language: str = "en"):
    """Get or create the demo interface for the specified language (legacy support)"""
    # For backward compatibility, create language-specific demo
    try:
        # Note: This would need NextGenApp import to work
        # app = NextGenApp(language=language)
        # return app.create_interface()
        pass
    except Exception as e:
        _logger.exception("Error creating demo for language %s: %s", language, e)
        # Create a minimal working demo to prevent KeyError
        with gr.Blocks() as demo:
            gr.Markdown("# CMW Platform Agent")
            gr.Markdown("Application is initializing...")
        return demo

def create_safe_demo():
    """Create a safe demo instance that won't cause reload errors"""
    try:
        demo_instance = get_demo("en")  # Default to English
        return demo_instance
    except Exception as e:
        _logger.exception("Error creating safe demo: %s", e)
        # Create a minimal working demo to prevent KeyError
        with gr.Blocks() as demo:
            gr.Markdown("# CMW Platform Agent")
            gr.Markdown("Application is initializing...")
        return demo
