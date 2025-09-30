"""
Dead Code: App Functions
=======================

These app-related functions were defined but never called anywhere in the codebase.
They can be safely removed unless needed for future app functionality.

Extracted from:
- agent_ng/app_ng_modular.py
"""

# From agent_ng/app_ng_modular.py
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

def create_safe_demo():
    """Create a safe demo instance that won't cause reload errors"""
    try:
        demo_instance = get_demo("en")  # Default to English
        # Ensure the demo has the required attributes for Gradio reloading
        if not hasattr(demo_instance, '_queue'):
            demo_instance._queue = None
        return demo_instance
    except Exception as e:
        _logger.exception("Error creating safe demo: %s", e)
        # Create a minimal working demo with required attributes
        with gr.Blocks() as demo:
            gr.Markdown("# CMW Platform Agent")
            gr.Markdown("Application is initializing...")
        # Ensure required attributes exist
        if not hasattr(demo, '_queue'):
            demo._queue = None
        return demo

def reload_demo():
    """Reload the demo for Gradio hot reloading"""
    global demo
    try:
        _logger.info("Reloading demo...")
        demo = get_demo_with_language_detection()
        return demo
    except Exception as e:
        _logger.exception("Error reloading demo: %s", e)
        # Return current demo or create minimal one
        if demo is None:
            with gr.Blocks() as fallback_demo:
                gr.Markdown("# CMW Platform Agent")
                gr.Markdown("Error during reload - please restart the application")
            return fallback_demo
        return demo
