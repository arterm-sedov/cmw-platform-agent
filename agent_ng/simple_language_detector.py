"""
Simple Language Detector using Gradio I18n
==========================================

This module provides a simple way to detect the current interface language
using Gradio's native I18n system with a fake "language" resource.
"""

import gradio as gr

# Create I18n instance with fake "language" resource for detection
i18n = gr.I18n(
    en={"language": "en"},
    ru={"language": "ru"}
)

def get_current_language(request: gr.Request = None) -> str:  # noqa: ARG001
    """
    Get current language using Gradio's I18n system.

    Args:
        request: Gradio request object (optional)

    Returns:
        Language code ('en' or 'ru')
    """
    try:
        return i18n("language")
    except Exception:
        return "en"
