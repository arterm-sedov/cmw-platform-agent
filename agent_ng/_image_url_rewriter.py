"""Rewrite LLM-generated inline image references to Gradio-servable /file= URLs.

When the LLM outputs markdown containing:
    <img src="llm_image_20260425_182217_65874bd8.png" alt="...">
or:
    ![alt text](llm_image_20260425_182217_65874bd8.png)

the bare filename is not served by Gradio. This module rewrites those to
fully-qualified URLs that Gradio can serve via its /gradio_api/file= endpoint.
"""

from __future__ import annotations

import re
from typing import Any

# Match HTML img tag: <img src="llm_image_*.{ext}">
_HTML_PATTERN = re.compile(
    r'<img\s+src="(llm_image_[^"]+\.(?:png|jpg|jpeg|gif|webp))"[^>]*>'
)

# Match Markdown image link: ![alt](llm_image_*.{ext})
_MD_PATTERN = re.compile(
    r'!\[([^\]]*)\]\((llm_image_[^)]+\.(?:png|jpg|jpeg|gif|webp))\)'
)


def rewrite_llm_inline_images(content: str, agent: Any) -> str:
    """Rewrite bare llm_image_* file references to Gradio-servable URLs.

    Handles both HTML and Markdown image syntax:
    - HTML:    <img src="llm_image_foo.png" alt="...">
    - Markdown: ![alt text](llm_image_foo.png)

    Args:
        content: Content that may contain LLM-generated image references.
        agent: Object with get_file_path(name) -> str | None.

    Returns:
        Content with matched image references rewritten.
    """

    if agent is None or not callable(getattr(agent, "get_file_path", None)):
        return content

    # First, rewrite Markdown links
    def md_replacer(m: re.Match[str]) -> str:
        alt_text = m.group(1)
        filename = m.group(2)
        abs_path = agent.get_file_path(filename)
        if abs_path is None:
            return m.group(0)
        # Keep the markdown syntax, replace filename with Gradio URL
        return f"![{alt_text}](/gradio_api/file={abs_path})"

    content = _MD_PATTERN.sub(md_replacer, content)

    # Then, rewrite HTML img tags
    def html_replacer(m: re.Match[str]) -> str:
        filename = m.group(1)
        abs_path = agent.get_file_path(filename)
        if abs_path is None:
            return m.group(0)
        # Replace with full Gradio URL, preserve rest of tag
        return f'<img src="/gradio_api/file={abs_path}">'

    content = _HTML_PATTERN.sub(html_replacer, content)

    return content