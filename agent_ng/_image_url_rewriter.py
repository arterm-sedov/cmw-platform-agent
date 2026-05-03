"""Rewrite bare inline image filenames to Gradio-servable ``/gradio_api/file=`` URLs.

Filenames can appear as:

- **Registered logical names** — e.g. ``llm_image_<ts>_<id>.png`` from image tools.
- **Cache basenames** — ``FileUtils.generate_unique_filename`` yields
  ``<session_stem>_<original_stem>_<ms>_<hash>.png`` (often
  ``gradio_<sid>_llm_image_….png``).
- **User uploads** registered under whatever basename showed up in ``[Files: …]``.

We only rewrite when ``agent.get_file_path(<name>)`` returns a path — so we do **not**
need a brittle whitelist of prefixes, and unrelated strings are left untouched.
"""

from __future__ import annotations

import re
from typing import Any

_BARE_IMG_BASENAME = r"[\w.-]+\.(?:png|jpg|jpeg|gif|webp)"
_IMG_TAG_RE = re.compile(r"(?is)<img\b[^>]*>")
_SRC_DOUBLE = re.compile(rf'(?is)\bsrc\s*=\s*"({_BARE_IMG_BASENAME})"')


def _is_skip_src(value: str) -> bool:
    v = value.strip()
    lowered = v.lower()
    return lowered.startswith(
        ("http://", "https://", "/gradio_api/file=", "//", "data:")
    )


def rewrite_llm_inline_images(content: str, agent: Any) -> str:
    """Rewrite session-registered bare image ``src`` / markdown targets.

    Skips URLs (http(s), protocol-relative), ``data:``, and paths already using
    ``/gradio_api/file=``.
    """

    if agent is None or not callable(getattr(agent, "get_file_path", None)):
        return content

    def md_replacer(m: re.Match[str]) -> str:
        alt_text = m.group(1)
        basename = m.group(2)
        if _is_skip_src(basename):
            return m.group(0)
        abs_path = agent.get_file_path(basename)
        if abs_path is None:
            return m.group(0)
        return f"![{alt_text}](/gradio_api/file={abs_path})"

    _md_pat = re.compile(
        rf"!\[([^\]]*)]\(({_BARE_IMG_BASENAME})\)", flags=re.IGNORECASE
    )
    content = _md_pat.sub(md_replacer, content)

    def rewrite_one_img_tag(match: re.Match[str]) -> str:
        fragment = match.group(0)

        def src_repl(s: re.Match[str]) -> str:
            basename = s.group(1)
            if _is_skip_src(basename):
                return s.group(0)
            abs_path = agent.get_file_path(basename)
            if abs_path is None:
                return s.group(0)
            return f'src="/gradio_api/file={abs_path}"'

        return _SRC_DOUBLE.sub(src_repl, fragment)

    return _IMG_TAG_RE.sub(rewrite_one_img_tag, content)
