r"""Minimal Gradio smoke server — validates /file= URL serving for inline images.

This server writes real PNG files to the filesystem and renders chatbot messages
with three kinds of img-src references:

  1. Bare filename   <img src="llm_image_foo.png">         — Gradio needs allowed_paths
  2. /file=<abs>     <img src="/file=C:\TEMP\...\foo.png">  — our rewriter produces this
  3. http://...      <img src="http://host/gradio_api/...">  — what Gradio serves

By inspecting browser DevTools we can confirm which forms Gradio rewrites
to working URLs and which result in broken images.

Run from the repo root:
    .venv\Scripts\python.exe docs\image_generation/smoke_inline_url_rewrite.py
"""

from __future__ import annotations

import socket
import struct
import sys
import zlib
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.file_utils import FileUtils  # noqa: E402


# ---------------------------------------------------------------------------#
# PNG fixture creation                                                        #
# ---------------------------------------------------------------------------#


def _make_minimal_png(
    width: int = 1, height: int = 1, r: int = 30, g: int = 90, b: int = 150
) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw_row = b"\x00" + bytes([r, g, b] * width)
    idat_data = zlib.compress(raw_row * height)
    return sig + chunk(b"IHDR", ihdr_data) + chunk(b"IDAT", idat_data) + chunk(b"IEND", b"")


_CACHE_DIR = Path(FileUtils.get_gradio_cache_path())
_SMOKE_DIR = _CACHE_DIR / "smoke_inline_rewrite"
_SMOKE_DIR.mkdir(parents=True, exist_ok=True)

_PNG_SMALL = _SMOKE_DIR / "llm_image_smoke_test.png"
if not _PNG_SMALL.exists():
    _PNG_SMALL.write_bytes(_make_minimal_png())


# ---------------------------------------------------------------------------#
# Message helpers                                                             #
# ---------------------------------------------------------------------------#


def _text_msg(role: str, content: str, title: str | None = None) -> dict:
    msg: dict = {"role": role, "content": content}
    if title:
        msg["metadata"] = {"title": title}
    return msg


# ---------------------------------------------------------------------------#
# Scenario helpers                                                            #
# ---------------------------------------------------------------------------#


def scenario_bare_filename(history: list[dict]) -> list[dict]:
    """Scenario 1: bare <img src="llm_image_*.png"> in markdown string."""
    history = list(history or [])
    history.append(_text_msg("user", "Покажи картинку"))
    history.append(_text_msg(
        "assistant",
        "# Результат\n\n"
        '<img src="llm_image_smoke_test.png" alt="Smoke test">\n\n'
        "Если img выше не сломалась — Gradio нашёл её в allowed_paths.",
        title="Вызван инструмент: generate_ai_image",
    ))
    return history


def scenario_file_url(history: list[dict]) -> list[dict]:
    """Scenario 2: /file= absolute URL form — what our rewriter produces."""
    history = list(history or [])
    history.append(_text_msg("user", "Покажи картинку (file= URL)"))
    history.append(_text_msg(
        "assistant",
        "# Результат\n\n"
        f'<img src="/file={_PNG_SMALL}" alt="Smoke test /file= form">\n\n'
        "Этот src должен сразу работать (не требует allowed_paths).",
        title="Вызван инструмент: generate_ai_image",
    ))
    return history


def scenario_mixed(history: list[dict]) -> list[dict]:
    """Scenario 3: both forms in one message — verify rewrite target."""
    history = list(history or [])
    history.append(_text_msg("user", "Смешанный режим"))
    history.append(_text_msg(
        "assistant",
        "# Mixed\n\n"
        'bare: <img src="llm_image_smoke_test.png">\n'
        f'/file=: <img src="/file={_PNG_SMALL}">\n\n'
        "Inspect DevTools — первый img сломается, второй должен быть OK.",
    ))
    return history


# ---------------------------------------------------------------------------#
# Build UI                                                                    #
# ---------------------------------------------------------------------------#


def _free_port() -> int:
    for port in range(7870, 7880):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise OSError("No free port in 7870-7879")


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Inline URL Rewrite Smoke") as demo:
        gr.Markdown(
            "## /file= URL Rewrite Smoke Test\n\n"
            "Three scenarios — inspect DevTools → Elements → find the `<img` tags.  "
            "The `src` attribute shows which URL form Gradio accepts.\n\n"
            f"- **Gradio cache:** `{_CACHE_DIR}`\n"
            f"- **Smoke dir:** `{_SMOKE_DIR}`\n"
            f"- **Smoke PNG:** `{_PNG_SMALL.name}`\n"
        )

        chatbot = gr.Chatbot(
            type="messages",
            height=500,
            show_copy_button=True,
            elem_id="smoke-chatbot",
        )

        with gr.Row():
            b1 = gr.Button("1. Bare filename")
            b2 = gr.Button("2. /file= URL")
            b3 = gr.Button("3. Mixed")
            reset = gr.Button("Clear")

        b1.click(scenario_bare_filename, inputs=chatbot, outputs=chatbot)
        b2.click(scenario_file_url, inputs=chatbot, outputs=chatbot)
        b3.click(scenario_mixed, inputs=chatbot, outputs=chatbot)
        reset.click(lambda: [], inputs=None, outputs=chatbot)

        gr.Markdown(
            "### DevTools inspection guide\n\n"
            "1. Open browser DevTools (F12) → Elements tab\n"
            "2. Search for `llm_image_smoke_test`\n"
            "3. Check the `src` attribute:\n"
            "   - `http://localhost:PORT/gradio_api/file=...` → **working**\n"
            "   - `llm_image_smoke_test.png` (unchanged) → **broken**\n"
            "   - `/file=...` (unchanged) → Gradio passthrough, **may be broken**\n"
        )

    return demo


# ---------------------------------------------------------------------------#
# Entry point                                                                 #
# ---------------------------------------------------------------------------#


def main() -> None:
    port = _free_port()
    demo = build_demo()

    print(f"Cache dir : {_CACHE_DIR}")
    print(f"Smoke dir: {_SMOKE_DIR}")
    print(f"Image    : {_PNG_SMALL.name}  ({_PNG_SMALL.stat().st_size} B)")
    print(f"Port     : {port}")
    print()
    print(f"Open http://localhost:{port}")
    print()
    print("Click each button, then inspect DevTools → Elements for the img src.")
    print("Press Ctrl+C to stop.")

    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        show_error=True,
        allowed_paths=[str(_CACHE_DIR), str(_SMOKE_DIR)],
        prevent_thread_lock=False,
    )


if __name__ == "__main__":
    main()
