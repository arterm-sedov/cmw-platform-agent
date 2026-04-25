"""
Standalone smoke server for validating inline image / file rendering in
``gr.Chatbot(type="messages")`` on Gradio 5.49 (Windows).

Hypothesis under test
---------------------
The dict-content form ``{"role": ..., "content": {"path": <abs>, "alt_text": ...}}``
renders a file inline in a chat bubble, picking the correct renderer (image,
file-chip, etc.) from the extension. This mirrors the production change we plan
to make in ``agent_ng/native_langchain_streaming.py`` + ``agent_ng/app_ng_modular.py``
so we can validate path handling, allowed_paths and Windows quirks in isolation
before touching production code.

Scenarios covered
-----------------
1. Baseline — text-only "Tool called" accordion (parity with today).
2. Assistant-side image from ``docs/image_generation/progress_reports/*.png``.
3. Assistant-side image from ``GRADIO_TEMP_DIR`` (the same bear the real tool wrote).
4. User-side image rendering (what we plan to mirror in chat_tab.py).
5. Non-image file (``*.md`` — proves the "generic" claim for PDFs etc.).
6. Missing file — graceful degradation check.
7. Large image (>1 MB) — performance sanity.
8. Streamed sequence — tool-call accordion → image bubble → caption → LLM text,
   simulating the full flow we'd see after the production change lands.

Run
---
    .venv\\Scripts\\python.exe docs\\image_generation\\smoke_chat_inline_images.py

Then open http://localhost:7870. Each button appends its scenario to the chat.
"""

from __future__ import annotations

import socket
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.file_utils import FileUtils  # noqa: E402

# -----------------------------------------------------------------------------#
# Resources                                                                    #
# -----------------------------------------------------------------------------#

_PROGRESS = _REPO / "docs" / "image_generation" / "progress_reports"
_CACHE_DIR = Path(FileUtils.get_gradio_cache_path())

# Fixtures (all resolved at import time so any missing file is flagged early).
_IMG_SMALL = _PROGRESS / "20260425_google__gemini-2.5-flash-image_russian.png"
_IMG_BIG = _PROGRESS / "20260425_google__gemini-3-pro-image-preview_russian.png"
_IMG_BEAR = (
    _PROGRESS
    / "gradio_h45vn38nsye_llm_image_20260425_160512_0fec191a_1777122312910_aeb49ea1.png"
)
_MD_FILE = _PROGRESS / "20260425_model_comparison.md"
_MISSING = _PROGRESS / "this_file_does_not_exist_yet.png"


def _fmt_size(num_bytes: int) -> str:
    """Cheap human-readable size, matching the style used by ``[Files: …]``."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{num_bytes} B"
        num_bytes /= 1024  # type: ignore[assignment]
    return f"{num_bytes:.1f} GB"


def _caption_for(path: Path) -> str:
    """Build a ``📎 name — size`` caption line (same shape as user uploads)."""
    try:
        size = path.stat().st_size
    except OSError:
        return f"📎 {path.name}"
    return f"📎 {path.name} — {_fmt_size(size)}"


def _file_msg(role: str, path: Path) -> dict:
    """One message carrying a file via dict-content, Windows-safe absolute path."""
    # Normalize via resolve() so Gradio's path-check accepts the Windows form.
    abs_path = str(path.resolve())
    return {
        "role": role,
        "content": {"path": abs_path, "alt_text": path.name},
    }


def _text(role: str, content: str, title: str | None = None) -> dict:
    """Plain text message, optionally with a metadata accordion title."""
    msg: dict = {"role": role, "content": content}
    if title:
        msg["metadata"] = {"title": title}
    return msg


# -----------------------------------------------------------------------------#
# Scenario handlers                                                            #
# -----------------------------------------------------------------------------#


def scenario_baseline(history: list[dict]) -> list[dict]:
    """Scenario 1: text-only — parity with today's rendering."""
    history = list(history or [])
    history.append(_text("user", "Расскажи, какие приложения есть?"))
    history.append(
        _text(
            "assistant",
            "Результат: {'success': True, 'applications': [...]} ... [truncated]",
            title="Вызван инструмент: list_applications",
        )
    )
    history.append(
        _text("assistant", "В системе есть 3 приложения: X, Y, Z.")
    )
    return history


def scenario_assistant_image_from_repo(history: list[dict]) -> list[dict]:
    """Scenario 2: assistant renders an image from the repo folder."""
    history = list(history or [])
    history.append(_text("user", "Сгенерируй картинку: значок рабочего процесса"))
    history.append(
        _text(
            "assistant",
            "Результат: {'success': True, 'file_reference': '"
            + _IMG_SMALL.name
            + "', 'cost': 0.067}",
            title="Вызван инструмент: generate_ai_image",
        )
    )
    history.append(_file_msg("assistant", _IMG_SMALL))
    history.append(_text("assistant", _caption_for(_IMG_SMALL)))
    history.append(_text("assistant", "Готово! 🎨 Вот значок рабочего процесса."))
    return history


def scenario_assistant_image_from_tempdir(history: list[dict]) -> list[dict]:
    """Scenario 3: image whose path is inside GRADIO_TEMP_DIR.

    Uses the actual bear PNG Gradio wrote during a real generate_ai_image call —
    we copied it into progress_reports, but for this scenario we simulate the
    prod case where it still lives in the cache dir.
    """
    history = list(history or [])
    # Copy bear into the cache dir so the path really is under GRADIO_TEMP_DIR.
    target = _CACHE_DIR / _IMG_BEAR.name
    if not target.exists():
        try:
            target.write_bytes(_IMG_BEAR.read_bytes())
        except OSError as exc:
            history.append(_text("assistant", f"(copy failed: {exc})"))
            return history
    history.append(_text("user", "Сгенерируй простую картинку медведя"))
    history.append(
        _text(
            "assistant",
            "Результат: {'success': True, 'file_reference': '"
            + target.name
            + "', 'cost': 0.067}",
            title="Вызван инструмент: generate_ai_image",
        )
    )
    history.append(_file_msg("assistant", target))
    history.append(_text("assistant", _caption_for(target)))
    history.append(_text("assistant", "Готово! 🐻"))
    return history


def scenario_user_image(history: list[dict]) -> list[dict]:
    """Scenario 4: symmetric user-side rendering (what we plan to add to chat_tab.py)."""
    history = list(history or [])
    history.append(_file_msg("user", _IMG_SMALL))
    history.append(
        _text(
            "user",
            f"Опиши это изображение.\n[Files: {_IMG_SMALL.name} ({_fmt_size(_IMG_SMALL.stat().st_size)})]",
        )
    )
    history.append(
        _text(
            "assistant",
            "Результат: {'success': True, 'result': 'Минималистичный значок...'}",
            title="Вызван инструмент: analyze_image_ai",
        )
    )
    history.append(
        _text(
            "assistant",
            "На изображении — минималистичный значок рабочего процесса в корпоративном стиле.",
        )
    )
    return history


def scenario_non_image_file(history: list[dict]) -> list[dict]:
    """Scenario 5: non-image file renders as a download chip (generic path)."""
    history = list(history or [])
    history.append(_text("user", "Сравнение моделей — покажи отчёт"))
    history.append(
        _text(
            "assistant",
            "Результат: {'success': True, 'file_reference': '"
            + _MD_FILE.name
            + "'}",
            title="Вызван инструмент: export_report",
        )
    )
    history.append(_file_msg("assistant", _MD_FILE))
    history.append(_text("assistant", _caption_for(_MD_FILE)))
    history.append(_text("assistant", "Готово, отчёт сохранён."))
    return history


def scenario_missing_file(history: list[dict]) -> list[dict]:
    """Scenario 6: broken path — confirms graceful degradation."""
    history = list(history or [])
    history.append(_text("user", "Покажи несуществующую картинку"))
    history.append(_file_msg("assistant", _MISSING))
    history.append(_text("assistant", _caption_for(_MISSING)))
    history.append(
        _text(
            "assistant",
            "(ожидаемое поведение: Gradio показывает заглушку или 404)",
        )
    )
    return history


def scenario_large_image(history: list[dict]) -> list[dict]:
    """Scenario 7: ~2 MB PNG — performance sanity check."""
    history = list(history or [])
    history.append(_text("user", "Сгенерируй большое hero-изображение"))
    history.append(
        _text(
            "assistant",
            "Результат: {'success': True, 'file_reference': '"
            + _IMG_BIG.name
            + "', 'cost': 0.135}",
            title="Вызван инструмент: generate_ai_image",
        )
    )
    history.append(_file_msg("assistant", _IMG_BIG))
    history.append(_text("assistant", _caption_for(_IMG_BIG)))
    history.append(_text("assistant", "Готово, это версия 1408×768."))
    return history


def scenario_streamed(history: list[dict]):
    """Scenario 8: streamed sequence — closest simulation of the prod flow.

    Yields the history step by step so the user sees the exact order:
    user ask → tool-called accordion → image bubble → caption → LLM reply.
    Small sleeps mimic network/LLM latency.
    """
    history = list(history or [])

    history.append(_text("user", "Сгенерируй простую картинку медведя"))
    yield history
    time.sleep(0.4)

    history.append(
        _text(
            "assistant",
            "Количество вызовов: 1\n\nРезультат: {'success': True, "
            "'file_reference': '"
            + _IMG_BEAR.name
            + "', 'cost': 0.067, 'mime_type': 'image/png', "
            "'size_bytes': 2019542}",
            title="Вызван инструмент: generate_ai_image",
        )
    )
    yield history
    time.sleep(0.6)

    history.append(_file_msg("assistant", _IMG_BEAR))
    yield history
    time.sleep(0.2)

    history.append(_text("assistant", _caption_for(_IMG_BEAR)))
    yield history
    time.sleep(0.3)

    # Simulate character-by-character LLM response for realism.
    reply = "Готово! 🐻 Сгенерировал простую картинку медведя."
    running = ""
    assistant_idx = len(history)
    history.append(_text("assistant", ""))
    for ch in reply:
        running += ch
        history[assistant_idx] = _text("assistant", running)
        yield history
        time.sleep(0.015)


# -----------------------------------------------------------------------------#
# Build the UI                                                                 #
# -----------------------------------------------------------------------------#


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Smoke: inline chat images") as demo:
        gr.Markdown(
            "# Smoke test — inline image/file rendering\n\n"
            "Each button appends one scenario to the chat. History accumulates "
            "so you can compare scenarios side by side.\n\n"
            f"- **Gradio temp dir:** `{_CACHE_DIR}`\n"
            f"- **Repo fixtures:** `{_PROGRESS}`\n"
        )

        chatbot = gr.Chatbot(
            label="Chat",
            height=600,
            show_label=True,
            container=True,
            show_copy_button=True,
            type="messages",
            elem_id="smoke-chatbot",
        )

        with gr.Row():
            b1 = gr.Button("1. Baseline (text only)")
            b2 = gr.Button("2. Assistant image (repo path)")
            b3 = gr.Button("3. Assistant image (cache path)")
            b4 = gr.Button("4. User image upload")
        with gr.Row():
            b5 = gr.Button("5. Non-image file (.md)")
            b6 = gr.Button("6. Missing file")
            b7 = gr.Button("7. Large image (2 MB)")
            b8 = gr.Button("8. Streamed sequence", variant="primary")
        with gr.Row():
            reset = gr.Button("Clear chat", variant="secondary")

        b1.click(scenario_baseline, inputs=chatbot, outputs=chatbot)
        b2.click(scenario_assistant_image_from_repo, inputs=chatbot, outputs=chatbot)
        b3.click(scenario_assistant_image_from_tempdir, inputs=chatbot, outputs=chatbot)
        b4.click(scenario_user_image, inputs=chatbot, outputs=chatbot)
        b5.click(scenario_non_image_file, inputs=chatbot, outputs=chatbot)
        b6.click(scenario_missing_file, inputs=chatbot, outputs=chatbot)
        b7.click(scenario_large_image, inputs=chatbot, outputs=chatbot)
        b8.click(scenario_streamed, inputs=chatbot, outputs=chatbot)
        reset.click(lambda: [], inputs=None, outputs=chatbot)

    return demo


def _first_free_port(candidates: range) -> int:
    """Return the first port in ``candidates`` that is free on localhost."""
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    msg = f"no free port in {candidates}"
    raise OSError(msg)


def main() -> None:
    # Sanity-check fixture existence so failures are loud, not subtle.
    for label, path in [
        ("IMG_SMALL", _IMG_SMALL),
        ("IMG_BIG", _IMG_BIG),
        ("IMG_BEAR", _IMG_BEAR),
        ("MD_FILE", _MD_FILE),
    ]:
        status = "OK" if path.exists() else "MISSING"
        size = _fmt_size(path.stat().st_size) if path.exists() else "-"
        print(f"  [{status}] {label:<10} {size:>10}  {path}")

    port = _first_free_port(range(7870, 7880))
    print(f"\nStarting smoke server on http://127.0.0.1:{port}\n")

    demo = build_demo()
    # Allow both the repo progress_reports dir AND the Gradio cache dir so
    # paths from either location serve without 403.
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        show_error=True,
        allowed_paths=[str(_PROGRESS), str(_CACHE_DIR), tempfile.gettempdir()],
        inbrowser=False,
        prevent_thread_lock=False,
    )


if __name__ == "__main__":
    main()
