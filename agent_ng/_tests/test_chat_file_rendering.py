"""
Tests for inline file/image rendering in the Gradio chat.

Behavior contracts
------------------

Streaming layer (native_langchain_streaming.py):
- When a tool returns ``{"success": True, "file_reference": "<name>", ...}``,
  the ``tool_end`` event's ``metadata`` gains a ``file_attachment`` dict:
  ``{"path": <abs_path>, "display_name": <name>, "size_bytes": <int>}``.
- ``file_attachment`` is absent (or None) when:
  - the tool result has ``success=False``,
  - the result carries no ``file_reference``,
  - the ``file_reference`` can't be resolved (no agent, or file missing on disk),
  - the tool returns a non-dict (string, None).
- The existing ``content`` / ``tool_output`` strings are unaffected — the LLM
  still sees the same stringified dict it always did.

App rendering layer (app_ng_modular.py):
- When ``metadata["file_attachment"]`` is present and the path resolves to an
  existing file, **two** messages are appended after the tool-called accordion:
  1. ``{"role": "assistant", "content": {"path": <abs>, "alt_text": <name>}}``
     — the inline file/image bubble Gradio 5 renders natively.
  2. ``{"role": "assistant", "content": "📎 <display_name> — <size>"}``
     — a caption line matching the style used for user-uploaded files.
- When the path does NOT exist on disk, skip both extra messages (no error).
- When ``file_attachment`` is absent, behaviour is identical to today (no
  regression).

User-side (chat_tab.py):
- Uploaded image files gain an inline ``{"role": "user", "content": {"path":
  <abs>, "alt_text": <name>}}`` message before the ``[Files: …]`` text.
- Non-image uploads (PDF, xlsx, …) use the same dict — Gradio shows them as
  file-chips.
- Text-only messages are unaffected.

Run:  pytest agent_ng/_tests/test_chat_file_rendering.py -v
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

from agent_ng._file_attachment import (
    build_file_attachment,
    build_file_bubbles,
    build_file_bubbles_for_role,
    is_file_bubble,
)
from agent_ng.token_counter import (
    ConversationTokenTracker,
    convert_chat_history_to_messages,
)

# Fixture path — hard-coded for tests only; not a real runtime path.
_FIXTURE_PNG = "/tmp/x.png"  # noqa: S108
_FIXTURE_BEAR = "/tmp/bear.png"  # noqa: S108

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------#
# Helpers — build minimal mock objects that match the production interface   #
# ---------------------------------------------------------------------------#

def _make_agent(session_id: str = "sess-render-1") -> Any:
    """Minimal CmwAgent stub for tests that resolve file paths."""

    class _Agent:
        def __init__(self) -> None:
            self.session_id = session_id
            self._registry: dict[tuple[str, str], str] = {}

        def register_file(self, name: str, path: str) -> None:
            self._registry[(self.session_id, name)] = path

        def get_file_path(self, name: str) -> str | None:
            p = self._registry.get((self.session_id, name))
            return p if p and os.path.isfile(p) else None

    return _Agent()


def _make_png(tmp_path: Path, name: str = "test.png") -> Path:
    """Write a minimal valid PNG (1x1 pixel) so ``Path.exists()`` passes."""
    # 67-byte valid PNG
    data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
        b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _fmt_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{num_bytes} B"
        num_bytes /= 1024  # type: ignore[assignment]
    return f"{num_bytes:.1f} GB"


# ---------------------------------------------------------------------------#
# Unit: file-attachment helper extracted from streaming layer               #
# ---------------------------------------------------------------------------#
# We test the helper function directly so we can import it without spinning up
# the full streaming pipeline.  The function will live in a new helper module
# ``agent_ng/_file_attachment.py`` (thin, importable by both streaming and app).

class TestBuildFileAttachment:
    """Contracts for ``_build_file_attachment(tool_result, agent)``."""

    def test_success_with_file_reference_and_existing_file(
        self, tmp_path: Path
    ) -> None:
        png = _make_png(tmp_path, "llm_image_123.png")
        agent = _make_agent()
        agent.register_file("llm_image_123.png", str(png))

        result = {"success": True, "file_reference": "llm_image_123.png"}
        att = build_file_attachment(result, agent)

        assert att is not None
        assert att["path"] == str(png.resolve())
        assert att["display_name"] == "llm_image_123.png"
        assert att["size_bytes"] == png.stat().st_size

    def test_returns_none_when_success_false(self, tmp_path: Path) -> None:
        png = _make_png(tmp_path)
        agent = _make_agent()
        agent.register_file("test.png", str(png))

        att = build_file_attachment(
            {"success": False, "file_reference": "test.png", "error": "failed"},
            agent,
        )
        assert att is None

    def test_returns_none_when_no_file_reference(self) -> None:
        att = build_file_attachment({"success": True, "result": "text"}, _make_agent())
        assert att is None

    def test_returns_none_when_file_missing_on_disk(self, tmp_path: Path) -> None:
        agent = _make_agent()
        # Register a path that doesn't actually exist on disk
        agent.register_file("ghost.png", str(tmp_path / "ghost.png"))
        att = build_file_attachment(
            {"success": True, "file_reference": "ghost.png"}, agent
        )
        assert att is None

    def test_returns_none_when_agent_is_none(self) -> None:
        att = build_file_attachment(
            {"success": True, "file_reference": "x.png"}, agent=None
        )
        assert att is None

    def test_returns_none_when_agent_lacks_get_file_path(self) -> None:
        class _BareAgent:
            session_id = "sess"

        att = build_file_attachment(
            {"success": True, "file_reference": "x.png"}, agent=_BareAgent()
        )
        assert att is None

    def test_returns_none_for_non_dict_result(self) -> None:
        for result in ["some text", None, 42, ["a", "b"]]:
            assert build_file_attachment(result, _make_agent()) is None  # type: ignore[arg-type]

    def test_returns_none_for_empty_file_reference(self) -> None:
        att = build_file_attachment(
            {"success": True, "file_reference": ""}, _make_agent()
        )
        assert att is None


# ---------------------------------------------------------------------------#
# Unit: app rendering helper                                                #
# ---------------------------------------------------------------------------#

class TestBuildFileBubbles:
    """Contracts for ``_build_file_bubbles(attachment)``."""

    def test_image_produces_two_messages(self, tmp_path: Path) -> None:
        png = _make_png(tmp_path, "llm_image_abc.png")
        att = {
            "path": str(png.resolve()),
            "display_name": "llm_image_abc.png",
            "size_bytes": png.stat().st_size,
        }
        bubbles = build_file_bubbles(att)

        assert len(bubbles) == 2
        # First bubble: inline file dict
        assert bubbles[0]["role"] == "assistant"
        assert bubbles[0]["content"] == {
            "path": str(png.resolve()),
            "alt_text": "llm_image_abc.png",
        }
        # Second bubble: caption line
        assert bubbles[1]["role"] == "assistant"
        assert "llm_image_abc.png" in bubbles[1]["content"]
        assert "📎" in bubbles[1]["content"]

    def test_caption_includes_human_readable_size(self, tmp_path: Path) -> None:
        png = _make_png(tmp_path)
        size = png.stat().st_size
        att = {"path": str(png.resolve()), "display_name": "x.png", "size_bytes": size}
        bubbles = build_file_bubbles(att)

        caption = bubbles[1]["content"]
        assert _fmt_size(size) in caption

    def test_missing_path_returns_empty_list(self, tmp_path: Path) -> None:
        att = {
            "path": str(tmp_path / "gone.png"),
            "display_name": "gone.png",
            "size_bytes": 1000,
        }
        # File does not exist on disk — must produce nothing (no broken bubble)
        bubbles = build_file_bubbles(att)
        assert bubbles == []

    def test_none_attachment_returns_empty_list(self) -> None:
        assert build_file_bubbles(None) == []  # type: ignore[arg-type]

    def test_non_image_file_still_produces_two_messages(self, tmp_path: Path) -> None:
        """Generic path: Gradio renders non-images as download chips."""
        md = tmp_path / "report.md"
        md.write_text("# hello", encoding="utf-8")
        att = {
            "path": str(md.resolve()),
            "display_name": "report.md",
            "size_bytes": md.stat().st_size,
        }
        bubbles = build_file_bubbles(att)
        assert len(bubbles) == 2
        assert bubbles[0]["content"]["path"] == str(md.resolve())


# ---------------------------------------------------------------------------#
# Integration: app_ng_modular.py rendering path                             #
# ---------------------------------------------------------------------------#

class TestAppRenderFileAttachment:
    """The tool_end handler in app_ng_modular.py appends file bubbles
    when metadata['file_attachment'] is present and the file exists."""

    def _run_tool_end(
        self,
        attachment: dict | None,
        working_history: list | None = None,
    ) -> list[dict]:
        """Simulate what app_ng_modular does for a single tool_end event."""
        history = list(working_history or [])
        # The existing accordion message (always present — regression guard)
        history.append({
            "role": "assistant",
            "content": "result text",
            "metadata": {"title": "Tool called: generate_ai_image"},
        })
        # New: append file bubbles when attachment is resolved
        history.extend(build_file_bubbles(attachment))
        return history

    def test_accordion_always_present(self) -> None:
        history = self._run_tool_end(attachment=None)
        titles = [m.get("metadata", {}).get("title", "") for m in history]
        assert any("Tool called" in t for t in titles)

    def test_image_appended_after_accordion(self, tmp_path: Path) -> None:
        png = _make_png(tmp_path, "bear.png")
        att = {
            "path": str(png.resolve()),
            "display_name": "bear.png",
            "size_bytes": png.stat().st_size,
        }
        history = self._run_tool_end(att)

        # accordion + image bubble + caption = 3 messages
        assert len(history) == 3
        file_msg = history[1]
        assert file_msg["role"] == "assistant"
        assert file_msg["content"]["path"] == str(png.resolve())
        caption_msg = history[2]
        assert "📎" in caption_msg["content"]
        assert "bear.png" in caption_msg["content"]

    def test_no_file_bubble_when_no_attachment(self) -> None:
        history = self._run_tool_end(attachment=None)
        assert len(history) == 1  # only accordion, same as today

    def test_no_file_bubble_when_file_missing(self, tmp_path: Path) -> None:
        att = {
            "path": str(tmp_path / "gone.png"),
            "display_name": "gone.png",
            "size_bytes": 99,
        }
        history = self._run_tool_end(att)
        # File doesn't exist → skip bubbles, only accordion
        assert len(history) == 1


# ---------------------------------------------------------------------------#
# Integration: user-side file rendering (chat_tab.py mirror)               #
# ---------------------------------------------------------------------------#

class TestUserSideFileRendering:
    """Uploaded image files gain an inline preview bubble before the text."""

    def test_image_file_produces_user_bubble(self, tmp_path: Path) -> None:
        png = _make_png(tmp_path, "uploaded.png")
        att = {
            "path": str(png.resolve()),
            "display_name": "uploaded.png",
            "size_bytes": png.stat().st_size,
        }
        bubbles = build_file_bubbles_for_role(att, role="user")

        assert len(bubbles) == 1  # user side: only the inline file, no caption
        assert bubbles[0]["role"] == "user"
        assert bubbles[0]["content"]["path"] == str(png.resolve())
        assert bubbles[0]["content"]["alt_text"] == "uploaded.png"

    def test_non_image_file_also_produces_user_bubble(self, tmp_path: Path) -> None:
        pdf = tmp_path / "document.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        att = {
            "path": str(pdf.resolve()),
            "display_name": "document.pdf",
            "size_bytes": pdf.stat().st_size,
        }
        bubbles = build_file_bubbles_for_role(att, role="user")
        assert len(bubbles) == 1
        assert bubbles[0]["content"]["path"] == str(pdf.resolve())

    def test_missing_file_produces_no_bubble(self, tmp_path: Path) -> None:
        att = {
            "path": str(tmp_path / "nope.png"),
            "display_name": "nope.png",
            "size_bytes": 0,
        }
        bubbles = build_file_bubbles_for_role(att, role="user")
        assert bubbles == []


# ---------------------------------------------------------------------------#
# Unit: is_file_bubble filter                                               #
# ---------------------------------------------------------------------------#

class TestIsFileBubble:
    """Guards that file-bubble messages are correctly identified so callers
    (token_counter, token_budget, download-as-markdown) can skip them."""

    def test_dict_content_is_file_bubble(self) -> None:
        msg = {
            "role": "assistant",
            "content": {"path": _FIXTURE_PNG, "alt_text": "x.png"},
        }
        assert is_file_bubble(msg) is True

    def test_string_content_is_not_file_bubble(self) -> None:
        assert is_file_bubble({"role": "assistant", "content": "Готово! 🐻"}) is False

    def test_empty_string_content_is_not_file_bubble(self) -> None:
        assert is_file_bubble({"role": "assistant", "content": ""}) is False

    def test_none_content_is_not_file_bubble(self) -> None:
        # None is falsy but not a dict — should not be treated as a file bubble.
        assert is_file_bubble({"role": "assistant", "content": None}) is False

    def test_missing_content_is_not_file_bubble(self) -> None:
        assert is_file_bubble({"role": "user"}) is False


# ---------------------------------------------------------------------------#
# Regression: token_counter must not pass dict content to HumanMessage     #
# ---------------------------------------------------------------------------#

class TestTokenCounterSkipsFileBubbles:
    """Regression guard: convert_chat_history_to_messages must skip file
    bubbles so the LLM never receives a vision-formatted message for a model
    that doesn't support image input (the exact error the user reported)."""

    def test_file_bubble_skipped_entirely(self) -> None:
        history = [
            {"role": "user", "content": "Сгенерируй медведя"},
            # file bubble — should be skipped
            {
                "role": "assistant",
                "content": {"path": _FIXTURE_BEAR, "alt_text": "bear.png"},
            },
            # caption — string, should be included
            {"role": "assistant", "content": "📎 bear.png — 1.9 MB"},
            {"role": "assistant", "content": "Готово! 🐻"},
        ]
        messages = convert_chat_history_to_messages(history)

        # 3 string messages, 0 dict messages
        assert len(messages) == 3
        for m in messages:
            assert isinstance(m.content, str), (
                f"content must be str, got {type(m.content)}: {m.content!r}"
            )

    def test_mixed_history_correct_count(self) -> None:
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {
                "role": "assistant",
                "content": {"path": _FIXTURE_PNG, "alt_text": "x.png"},
            },
            {"role": "assistant", "content": "📎 x.png — 500 KB"},
            {"role": "assistant", "content": "Here's your image."},
        ]
        messages = convert_chat_history_to_messages(history)
        assert len(messages) == 4  # dict bubble skipped, 4 string messages remain


# ---------------------------------------------------------------------------#
# Tool cost accumulation                                                    #
# ---------------------------------------------------------------------------#

class TestAddToolCost:
    """TokenCounter.add_tool_cost() feeds session and conversation totals."""

    def _make_tracker(self) -> ConversationTokenTracker:
        """Minimal tracker with zeroed cost accumulators.

        ``ConversationTokenTracker`` is the class used at runtime
        (via ``agent.token_tracker``). Bypass its full init to avoid
        needing an LLMManager / pricing config.
        """
        tc = ConversationTokenTracker.__new__(ConversationTokenTracker)
        tc.session_cost = 0.0
        tc.conversation_cost = 0.0
        tc._turn_cost = None  # noqa: SLF001 — test-only state seeding
        return tc

    def test_positive_amount_added_to_both_accumulators(self) -> None:
        tc = self._make_tracker()
        tc.add_tool_cost(0.067)

        assert abs(tc.session_cost - 0.067) < 1e-9
        assert abs(tc.conversation_cost - 0.067) < 1e-9

    def test_multiple_calls_accumulate(self) -> None:
        tc = self._make_tracker()
        tc.add_tool_cost(0.067)
        tc.add_tool_cost(0.030)
        assert abs(tc.session_cost - 0.097) < 1e-9
        assert abs(tc.conversation_cost - 0.097) < 1e-9

    def test_zero_ignored(self) -> None:
        tc = self._make_tracker()
        tc.add_tool_cost(0.0)
        assert tc.session_cost == 0.0
        assert tc.conversation_cost == 0.0

    def test_negative_ignored(self) -> None:
        tc = self._make_tracker()
        tc.add_tool_cost(-1.0)
        assert tc.session_cost == 0.0
        assert tc.conversation_cost == 0.0

    def test_none_ignored(self) -> None:
        tc = self._make_tracker()
        tc.add_tool_cost(None)  # type: ignore[arg-type]
        assert tc.session_cost == 0.0
        assert tc.conversation_cost == 0.0


class TestToolCostExtractedFromToolResult:
    """The streaming layer extracts 'cost' from a tool result dict into
    tool_end event metadata as 'tool_cost'."""

    def _extract_tool_cost(self, tool_result: Any) -> float | None:
        """Replicate the extraction logic from native_langchain_streaming.py."""
        _raw = (
            tool_result.get("cost")
            if isinstance(tool_result, dict)
            else None
        )
        return (
            float(_raw)
            if isinstance(_raw, (int, float)) and _raw > 0
            else None
        )

    def test_image_result_cost_extracted(self) -> None:
        result = {"success": True, "file_reference": "x.png", "cost": 0.067}
        assert abs(self._extract_tool_cost(result) - 0.067) < 1e-9

    def test_zero_cost_returns_none(self) -> None:
        assert self._extract_tool_cost({"success": True, "cost": 0.0}) is None

    def test_negative_cost_returns_none(self) -> None:
        assert self._extract_tool_cost({"success": True, "cost": -1.0}) is None

    def test_missing_cost_returns_none(self) -> None:
        assert self._extract_tool_cost({"success": True}) is None

    def test_string_cost_returns_none(self) -> None:
        assert self._extract_tool_cost({"success": True, "cost": "0.067"}) is None

    def test_non_dict_result_returns_none(self) -> None:
        assert self._extract_tool_cost("just text") is None
        assert self._extract_tool_cost(None) is None

    def test_integer_cost_accepted(self) -> None:
        """Integer cost (e.g. 1) should be accepted and converted to float."""
        assert self._extract_tool_cost({"success": True, "cost": 1}) == 1.0
