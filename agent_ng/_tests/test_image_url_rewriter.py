"""Unit tests for agent_ng/_image_url_rewriter.py."""

from __future__ import annotations

import pytest

from agent_ng._image_url_rewriter import rewrite_llm_inline_images


class DummyAgent:
    """Mock agent with a get_file_path method."""

    def __init__(self, mapping: dict | None = None):
        self._mapping = mapping or {}

    def get_file_path(self, filename: str) -> str | None:
        return self._mapping.get(filename)


def test_basic_rewrite_png() -> None:
    agent = DummyAgent({"llm_image_abc.png": "C:\\Temp\\llm_image_abc.png"})
    content = 'Here is an image: <img src="llm_image_abc.png" alt="test">'
    expected = 'Here is an image: <img src="/gradio_api/file=C:\\Temp\\llm_image_abc.png">'
    assert rewrite_llm_inline_images(content, agent) == expected


def test_multiple_images() -> None:
    agent = DummyAgent({"llm_image_1.png": "/tmp/1.png", "llm_image_2.jpg": "/tmp/2.jpg"})
    content = 'First: <img src="llm_image_1.png"> Second: <img src="llm_image_2.jpg"> Third: <img src="llm_image_3.png">'
    assert "/gradio_api/file=/tmp/1.png" in rewrite_llm_inline_images(content, agent)
    assert "/gradio_api/file=/tmp/2.jpg" in rewrite_llm_inline_images(content, agent)


def test_markdown_image_link() -> None:
    agent = DummyAgent({"llm_image_foo.png": "/tmp/foo.png"})
    content = "![Alt Text](llm_image_foo.png)"
    expected = "![Alt Text](/gradio_api/file=/tmp/foo.png)"
    assert rewrite_llm_inline_images(content, agent) == expected


def test_markdown_image_link_multiple() -> None:
    agent = DummyAgent({"llm_image_a.png": "/tmp/a.png", "llm_image_b.jpg": "/tmp/b.jpg"})
    content = "First ![a](llm_image_a.png) then ![b](llm_image_b.jpg)"
    result = rewrite_llm_inline_images(content, agent)
    assert "/gradio_api/file=/tmp/a.png" in result
    assert "/gradio_api/file=/tmp/b.jpg" in result


def test_unrelated_img_untouched() -> None:
    agent = DummyAgent({"llm_image_foo.png": "/tmp/foo.png"})
    content = '<img src="user_upload.png"> <img src="llm_image_foo.png">'
    result = rewrite_llm_inline_images(content, agent)
    assert "user_upload.png" in result
    assert "/gradio_api/file=/tmp/foo.png" in result


def test_no_images() -> None:
    agent = DummyAgent({"llm_image_foo.png": "/tmp/foo.png"})
    content = "Plain text with no images."
    assert rewrite_llm_inline_images(content, agent) == content


def test_agent_is_none() -> None:
    content = '<img src="llm_image_foo.png">'
    assert rewrite_llm_inline_images(content, None) == content


def test_file_not_in_registry() -> None:
    agent = DummyAgent({})
    content = '<img src="llm_image_foo.png">'
    assert rewrite_llm_inline_images(content, agent) == content


def test_windows_backslash_path() -> None:
    agent = DummyAgent({"llm_image_foo.png": "C:\\Temp\\llm_image_foo.png"})
    content = '<img src="llm_image_foo.png">'
    expected = '<img src="/gradio_api/file=C:\\Temp\\llm_image_foo.png">'
    assert rewrite_llm_inline_images(content, agent) == expected


def test_non_png_extensions() -> None:
    agent = DummyAgent({"llm_image_foo.jpg": "/tmp/foo.jpg"})
    content = '<img src="llm_image_foo.jpg">'
    assert "/gradio_api/file=/tmp/foo.jpg" in rewrite_llm_inline_images(content, agent)


if __name__ == "__main__":
    pytest.main([__file__])