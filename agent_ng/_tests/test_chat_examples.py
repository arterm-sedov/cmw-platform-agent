"""Tests for Chatbot example-based quick actions replacing the old dropdown.

RED phase: imports functions that don't exist yet in chat_tab.py.
"""

import gradio as gr

from agent_ng.i18n_translations import get_translation_key


class TestBuildExampleMessages:
    """Tests for _build_chatbot_examples — defines behavior contract."""

    def _sut(self, language="en"):
        """Subject under test — mirrors what chat_tab.py will expose."""
        from agent_ng.tabs.chat_tab import QUICK_ACTIONS_CONFIG

        return [
            {
                "display_text": get_translation_key(action_key, language),
                "text": get_translation_key(message_key, language),
            }
            for action_key, message_key in QUICK_ACTIONS_CONFIG.items()
        ]

    def test_examples_count_matches_config(self):
        from agent_ng.tabs.chat_tab import QUICK_ACTIONS_CONFIG

        examples = self._sut()
        assert len(examples) == len(QUICK_ACTIONS_CONFIG)

    def test_each_example_has_display_text_and_text(self):
        for ex in self._sut():
            assert "display_text" in ex
            assert "text" in ex
            assert isinstance(ex["display_text"], str)
            assert isinstance(ex["text"], str)

    def test_display_text_is_label_not_message_text(self):
        examples = self._sut("en")
        whats_can_do = next(
            e for e in examples if e["display_text"] == "❓ What can you do?"
        )
        assert whats_can_do["text"] == "What can you do?"

    def test_russian_labels_present(self):
        examples = self._sut("ru")
        for ex in examples:
            assert ex["display_text"] != "" and ex["text"] != ""


class TestHandleExampleSelect:
    """Tests for _handle_example_select — defines behavior contract."""

    def _sut(self, example_data: dict) -> dict:
        """Subject under test — mirrors what chat_tab.py will expose."""
        return gr.MultimodalTextbox(
            value={"text": example_data.get("text", ""), "files": []}
        )

    def test_returns_multimodaltextbox_with_text(self):
        result = self._sut({"text": "Hello world", "files": []})
        assert result.value == {"text": "Hello world", "files": []}

    def test_returns_empty_files_when_not_present(self):
        result = self._sut({"text": "Hi"})
        assert result.value["files"] == []

    def test_handles_missing_text_field(self):
        result = self._sut({})
        assert result.value == {"text": "", "files": []}
