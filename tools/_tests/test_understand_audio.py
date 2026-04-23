"""
Tests for understand_audio tool with VisionToolManager
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the underlying function, not the @tool decorated version
from tools import tools
understand_audio = tools.understand_audio.func  # Get the actual function


class TestUnderstandAudio:
    """Test understand_audio tool with VisionToolManager"""

    def test_understand_audio_with_file_path(self):
        """Test analyzing audio from file path"""
        # Create temp audio file (just a placeholder)
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b'fake audio data')
            audio_path = f.name

        try:
            mock_agent = Mock()
            mock_agent.file_registry = {Path(audio_path).name: audio_path}

            with patch('agent_ng.vision_tool_manager.VisionToolManager.analyze_audio') as mock_analyze:
                mock_analyze.return_value = "This audio contains speech"

                result = understand_audio(
                    file_reference=Path(audio_path).name,
                    prompt="What's in this audio?",
                    agent=mock_agent
                )

                assert isinstance(result, str)
                import json
                result_data = json.loads(result)
                assert result_data["type"] == "tool_response"
                assert result_data["tool_name"] == "understand_audio"

        finally:
            Path(audio_path).unlink()

    def test_understand_audio_with_url(self):
        """Test analyzing audio from URL"""
        mock_agent = Mock()

        with patch('tools.file_utils.FileUtils.resolve_file_reference') as mock_resolve:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                f.write(b'fake audio')
                temp_path = f.name

            try:
                mock_resolve.return_value = temp_path

                with patch('agent_ng.vision_tool_manager.VisionToolManager.analyze_audio') as mock_analyze:
                    mock_analyze.return_value = "Audio analysis result"

                    result = understand_audio(
                        file_reference="https://example.com/audio.mp3",
                        prompt="Transcribe this audio",
                        agent=mock_agent
                    )

                    assert isinstance(result, str)
                    import json
                    result_data = json.loads(result)
                    assert result_data["type"] == "tool_response"

            finally:
                Path(temp_path).unlink()

    def test_understand_audio_file_not_found(self):
        """Test error handling when file not found"""
        mock_agent = Mock()

        with patch('tools.file_utils.FileUtils.resolve_file_reference') as mock_resolve:
            mock_resolve.return_value = None

            result = understand_audio(
                file_reference="nonexistent.mp3",
                prompt="Transcribe",
                agent=mock_agent
            )

            import json
            result_data = json.loads(result)
            assert "error" in result_data

    def test_understand_audio_with_system_prompt(self):
        """Test with custom system prompt"""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b'fake audio')
            audio_path = f.name

        try:
            mock_agent = Mock()
            mock_agent.file_registry = {Path(audio_path).name: audio_path}

            with patch('agent_ng.vision_tool_manager.VisionToolManager.analyze_audio') as mock_analyze:
                mock_analyze.return_value = "Detailed audio analysis"

                result = understand_audio(
                    file_reference=Path(audio_path).name,
                    prompt="Analyze this audio",
                    system_prompt="You are an audio expert",
                    agent=mock_agent
                )

                assert isinstance(result, str)
                import json
                result_data = json.loads(result)
                assert "result" in result_data

        finally:
            Path(audio_path).unlink()

    def test_understand_audio_with_timestamps(self):
        """Test audio analysis with start/end timestamps"""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b'fake audio')
            audio_path = f.name

        try:
            mock_agent = Mock()
            mock_agent.file_registry = {Path(audio_path).name: audio_path}

            with patch('agent_ng.vision_tool_manager.VisionToolManager.analyze_audio') as mock_analyze:
                mock_analyze.return_value = "Audio segment analysis"

                result = understand_audio(
                    file_reference=Path(audio_path).name,
                    prompt="What's said in this segment?",
                    start_time="00:30",
                    end_time="01:00",
                    agent=mock_agent
                )

                assert isinstance(result, str)
                import json
                result_data = json.loads(result)
                assert "result" in result_data

        finally:
            Path(audio_path).unlink()


class TestUnderstandAudioIntegration:
    """Integration tests with real VisionToolManager (requires API key)"""

    @pytest.mark.integration
    def test_real_audio_analysis(self):
        """Test with real API call (requires OPENROUTER_API_KEY)"""
        import os
        if not os.getenv('OPENROUTER_API_KEY'):
            pytest.skip("OPENROUTER_API_KEY not set")

        # Would need a real test audio file
        pytest.skip("Real audio test requires test audio file")
