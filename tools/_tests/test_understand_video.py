"""
Tests for understand_video tool with VisionToolManager
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the underlying function, not the @tool decorated version
from tools import tools
understand_video = tools.understand_video.func  # Get the actual function


class TestUnderstandVideo:
    """Test understand_video tool with VisionToolManager"""

    def test_understand_video_with_file_path(self):
        """Test analyzing video from file path"""
        # Create temp video file (just a placeholder)
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(b'fake video data')
            video_path = f.name

        try:
            mock_agent = Mock()
            mock_agent.file_registry = {Path(video_path).name: video_path}

            with patch('agent_ng.vision_tool_manager.VisionToolManager.analyze_video') as mock_analyze:
                mock_analyze.return_value = "This video shows a person walking"

                result = understand_video(
                    file_reference=Path(video_path).name,
                    prompt="What's happening in this video?",
                    agent=mock_agent
                )

                assert isinstance(result, str)
                import json
                result_data = json.loads(result)
                assert result_data["type"] == "tool_response"
                assert result_data["tool_name"] == "understand_video"

        finally:
            Path(video_path).unlink()

    def test_understand_video_with_url(self):
        """Test analyzing video from URL"""
        mock_agent = Mock()

        with patch('tools.file_utils.FileUtils.resolve_file_reference') as mock_resolve:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                f.write(b'fake video')
                temp_path = f.name

            try:
                mock_resolve.return_value = temp_path

                with patch('agent_ng.vision_tool_manager.VisionToolManager.analyze_video') as mock_analyze:
                    mock_analyze.return_value = "Video analysis result"

                    result = understand_video(
                        file_reference="https://example.com/video.mp4",
                        prompt="Describe this video",
                        agent=mock_agent
                    )

                    assert isinstance(result, str)
                    import json
                    result_data = json.loads(result)
                    assert result_data["type"] == "tool_response"

            finally:
                Path(temp_path).unlink()

    def test_understand_video_file_not_found(self):
        """Test error handling when file not found"""
        mock_agent = Mock()

        with patch('tools.file_utils.FileUtils.resolve_file_reference') as mock_resolve:
            mock_resolve.return_value = None

            result = understand_video(
                file_reference="nonexistent.mp4",
                prompt="Describe",
                agent=mock_agent
            )

            import json
            result_data = json.loads(result)
            assert "error" in result_data

    def test_understand_video_with_system_prompt(self):
        """Test with custom system prompt"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(b'fake video')
            video_path = f.name

        try:
            mock_agent = Mock()
            mock_agent.file_registry = {Path(video_path).name: video_path}

            with patch('agent_ng.vision_tool_manager.VisionToolManager.analyze_video') as mock_analyze:
                mock_analyze.return_value = "Detailed video analysis"

                result = understand_video(
                    file_reference=Path(video_path).name,
                    prompt="Analyze this video",
                    system_prompt="You are a video expert",
                    agent=mock_agent
                )

                assert isinstance(result, str)
                import json
                result_data = json.loads(result)
                assert "result" in result_data

        finally:
            Path(video_path).unlink()

    def test_understand_video_with_timestamps(self):
        """Test video analysis with start/end timestamps"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(b'fake video')
            video_path = f.name

        try:
            mock_agent = Mock()
            mock_agent.file_registry = {Path(video_path).name: video_path}

            with patch('agent_ng.vision_tool_manager.VisionToolManager.analyze_video') as mock_analyze:
                mock_analyze.return_value = "Video segment analysis"

                result = understand_video(
                    file_reference=Path(video_path).name,
                    prompt="What happens in this segment?",
                    start_time="00:30",
                    end_time="01:00",
                    agent=mock_agent
                )

                assert isinstance(result, str)
                import json
                result_data = json.loads(result)
                assert "result" in result_data

        finally:
            Path(video_path).unlink()


class TestUnderstandVideoIntegration:
    """Integration tests with real VisionToolManager (requires API key)"""

    @pytest.mark.integration
    def test_real_video_analysis(self):
        """Test with real API call (requires OPENROUTER_API_KEY)"""
        import os
        if not os.getenv('OPENROUTER_API_KEY'):
            pytest.skip("OPENROUTER_API_KEY not set")

        # Would need a real test video file
        pytest.skip("Real video test requires test video file")
