"""
Tests for VL-powered tools (analyze_image_ai, understand_video, understand_audio)

Following TDD principles - test behavior, not implementation
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the tools we'll be testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools import tools
analyze_image_ai = tools.analyze_image_ai.func  # Get the actual function


class TestAnalyzeImageAI:
    """Test analyze_image_ai tool with VisionToolManager"""

    def test_analyze_image_ai_with_file_path(self):
        """Test analyzing image from file path"""
        # Create temp image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(f.name)
            image_path = f.name

        try:
            # Mock agent with file registry
            mock_agent = Mock()
            mock_agent.file_registry = {Path(image_path).name: image_path}

            # Call tool
            result = analyze_image_ai(
                file_reference=Path(image_path).name,
                prompt="What color is this image?",
                agent=mock_agent
            )

            # Verify result is JSON string
            assert isinstance(result, str)
            import json
            result_data = json.loads(result)

            # Verify structure
            assert result_data["type"] == "tool_response"
            assert result_data["tool_name"] == "analyze_image_ai"
            assert "result" in result_data

        finally:
            Path(image_path).unlink()

    def test_analyze_image_ai_with_url(self):
        """Test analyzing image from URL"""
        mock_agent = Mock()

        with patch('tools.file_utils.FileUtils.resolve_file_reference') as mock_resolve:
            # Mock URL download
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                from PIL import Image
                img = Image.new('RGB', (100, 100), color='blue')
                img.save(f.name)
                temp_path = f.name

            try:
                mock_resolve.return_value = temp_path

                result = analyze_image_ai(
                    file_reference="https://example.com/image.jpg",
                    prompt="Describe this image",
                    agent=mock_agent
                )

                assert isinstance(result, str)
                import json
                result_data = json.loads(result)
                assert result_data["type"] == "tool_response"

            finally:
                Path(temp_path).unlink()

    def test_analyze_image_ai_fast_mode(self):
        """Test default analysis"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='green')
            img.save(f.name)
            image_path = f.name

        try:
            mock_agent = Mock()
            mock_agent.file_registry = {Path(image_path).name: image_path}

            # Call tool
            result = analyze_image_ai(
                file_reference=Path(image_path).name,
                prompt="Quick description",
                agent=mock_agent
            )

            assert isinstance(result, str)
            import json
            result_data = json.loads(result)
            assert "result" in result_data

        finally:
            Path(image_path).unlink()

    def test_analyze_image_ai_quality_mode(self):
        """Test analysis with system prompt"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='yellow')
            img.save(f.name)
            image_path = f.name

        try:
            mock_agent = Mock()
            mock_agent.file_registry = {Path(image_path).name: image_path}

            # Call with system prompt
            result = analyze_image_ai(
                file_reference=Path(image_path).name,
                prompt="Detailed analysis",
                system_prompt="Analyze carefully",
                agent=mock_agent
            )

            assert isinstance(result, str)
            import json
            result_data = json.loads(result)
            assert "result" in result_data

        finally:
            Path(image_path).unlink()

    def test_analyze_image_ai_file_not_found(self):
        """Test error handling when file not found"""
        mock_agent = Mock()

        with patch('tools.file_utils.FileUtils.resolve_file_reference') as mock_resolve:
            # Mock file not found
            mock_resolve.return_value = None

            result = analyze_image_ai(
                file_reference="nonexistent.jpg",
                prompt="Describe",
                agent=mock_agent
            )

            import json
            result_data = json.loads(result)
            assert "error" in result_data
            assert "not found" in result_data["error"].lower()

    def test_analyze_image_ai_invalid_image(self):
        """Test error handling with invalid image file"""
        # Create temp non-image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, mode='w') as f:
            f.write("not an image")
            file_path = f.name

        try:
            mock_agent = Mock()
            mock_agent.file_registry = {Path(file_path).name: file_path}

            result = analyze_image_ai(
                file_reference=Path(file_path).name,
                prompt="Describe",
                agent=mock_agent
            )

            import json
            result_data = json.loads(result)
            assert "error" in result_data

        finally:
            Path(file_path).unlink()

    def test_analyze_image_ai_with_system_prompt(self):
        """Test with custom system prompt"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='purple')
            img.save(f.name)
            image_path = f.name

        try:
            mock_agent = Mock()
            mock_agent.file_registry = {Path(image_path).name: image_path}

            result = analyze_image_ai(
                file_reference=Path(image_path).name,
                prompt="Describe colors",
                system_prompt="You are a color expert",
                agent=mock_agent
            )

            assert isinstance(result, str)
            import json
            result_data = json.loads(result)
            assert "result" in result_data

        finally:
            Path(image_path).unlink()


class TestAnalyzeImageAIIntegration:
    """Integration tests with real VisionToolManager (requires API key)"""

    @pytest.mark.integration
    def test_real_image_analysis(self):
        """Test with real API call (requires OPENROUTER_API_KEY)"""
        import os
        if not os.getenv('OPENROUTER_API_KEY'):
            pytest.skip("OPENROUTER_API_KEY not set")

        # Use test image from experiments
        test_image = Path("experiments/test_files/test_image.jpg")
        if not test_image.exists():
            pytest.skip("Test image not found")

        mock_agent = Mock()
        mock_agent.file_registry = {test_image.name: str(test_image)}

        result = analyze_image_ai(
            file_reference=test_image.name,
            prompt="Describe the shapes and colors in this image",
            agent=mock_agent
        )

        import json
        result_data = json.loads(result)
        assert result_data["type"] == "tool_response"
        assert "result" in result_data
        assert len(result_data["result"]) > 50  # Should have substantial response
