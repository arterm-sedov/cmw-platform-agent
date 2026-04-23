"""
Tests for VisionInput dataclass

Following TDD principles - test behavior, not implementation
"""
import pytest
from pathlib import Path
import tempfile
import base64

from agent_ng.vision_input import VisionInput, MediaType, create_vision_input


class TestVisionInputCreation:
    """Test VisionInput creation and validation"""

    def test_text_only_input(self):
        """Test creating text-only input"""
        input = VisionInput(prompt="Hello world")
        assert input.prompt == "Hello world"
        assert input.media_type == MediaType.TEXT
        assert not input.has_media()

    def test_image_path_input(self):
        """Test creating input with image path"""
        # Create temp image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(b'fake image data')
            image_path = f.name

        try:
            input = VisionInput(prompt="Describe image", image_path=image_path)
            assert input.prompt == "Describe image"
            assert input.media_type == MediaType.IMAGE
            assert input.has_media()
            assert input.get_media_path() == image_path
        finally:
            Path(image_path).unlink()

    def test_image_url_input(self):
        """Test creating input with image URL"""
        input = VisionInput(
            prompt="Describe image",
            image_url="https://example.com/image.jpg"
        )
        assert input.media_type == MediaType.IMAGE
        assert input.get_media_url() == "https://example.com/image.jpg"

    def test_image_base64_input(self):
        """Test creating input with base64 image"""
        base64_data = base64.b64encode(b'fake image').decode('utf-8')
        input = VisionInput(
            prompt="Describe image",
            image_base64=base64_data
        )
        assert input.media_type == MediaType.IMAGE
        assert input.get_media_base64() == base64_data

    def test_video_path_input(self):
        """Test creating input with video path"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(b'fake video data')
            video_path = f.name

        try:
            input = VisionInput(prompt="Describe video", video_path=video_path)
            assert input.media_type == MediaType.VIDEO
            assert input.get_media_path() == video_path
        finally:
            Path(video_path).unlink()

    def test_audio_path_input(self):
        """Test creating input with audio path"""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b'fake audio data')
            audio_path = f.name

        try:
            input = VisionInput(prompt="Transcribe audio", audio_path=audio_path)
            assert input.media_type == MediaType.AUDIO
            assert input.get_media_path() == audio_path
        finally:
            Path(audio_path).unlink()


class TestVisionInputValidation:
    """Test VisionInput validation"""

    def test_validate_requires_prompt(self):
        """Test that prompt is required"""
        with pytest.raises(ValueError, match="Prompt is required"):
            input = VisionInput(prompt="")
            input.validate()

    def test_validate_single_image_source(self):
        """Test that only one image source is allowed"""
        with pytest.raises(ValueError, match="Only one image source"):
            input = VisionInput(
                prompt="Test",
                image_path="test.jpg",
                image_url="https://example.com/test.jpg"
            )
            input.validate()

    def test_validate_file_exists(self):
        """Test that file path must exist"""
        with pytest.raises(ValueError, match="Media file not found"):
            input = VisionInput(
                prompt="Test",
                image_path="/nonexistent/file.jpg"
            )
            input.validate()

    def test_validate_success(self):
        """Test successful validation"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(b'fake image')
            image_path = f.name

        try:
            input = VisionInput(prompt="Test", image_path=image_path)
            assert input.validate() == True
        finally:
            Path(image_path).unlink()


class TestVisionInputConversion:
    """Test VisionInput conversion methods"""

    def test_to_base64_from_file(self):
        """Test converting file to base64"""
        test_data = b'test image data'
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(test_data)
            image_path = f.name

        try:
            input = VisionInput(prompt="Test", image_path=image_path)
            base64_result = input.to_base64()

            # Verify base64 encoding
            decoded = base64.b64decode(base64_result)
            assert decoded == test_data
        finally:
            Path(image_path).unlink()

    def test_to_base64_returns_existing(self):
        """Test that existing base64 is returned"""
        existing_base64 = base64.b64encode(b'existing data').decode('utf-8')
        input = VisionInput(prompt="Test", image_base64=existing_base64)

        result = input.to_base64()
        assert result == existing_base64


class TestCreateVisionInput:
    """Test convenience function create_vision_input"""

    def test_create_with_image(self):
        """Test creating input with image"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(b'fake image')
            image_path = f.name

        try:
            input = create_vision_input(
                prompt="Describe this",
                media_path=image_path
            )
            assert input.media_type == MediaType.IMAGE
            assert input.image_path == image_path
        finally:
            Path(image_path).unlink()

    def test_create_with_video(self):
        """Test creating input with video"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(b'fake video')
            video_path = f.name

        try:
            input = create_vision_input(
                prompt="Describe this",
                media_path=video_path
            )
            assert input.media_type == MediaType.VIDEO
            assert input.video_path == video_path
        finally:
            Path(video_path).unlink()

    def test_create_with_explicit_type(self):
        """Test creating input with explicit media type"""
        input = create_vision_input(
            prompt="Test",
            media_url="https://example.com/file.dat",
            media_type=MediaType.IMAGE
        )
        assert input.media_type == MediaType.IMAGE
        assert input.image_url == "https://example.com/file.dat"
