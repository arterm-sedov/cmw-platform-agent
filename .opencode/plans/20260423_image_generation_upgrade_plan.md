# Image Generation Implementation Plan

**Date:** 2026-04-23
**Status:** ✅ Ready for TDD Implementation
**Estimated Time:** 4-5 hours
**Research:** See `docs/image_generation/progress_reports/20260423_progress_report.md`

---

## Summary

Add AI-powered image generation via OpenRouter API. Keep existing primitive generator as fallback.

**Tested Models:**
- ✅ `google/gemini-2.5-flash-image` - 6.2s, $0.039 (PRIMARY)
- ✅ `black-forest-labs/flux.2-pro` - 10.8s, $0.030 (CHEAPEST)
- ✅ `bytedance-seed/seedream-4.5` - 12.9s, $0.040 (FALLBACK)
- ⚠️ `openai/gpt-5-image-mini` - Blocked locally, but should work on HuggingFace Spaces

**Test Results:** 7 images generated successfully with CMW business prompts
**Test Images:** `docs/image_generation/progress_reports/test_*.png`

---

## Implementation (TDD Approach)

### Phase 1: Tests First (1 hour)

**File:** `agent_ng/_tests/test_image_engine.py`

```python
"""Tests for ImageEngine."""
import pytest
from agent_ng.image_engine import ImageEngine, ImageGenerationResult


class TestImageEngine:
    def test_generate_image_success(self):
        """Test successful image generation."""
        engine = ImageEngine()
        result = engine.generate("A blue circle")
        assert result.success
        assert result.image_data is not None
        assert len(result.image_data) > 0
    
    def test_generate_with_specific_model(self):
        """Test generation with specific model."""
        engine = ImageEngine()
        result = engine.generate("A red square", model="black-forest-labs/flux.2-pro")
        assert result.success
        assert result.model_used == "black-forest-labs/flux.2-pro"
    
    def test_invalid_model_with_fallback(self):
        """Test fallback on invalid model."""
        engine = ImageEngine()
        result = engine.generate("A green triangle", model="invalid/model")
        # Should fallback to working model
        assert result.model_used in [
            "google/gemini-2.5-flash-image",
            "black-forest-labs/flux.2-pro",
            "bytedance-seed/seedream-4.5"
        ]
    
    def test_missing_api_key(self):
        """Test error when API key missing."""
        import os
        old_key = os.environ.get('OPENROUTER_API_KEY')
        os.environ.pop('OPENROUTER_API_KEY', None)
        
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            ImageEngine()
        
        if old_key:
            os.environ['OPENROUTER_API_KEY'] = old_key
    
    def test_response_parsing(self):
        """Test base64 image extraction."""
        engine = ImageEngine()
        # Mock response data
        data = {
            'choices': [{
                'message': {
                    'images': [{
                        'image_url': {
                            'url': 'data:image/png;base64,iVBORw0KGgo='
                        }
                    }]
                }
            }],
            'usage': {'cost': 0.039, 'total_tokens': 1300}
        }
        result = engine._parse_response(data, "test-model")
        assert result.success
        assert result.cost == 0.039
        assert result.tokens == 1300
```

**File:** `agent_ng/_tests/test_generate_ai_image_tool.py`

```python
"""Tests for generate_ai_image tool."""
import json
from tools.tools import generate_ai_image


class TestGenerateAIImageTool:
    def test_tool_success(self):
        """Test tool generates image."""
        result_json = generate_ai_image(prompt="A simple icon")
        result = json.loads(result_json)
        assert result['success'] is True
        assert 'image_base64' in result
        assert result['model_used'] is not None
    
    def test_tool_with_model_param(self):
        """Test tool with specific model."""
        result_json = generate_ai_image(
            prompt="A logo",
            model="black-forest-labs/flux.2-pro"
        )
        result = json.loads(result_json)
        assert result['success'] is True
        assert result['model_used'] == "black-forest-labs/flux.2-pro"
    
    def test_tool_saves_file(self, tmp_path):
        """Test tool saves image to file."""
        save_path = str(tmp_path / "test.png")
        result_json = generate_ai_image(
            prompt="A circle",
            save_path=save_path
        )
        result = json.loads(result_json)
        assert result['success'] is True
        assert result['saved_to'] == save_path
        
        import os
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0
```

### Phase 2: Configuration (30 min)

**File:** `agent_ng/llm_configs.py`

Add to OPENROUTER models list:

```python
{
    "model": "google/gemini-2.5-flash-image",
    "token_limit": 1048576,
    "max_tokens": 8192,
    "temperature": 0,
    "force_tools": False,
    "image_generation": True
},
{
    "model": "black-forest-labs/flux.2-pro",
    "token_limit": 8192,
    "max_tokens": 1024,
    "temperature": 0,
    "force_tools": False,
    "image_generation": True
},
{
    "model": "bytedance-seed/seedream-4.5",
    "token_limit": 16384,
    "max_tokens": 2048,
    "temperature": 0,
    "force_tools": False,
    "image_generation": True
},
{
    "model": "openai/gpt-5-image-mini",
    "token_limit": 8192,
    "max_tokens": 1024,
    "temperature": 0,
    "force_tools": False,
    "image_generation": True,
    "note": "Works on HuggingFace Spaces (no region restrictions)"
}
```

### Phase 3: ImageEngine (1.5 hours)

**File:** `agent_ng/image_engine.py`

See full implementation in archived plan: `.opencode/plans/20260423_image_generation_upgrade_plan.md` (lines 150-350)

**Key components:**
- `ImageGenerationResult` dataclass
- `ImageEngine` class with `generate()` method
- `_call_api()` - OpenRouter API integration
- `_parse_response()` - Base64 extraction
- Automatic fallback on failure
- Support for Gemini, Flux, Seedream, and OpenAI models

### Phase 4: Tool Integration (1 hour)

**File:** `tools/tools.py`

Add `generate_ai_image` tool with Pydantic schema.

See full implementation in archived plan (lines 352-450)

### Phase 5: Verify (1 hour)

```bash
# Activate environment
.venv\Scripts\Activate.ps1

# Run tests
python -m pytest agent_ng/_tests/test_image_engine.py -v
python -m pytest agent_ng/_tests/test_generate_ai_image_tool.py -v

# Lint
ruff check agent_ng/image_engine.py
ruff format agent_ng/image_engine.py

# Type check
mypy agent_ng/image_engine.py
```

---

## Checkpoints

- [ ] Phase 1: Tests written
- [ ] Phase 2: Config updated
- [ ] Phase 3: ImageEngine implemented
- [ ] Phase 4: Tool integrated
- [ ] Phase 5: All tests passing

---

## Files

**Create:**
- `agent_ng/image_engine.py`
- `agent_ng/_tests/test_image_engine.py`
- `agent_ng/_tests/test_generate_ai_image_tool.py`

**Modify:**
- `agent_ng/llm_configs.py`
- `tools/tools.py`

**Keep:**
- `tools/tools.py:1505` - `generate_simple_image` (fallback)

---

## References

- **Test Results:** `docs/image_generation/progress_reports/20260423_progress_report.md`
- **Detailed Plan:** `.opencode/plans/20260423_image_generation_upgrade_plan.md` (archived)
- **Test Images:** `docs/image_generation/progress_reports/test_*.png`