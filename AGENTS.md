# AGENTS.md - CMW Platform Agent Coding Guidelines

This file provides essential information for AI coding agents working on the CMW Platform Agent codebase. These guidelines are battle-tested from production use and specific to this LangChain-based AI agent.

**Plan:** lean, dry, minimal, abstract, non-breaking, brilliant, genius code. Deduplicated reusable code following best practices in TDD, SDD, Python, 12-factor agents, 12-factor software, Gradio, and LangChain.

## Project Overview

- **Language**: Python 3.11+
- **Frameworks**: LangChain (core), Gradio (UI), Pydantic (validation)
- **Purpose**: AI agent for creating/managing Comindware Platform entities via natural language
- **Architecture**: Modular, LangChain-native with comprehensive error handling and internationalization

## Build/Lint/Test Commands

### Environment Setup (ALWAYS REQUIRED)

```powershell
# PowerShell
.venv\Scripts\Activate.ps1

# WSL/Linux  
.venv-ubuntu/bin/activate
```

**IMPORTANT**: Never run Python commands without activating the virtual environment first.

### Linting Commands
```bash
# Check specific file
ruff check agent_ng/langchain_agent.py

# Check modified files in agent_ng/
ruff check agent_ng/

# Check tools directory
ruff check tools/

# Auto-fix with unsafe fixes (when needed)
ruff check --fix --unsafe-fixes agent_ng/

# Using the custom linting script (preferred)
python lint.py file.py                    # Specific files
python lint.py --staged                   # Staged files only
python lint.py --changed                   # Changed vs HEAD
```

### Testing Commands
```bash
# Run all tests
python -m pytest agent_ng/_tests/

# Run single test file
python -m pytest agent_ng/_tests/test_token_counter.py

# Run single test class
python -m pytest agent_ng/_tests/test_token_counter.py::TestTokenCount

# Run single test method
python -m pytest agent_ng/_tests/test_token_counter.py::TestTokenCount::test_token_count_creation

# Run with verbose output
python -m pytest agent_ng/_tests/test_token_counter.py -v

# Run tests by pattern
python -m pytest agent_ng/_tests/ -k "test_token"
```

### Development Commands
```bash
# Start the Gradio application
python agent_ng/app_ng_modular.py

# PowerShell management script (comprehensive)
.\run-agent.ps1                    # Start background process
.\run-agent.ps1 -Action status     # Check status
.\run-agent.ps1 -Action tail       # Follow logs
.\run-agent.ps1 -Action stop       # Stop agent
.\run-agent.ps1 -Action restart    # Restart agent
.\run-agent.ps1 -Port 8080         # Custom port
.\run-agent.ps1 -AutoPort          # Auto-detect available port
```

## Code Style Guidelines

### Import Organization (Strict Hierarchy)
```python
# 1. Standard library imports
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# 2. Third-party imports
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

# 3. Local imports (relative preferred, with fallbacks)
from .utils import ensure_valid_answer
from .error_handler import get_error_handler
tools.file_utils import FileUtils
```

**Import Best Practices:**
- Always place imports at the top
- Use a consistent import hierarchy
- Implement robust fallback handling for relative imports:
```python
try:
    from .utils import get_tool_call_count
except ImportError:
    from agent_ng.utils import get_tool_call_count
```
- Use conditional imports for optional dependencies with availability flags

### Naming Conventions

- **Classes**: PascalCase (`ChatTab`, `LLMManager`, `TokenCount`)
- **Functions/Variables**: snake_case (`get_token_tracker`, `format_cost`)
- **Constants**: UPPER_SNAKE_CASE (`TOKEN_STATUS_CRITICAL`)
- **Private**: Leading underscore (`_create_chat_interface`)
- **Modules**: snake_case (`langchain_agent.py`)

### Type Annotations

- Comprehensive type hints for all function signatures
- Use Union types for optional parameters (`str | None`)
- Generic types with `Dict`, `List`, `Optional`
- Always specify return types for public functions

### Code Quality Standards

- **Line length**: 88 characters (Black-compatible)
- **Quotes**: Double quotes consistently
- **No orphan spaces** on empty lines
- **Flawless code** - reanalyze changes twice for issues
- **DRY principle**: Extract repeated code (2+ times) into helpers

## Design Principles

- **TDD:** Write tests first, define behavior contracts before implementation
- **SDD:** Plan with specs, understand requirements before coding
- **Non-breaking:** Never break existing functionality - verify all endpoints still work
- **Lean:** Minimal code, no overengineering, delete unnecessary complexity
- **Pythonic:** Follow Python idioms, use standard library, prefer clarity over cleverness
- **Brilliant:** Simple solutions that work, not complex ones that impress
- **Extensibility:** Ensure testability and extensibility

## 12-Factor App Principles

Based on [12-factor app](https://12factor.net/) + [12-factor AI agents](https://github.com/humanlayer/12-factor-agents) methodology:

- **Codebase:** One codebase tracked in revision control, many deploys.
- **Dependencies:** Declare all dependencies explicitly in `requirements.txt` and `pyproject.toml`. Use virtual environment isolation.
- **Config:** Store all environment-specific config in env vars (never in code). Use `.env` files for local dev, ensure codebase could be open-sourced without compromising credentials.
- **Backing Services:** Treat databases, caches, vector stores as attached resources accessed via URLs/credentials. No distinction between local and third-party in code.
- **Build, Release, Run:** Strictly separate build and run stages.
- **Processes:** Execute as stateless processes. Session data in backing services.
- **Port Binding:** Export services via port binding. Self-contained, port specified via env var.
- **Concurrency:** Scale via process model. Use Gradio concurrency limits and thread pools.
- **Disposability:** Fast startup, graceful shutdown on SIGTERM.
- **Dev/Prod Parity:** Keep dev, staging, prod similar.
- **Logs:** Treat as event streams. Prefer stdout for containerized.
- **Admin Processes:** Run as one-off processes using same codebase and config.

## Verification Checklist

Before considering work complete:

1. Tests pass
2. Lint passes
3. Shared logic (DRY)
4. No breakage of existing functionality

## Error Handling Patterns

### Multi-Layered Error Handling

```python
# 1. Graceful import fallbacks
try:
    from .utils import get_tool_call_count
except ImportError:
    from agent_ng.utils import get_tool_call_count

# 2. Comprehensive error classification
def ensure_valid_answer(answer: Any) -> str:
    if answer is None:
        return "No answer provided"
    elif not isinstance(answer, str):
        return str(answer)
    return answer

# 3. Never use empty except blocks
try:
    risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")  # Always add logging
```

### Error Handling Rules

- **No silent exceptions** - always add logging or fail explicitly
- **No nested exceptions** - they're ugly and non-debuggable  
- **Validate external data** thoroughly before processing
- **Use safe defaults** for optional fields (0.0, None, empty collections)
- **Centralize error handling** logic rather than scattering throughout code
- **Handle multiple response formats** gracefully (dict vs object, different field names)

## Framework-Specific Conventions

### LangChain Patterns

- Use pure LangChain patterns (chains, memory, tools)
- Follow LangChain Expression Language (LCEL)
- Support streaming with `astream()` and `astream_events()`
- Use Pydantic models for tool parameters
- Maintain conversation context across turns
- **LangChain purity** is mandatory - no competing patterns

### Gradio Patterns

- Ensure Gradio purity when possible
- Use Gradio's i18n system for internationalization
- Follow Gradio component patterns
- Implement proper state management

## Testing Guidelines

### Test Behavior, Not Implementation

**Core Principles:**

- Tests should validate what code **does**, not **how** it does it
- Test outcomes, not mechanisms
- Avoid hardcoded values in assertions
- Test behavior contracts (inputs → outputs)
- Use mocks judiciously - mock external dependencies, not internal details

**Benefits:**

- Tests remain valid when implementation changes
- Tests document intended functionality
- Easier to refactor without breaking tests
- Tests serve as specifications

### What to Test
- Test behaviors users care about, not implementation details
- Focus on: error handling, data integrity, user-facing functionality
- Cover edge cases: boundary conditions, missing data, invalid inputs
- Do NOT test: internal state management, singleton implementations, framework internals

### Test Organization
```python
class TestFeatureName:
    """Descriptive docstring"""
    
    def test_specific_behavior(self):
        """Test basic token count creation"""
        count = TokenCount(100, 50, 150, False, "tiktoken", 0.0)
        assert count.input_tokens == 100
        assert count.total_tokens == 150
```

### Test File Location
- Primary location: `agent_ng/_tests/`
- Alternative: relevant `cwd/_tests` where files are modified
- Fallback: `.misc_files` for miscellaneous tests

### Integration Tests

Use pytest markers for slow/integration tests:

```bash
pytest -m "not slow"         # Run fast unit tests only
pytest -m integration        # Run integration tests only
```

### Cross-Model Testing

Test multiple models/configurations with same logic. Use parametrized tests for different configs or inputs.

### Shared Logic Verification

When multiple endpoints compute the same thing, verify they produce identical results.

## Architectural Patterns

### Manager Pattern

Use centralized managers for core functionality:
```python
# Example managers in the codebase
LLMManager()           # LLM provider management
MemoryManager()        # Conversation memory
StatsManager()         # Statistics tracking
SessionManager()       # User session handling
```

### Adapter Pattern

Provider-specific adapters for LLM integrations:
```python
# Provider adapters handle different APIs
GeminiAdapter, GroqAdapter, MistralAdapter, etc.
```

### Factory Pattern

Used for creating LLM instances and token counters:
```python
def create_llm(provider: str, use_tools: bool = True) -> LLMInstance
def get_token_tracker() -> ConversationTokenTracker
```

## Documentation Standards

### File Organization

- Use environment variables for secrets (see `.env.example`)
- Put reports to `docs/**/progress_reports/` folder
- Use GitHub Markdown format with `YYYYMMDD_` filename prefix

### Docstring Patterns
```python
"""
Module docstring with key features and usage examples.

Key Features:
- Feature 1 description
- Feature 2 description

Usage:
    manager = LLMManager()
    llm = manager.get_llm("gemini")
"""
```

### Documentation Structure

- **Clear hierarchy:** Use consistent heading levels (H1 → H2 → H3). One H1 per file.
- **Front-load value:** Put key conclusions and recommendations first.
- **SCQA framework:** Situation (Ситуация) → Complication (Проблема) → Question (Вопрос для решения) → Answer (Рекомендуемый ответ) for executive summaries or report intros.
- **Chunked content:** Use bulleted lists, short paragraphs, and visual breaks. Avoid walls of text.
- **Actionable sections:** Each section should answer "So what?" and lead to a decision or next step.

## Agent Behavior & Commit Discipline

**Work Tracking:** Always write down as files to maintain context and avoid losing track:
- **Plans:** Actionable step-by-step plans with checkpoints → `.opencode/plans/YYYYMMDD_<topic>/plan.md`
- **Research:** Research results and findings → `.opencode/research/YYYYMMDD_<topic>/research.md`
- **Progress:** Progress reports for ongoing work → `.opencode/progress_reports/YYYYMMDD_<topic>/progress_YYYYMMDD.md`
- Use GitHub Markdown format. Parent folder dated with inception timestamp.

**Commit Discipline:** Do NOT create or push commits unless explicitly asked by the user.

### Commit Messages

- **Format:** Concise, structured, and strictly relevant to the changes.
- **Content:** Keep length to the necessary minimum. Avoid fluff.

### Agent Instructions

- **Information Gathering:** Search docs/internet before coding. Digest best practices from official sources. Scan doc hierarchy, not just 1-2 pages.
- **Planning:** PLAN your course of action before implementing. Write a plan to `.opencode/plans/` that is lean, dry, actionable, step-by-step, verifiable, and follows TDD-first best practices.
- **Verification:** Run `ruff check <modified_file>` after changes. Run relevant tests. Reanalyze changes twice for introduced issues.
- **Secrets:** NEVER hardcode secrets. Use environment variables. NEVER commit `.env` files to version control.
- **No Breakage:** Never break existing code.
- **Memory:** Compact memory proactively once in a while during conversation rather than waiting for overflow — prevent "dementia" by summarizing and pruning before resources exhaust.

## Research & Planning (Required)

Before any coding:

1. **Gather information**: Search for reference documentation on frameworks/libraries
2. **Go to official sources**: Digest best practices from official docs
3. **Comprehensive review**: Scan doc hierarchy, not just 1-2 pages
4. **Plan your course**: Base actions on ground truth after research

## UI/UX Principles

- **Clarity over clutter**: Remove redundant elements
- **Maximize data-ink ratio**: Every element must add value
- **Visual hierarchy**: Group related information with consistent formatting
- **Progressive disclosure**: Essential info prominently, details on demand
- **Data integrity**: Display zero values when meaningful

## Key Dependencies

### Core Stack

- `langchain>=0.3.27` - Primary framework
- `gradio>=5.49.1` - UI framework with i18n
- `pydantic>=2.11.10` - Data validation
- `tiktoken>=0.12.0` - Token counting

### Development Tools

- `ruff>=0.14.0` - Linting and formatting
- `pytest>=8.4.2` - Testing framework
- `python-dotenv` - Environment configuration

### Observability

- `langsmith>=0.4.34` - Tracing and debugging
- `langfuse>=3.6.2` - Alternative observability

---

**Remember**: This codebase follows LangChain-pure, super dry, super lean, abstract, modular, pythonic patterns. Always research first, plan thoroughly, and produce flawless code that respects established architectural patterns.