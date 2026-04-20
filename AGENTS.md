# AGENTS.md - CMW Platform Agent Coding Guidelines

AI coding agent guidelines for this LangChain-based Python 3.11+ project.

## Build/Lint/Test Commands

**Environment Setup (ALWAYS REQUIRED):**
- `.venv\Scripts\Activate.ps1` (PowerShell)
- `.venv-ubuntu/bin/activate` (WSL/Linux)

**Linting:**
```bash
ruff check <file_path>                    # Check specific file
ruff format <file_path>                  # Format specific file
python lint.py file.py                    # Custom lint script
python lint.py --staged                   # Staged files only
python lint.py --changed                  # Changed vs HEAD (default)
python lint.py --all                      # Full repository
```

**Type Checking:**
```bash
mypy <file_path>                          # Type check specific file
mypy agent_ng/                            # Type check module
```

**Testing:**
```bash
python -m pytest agent_ng/_tests/                          # All tests
python -m pytest agent_ng/_tests/test_file.py             # Single file
python -m pytest agent_ng/_tests/test_file.py::ClassName  # Single test class
python -m pytest agent_ng/_tests/test_file.py::test_method  # Single test method
python -m pytest agent_ng/_tests/ -k "pattern"            # By pattern match
```

**Run App:** `python agent_ng/app_ng_modular.py`

## Code Style Guidelines

**Imports (3 groups):** Standard library → Third-party → Local with fallback
```python
try:
    from .utils import helper
except ImportError:
    from agent_ng.utils import helper
```

**Naming:** Classes PascalCase, functions/variables snake_case, constants UPPER_SNAKE, private prefix `_`
**Code Quality:** Line 88 chars, double quotes, no orphan spaces, DRY (2+ uses → helper), run linter after changes
**Imports always on top**, consistent formatting, produce flawless code
**Line Limit:** Maximum 88 characters per line

## Research & Planning

- Before coding, search internet for reference documentation on frameworks/libraries
- Go to official documentation sources and digest best practices
- Scan docs hierarchy, don't just read a page or two
- Gather all information before planning course of action
- PLAN after gathering reference information, not before

## Design Principles

- **TDD:** Write tests first, define behavior contracts
- **SDD:** Plan with specs, understand requirements before coding
- **Non-breaking:** Never break existing functionality
- **Lean:** Minimal code, no overengineering
- **Pythonic:** Follow Python idioms, prefer clarity over cleverness
- **Modular:** Single responsibility, group related functionality
- **Open/Closed:** Design for extension without modification
- **Research first:** Search docs/internet before coding, scan full doc hierarchy
- **Abstraction:** Use interfaces and abstractions for extensibility

## Error Handling

- **No silent exceptions** - always add logging
- **No nested exceptions** - ugly and non-debuggable
- **No unnecessary try-catches** - only add when helpful
- **Validate external data** before processing
- **Safe defaults** for optional fields (0.0, None, empty collections)
- Handle multiple response formats gracefully (dict vs object)
- Centralize error handling and validation logic

## Framework Conventions

**LangChain:** Pure patterns, LCEL, streaming with `astream()`, Pydantic for tool params
**LangChain References:** https://python.langchain.com/docs/ - Streaming, Runnables, Tool Calling, LCEL
**Gradio:** Use i18n system, follow component patterns, proper state management
**Gradio References:** https://www.gradio.app/docs - Components, State, Event Listeners

## Testing Guidelines

- Test **behavior**, not implementation
- Focus on: error handling, data integrity, user-facing functionality
- Cover edge cases: boundary conditions, missing data, invalid inputs
- Location: `agent_ng/_tests/` or relevant `cwd/_tests`
- Do not test irrelevant patterns (internal state, singletons, framework internals)

## CMW Platform Terminology

**Critical:** Never expose legacy API terms to LLMs. Use human-readable terms:

| API Term | LLM-Friendly |
|----------|--------------|
| alias | system name |
| instance | record |
| user command | button |
| container | template |
| property | attribute |

## CMW Platform Architecture

**Key Concept:** Datasets, Toolbars, and Buttons are **separate API entities** with different endpoints:

| Entity | Tool to Get | Tool to Edit | Endpoint Pattern |
|--------|-------------|--------------|-------------------|
| Dataset | `get_dataset` | `edit_or_create_dataset` | `webapi/Dataset/{app}/Dataset@{tpl}.{dataset}` |
| Toolbar | `get_toolbar` | `edit_or_create_toolbar` | `webapi/Toolbar/{app}/Toolbar@{tpl}.{toolbar}` |
| Button | `get_button` | `edit_or_create_button` | `webapi/Button/{app}/Button@{tpl}.{button}` |

**Toolbar-Dataset Link:** Toolbars link to datasets via toolbar's `IsDefaultForLists` flag.

## Key Dependencies

- `langchain>=0.3.27` - Core framework
- `gradio>=5.49.1` - UI
- `pydantic>=2.11.10` - Validation
- `ruff>=0.14.0` - Linting
- `pytest>=8.4.2` - Testing
- `tiktoken>=0.12.0` - Token counting

## Commit Guidelines

- Only create commits when explicitly asked
- Keep messages concise, structured, strictly relevant to changes
- Never add files, stage, or push - only generate commit message text

## UI/UX Principles

- **Clarity over clutter** - Remove redundant elements
- **Maximize data-ink ratio** - Every element adds value
- **Visual hierarchy** - Group related information consistently
- **Progressive disclosure** - Essential info prominent, details on demand
- **Data integrity** - Display zero values when meaningful, not just absence

## File Organization

- Tests go to `agent_ng/_tests/` or relevant `cwd/_tests`
- Module docstrings with key features and usage examples
- Use `.env` files for local config, never commit secrets
- Progress reports to `docs/**/progress_reports/` with `YYYYMMDD_` prefix
- Documentation files to `docs/` folder

## Terminal Commands

- ALWAYS activate venv before running Python commands
- PowerShell: `.venv\Scripts\Activate.ps1`
- WSL/Linux: `.venv-ubuntu/bin/activate`
- Do not execute Python without active venv

## Python References

- PEP 8: https://peps.python.org/pep-0008/
- PEP 257: https://peps.python.org/pep-0257/
- Google Style: https://google.github.io/styleguide/pyguide.html

---

**Remember:** LangChain-pure, dry, lean, modular, pythonic patterns. Always research first, plan thoroughly, produce flawless code.