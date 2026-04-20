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
python lint.py file.py                  # Custom lint script
python lint.py --staged                  # Staged files only
python lint.py --changed                 # Changed vs HEAD (default)
python lint.py --all                     # Full repository
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

### Import Organization (3 groups)
1. Standard library (`asyncio`, `json`, `logging`, `typing`, etc.)
2. Third-party (`gradio`, `langchain`, `pydantic`, etc.)
3. Local imports (relative with fallback)

```python
try:
    from .utils import helper
except ImportError:
    from agent_ng.utils import helper
```

### Naming Conventions
- **Classes**: PascalCase (`ChatTab`, `LLMManager`)
- **Functions/Variables**: snake_case (`get_token_tracker`)
- **Constants**: UPPER_SNAKE_CASE (`TOKEN_STATUS_CRITICAL`)
- **Private**: Leading underscore (`_internal_func`)

### Code Quality
- Line length: 88 chars (Black-compatible)
- Double quotes consistently
- No orphan spaces on empty lines
- DRY: Extract repeated code (2+ times) into helpers
- Reanalyze changes twice before finalizing
- Run linter after every change

## Design Principles

- **TDD:** Write tests first, define behavior contracts
- **SDD:** Plan with specs, understand requirements before coding
- **Non-breaking:** Never break existing functionality
- **Lean:** Minimal code, no overengineering
- **Pythonic:** Follow Python idioms, prefer clarity over cleverness
- **Modular:** Single responsibility, group related functionality

## Error Handling

- **No silent exceptions** - always add logging
- **No nested exceptions** - ugly and non-debuggable
- **No unnecessary try-catches** - only add when helpful
- **Validate external data** before processing
- **Safe defaults** for optional fields (0.0, None, empty collections)
- Handle multiple response formats gracefully (dict vs object)

## Framework Conventions

**LangChain:** Pure patterns, LCEL, streaming with `astream()`, Pydantic for tool params
**Gradio:** Use i18n system, follow component patterns, proper state management

## Testing Guidelines

- Test **behavior**, not implementation
- Focus on: error handling, data integrity, user-facing functionality
- Cover edge cases: boundary conditions, missing data, invalid inputs
- Location: `agent_ng/_tests/`

## CMW Platform Terminology

**Critical:** Never expose legacy API terms to LLMs. Use human-readable terms:

| API Term | LLM-Friendly |
|----------|--------------|
| alias | system name |
| instance | record |
| user command | button |
| container | template |
| property | attribute |

## Agent Workflow

1. **Research first** - Search docs/internet before coding, scan full doc hierarchy
2. **Plan thoroughly** - Write plans to `.opencode/plans/YYYYMMDD_<topic>/plan.md`
3. **Run linter** - Always `ruff check <file>` after changes
4. **No breakage** - Verify existing functionality still works
5. **Env vars** - Use for secrets, never hardcode
6. **Docs** - Put reports to `docs/**/progress_reports/` with `YYYYMMDD_` prefix

## Key Dependencies

- `langchain>=0.3.27` - Core framework
- `gradio>=5.49.1` - UI
- `pydantic>=2.11.10` - Validation
- `ruff>=0.14.0` - Linting
- `pytest>=8.4.2` - Testing
- `tiktoken>=0.12.0` - Token counting

## Commit Guidelines

- Only create commits when explicitly asked by the user
- Keep messages concise, structured, and relevant to changes
- Avoid fluff and unnecessary length

## UI/UX Principles

- **Clarity over clutter** - Remove redundant elements
- **Maximize data-ink ratio** - Every element adds value
- **Visual hierarchy** - Group related information consistently
- **Progressive disclosure** - Essential info prominent, details on demand
- **Data integrity** - Display zero values when meaningful

## File Organization

- Tests go to `agent_ng/_tests/` or relevant `cwd/_tests`
- Module docstrings with key features and usage examples
- Use `.env` files for local config, never commit secrets
- Progress reports to `docs/**/progress_reports/` with `YYYYMMDD_` prefix

## Refactoring & Secrets

- Change only relevant parts when refactoring - never break existing code
- Never hardcode secrets - use environment variables
- Update logging and comments when changing code, don't delete them

## Cursor Rules Integration

Include these rules from `.cursor/rules/`:
- Terminal: Always activate venv before running Python commands
- Docs: Put reports to relevant `docs/**/progress_reports/` folder
- Framework purity: LangChain-pure, Gradio-pure patterns

## Framework References

- LangChain: https://python.langchain.com/docs/
- Gradio: https://www.gradio.app/docs
- Python: https://peps.python.org/pep-0008/
- Google Python Style: https://google.github.io/styleguide/pyguide.html

---

**Remember:** LangChain-pure, dry, lean, modular, pythonic patterns. Always research first, plan thoroughly, produce flawless code.