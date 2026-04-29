# AGENTS.md - CMW Platform Agent

Repo-specific guidance for this LangChain + Gradio Python 3.11+ project.

## Planning

Before any coding, changes or implementation:

- Do a deep codebase research.
- Do a deep web research.
- Write a concise plan **file**:
    - actionable
    - detailed
    - TDD
    - step-by-step tasks
    - checkpoints
    - expected verification commands
    - follows best practices in TDD, SDD, Python, 12-factor agents and software

## Common Engineering Baseline

Use these rules:

- Follow SDD for scope/contract clarity and TDD for behavior-first implementation.
- Keep code: lean, DRY, modular, and non-breaking, brilliant, abstract, minimal, genius.
- Follow best practices in TDD, SDD, Gradio, and LangChain implementations.
- Prefer Pythonic solutions: clarity over cleverness, explicit data contracts, strong typing.
- For LangChain see LangChain docs, repo and source code for reference, prefer LCEL/runnables, typed tool schemas, and streaming-safe patterns.
- For Gradio see Gradio's docs, repo and source code for reference, follow best practices, keep state/event flow explicit and UI logic separated from domain logic.
- Test behavior, not implementation details.
- Validate external data and avoid silent exception handling (`except: pass` is forbidden).
- Run lint and relevant tests for modified areas before completion.
- Never hardcode secrets; use environment variables and `.env.example` placeholders only.

## Dev Commands

```bash
# Activate venv first (required)
source .venv/bin/activate   # Linux/WSL
.venv\Scripts\Activate.ps1  # PowerShell

# Run app
python agent_ng/app_ng_modular.py

# Lint (custom script - default changes vs HEAD)
python lint.py                    # Changed files (default)
python lint.py --staged           # Staged files
python lint.py --all              # Full repo
ruff check <file.py>

# Typecheck
mypy agent_ng/

# Test
python -m pytest agent_ng/_tests/           # All tests
python -m pytest agent_ng/_tests/test_x.py   # Single file
python -m pytest -k "pattern"              # Filter by name
```

## Project Structure

- **Entry:** `agent_ng/app_ng_modular.py`
- **Core agent:** `agent_ng/langchain_agent.py`
- **Tools:** `tools/` (49 tools)
- **Tests:** `agent_ng/_tests/`
- **Config:** `.env.example` for API keys

## Code Style

**Ruff (pyproject.toml):** Line length 88, Python 3.11+, double quotes. Many rules are `unfixable` - ruff flags but won't auto-fix (F401, F403, T201, PLR, ANN, etc). Run `ruff check --fix --unsafe-fixes` to auto-fix.

**Imports:** Standard library → Third-party → Local with fallback pattern:
```python
try:
    from .utils import helper
except ImportError:
    from agent_ng.utils import helper
```
**Naming:** Classes PascalCase, functions/variables snake_case, constants UPPER_SNAKE, private prefix `_`

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
- **DRY:** 2+ uses → extract to helper, super dry, super lean

## Error Handling

- **No silent exceptions** - always add logging (empty `except: pass` is forbidden)
- **No nested exceptions** - ugly and non-debuggable
- **No unnecessary try-catches** - only add when helpful or adds value
- **Validate external data** before processing (check response.ok, validate structure)
- **Safe defaults** for optional fields (0.0, None, empty collections)
- **Handle multiple response formats** gracefully (dict vs object, different field names)
- **Centralize error handling** and validation logic

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
- Keep tests DRY with parametrization when validating the same behavior across multiple configs/models
- Prefer integration tests for endpoint-level behavior; use pytest markers to separate slow/integration suites:
  - `python -m pytest -m "not slow"` for fast unit checks
  - `python -m pytest -m integration` for integration checks
- When multiple endpoints share the same computation contract, add tests that verify equivalent outcomes

## Verification Checklist

Before considering work complete:

1. Run `ruff check` on modified files
2. Run relevant tests (unit/integration as applicable)
3. Confirm no user-facing regressions (non-breaking behavior)
4. Ensure shared logic remains DRY (extract helpers for repeated blocks)
5. Update docs/README when behavior, workflows, or commands changed

## CMW Platform Terminology

**Critical:** Never expose legacy API terms to LLMs. Use human-readable terms:

| API Term | LLM-Friendly |
|----------|--------------|
| alias | system name |
| instance | record |
| user command | button |
| container | template |
| property | attribute |
| dataset | table |

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

## Related Instruction Files

- **Cursor rules:** `.cursor/rules/cmw-platform-agent.mdc` - Code style and framework guidelines

## Commit Guidelines

- Only create commits when explicitly asked
- Generate commit message text, but do NOT add files, stage, or push
- Keep messages concise, structured, and strictly relevant to changes
- Avoid blabber - keep length to necessary minimum

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
- Never commit `.env`; use `.env.example` with placeholders only
- Never commit or include in any code or docs: any passwords, secrets, keys, real personal, business or entity names, use agnostic synthetic neutral placeholders for for any sensitive information. Load any secrets via load dotenv in tests and code. Never hardcode any sensitive information.
- Progress reports to `docs/**/progress_reports/` with `YYYYMMDD_` prefix
- Documentation files to `docs/` folder

## Documentation Hygiene

- Use clear heading hierarchy (single H1 per file)
- Add a blank line after headings and before lists in Markdown
- Keep sections action-oriented: each section should clearly imply a decision or next step

---

**Remember:** LangChain-pure, DRY, lean, modular, pythonic patterns. Always research first, plan thoroughly, produce flawless code.
