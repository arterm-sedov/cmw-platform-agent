# AGENTS.md - CMW Platform Agent

Repo-specific guidance for this LangChain + Gradio Python 3.11+ project.

## Research & Planning

Before any coding, changes or implementation:

- Do a deep codebase research.
- Do a deep web research in the internet for reference documentation on frameworks, libraries, and best practices.
- Gather all information before planning course of action.
- Plan after gathering reference information, not before.
- Write a concise plan **file**:
    - actionable
    - detailed
    - TDD
    - step-by-step tasks
    - checkpoints
    - expected verification commands
    - follows best practices in TDD, SDD, Python, 12-factor agents and software

## Common Engineering Baseline and Design Principles

- Follow SDD for scope/contract clarity and TDD for behavior-first implementation.
- Keep code: lean, DRY, modular, and non-breaking, brilliant, abstract, minimal, genius.
- Follow best practices:
    - **TDD:** Write tests first, define behavior contracts. Test behavior, not implementation details.
    - **SDD:** Plan with specs, understand requirements before coding.
    - **Non-breaking:** Never break existing functionality.
    - **Lean:** Minimal code, no overengineering.
    - **Pythonic:** Follow Python idioms, prefer clarity over cleverness, explicit data contracts, strong typing.
    - **Modular:** Single responsibility, group related functionality.
    - **Open/Closed:** Design for extension without modification.
    - **DRY:** 2+ uses -> extract to helper, super dry, super lean.
- For LangChain see LangChain docs, repo and source code for reference, prefer LCEL/runnables, typed tool schemas, and streaming-safe patterns.
- For Gradio see Gradio's docs, repo and source code for reference, follow best practices, keep state/event flow explicit and UI logic separated from domain logic.
- Validate external data and avoid silent exception handling (`except: pass` is forbidden).
- Run lint and relevant tests for modified areas before completion.
- Never hardcode secrets; use environment variables and `.env.example` placeholders only.

## Dev Commands

```bash
# Activate venv first (required)
# Use the active/default Cursor terminal session first
source .venv/bin/activate   # Linux/WSL
.venv-ubuntu/bin/activate   # WSL/Linux (alternate venv)
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


## Code Style & Conventions

- **Ruff (pyproject.toml):** Line length 88, Python 3.11+, double quotes. Many rules are `unfixable` - ruff flags but will not auto-fix (F401, F403, T201, PLR, ANN, etc). Run `ruff check --fix --unsafe-fixes` to auto-fix.
- **Imports:** Standard library -> third-party -> local with fallback pattern:
```python
try:
    from .utils import helper
except ImportError:
    from agent_ng.utils import helper
```
- **Naming:** Classes PascalCase, functions/variables snake_case, constants UPPER_SNAKE, private prefix `_`.
- **Docstrings:** Module docstrings with key features and usage examples.

## Framework Conventions

- **LangChain:** Pure patterns, LCEL, streaming with `astream()`, Pydantic for tool params.
- **LangChain References:** https://python.langchain.com/docs/ - Streaming, Runnables, Tool Calling, LCEL.
- **Gradio:** Use i18n system, follow component patterns, proper state management.
- **Gradio References:** https://www.gradio.app/docs - Components, State, Event Listeners.

## Error Handling

- **No silent exceptions:** Always add logging (`except: pass` is forbidden).
- **No nested exceptions:** Avoid nested handlers that reduce debuggability.
- **No unnecessary try-catches:** Add only when helpful.
- **Validate external data:** Check `response.ok`, validate structure, handle missing fields.
- **Safe defaults:** Use 0.0, None, empty collections where appropriate.
- **Handle multiple response formats:** Support dict/object and variant field names.
- **Centralize validation/error handling logic.**

## Testing Guidelines

- Test **behavior**, not implementation details.
- Focus on error handling, data integrity, and user-facing functionality.
- Cover edge cases: boundary conditions, missing data, invalid inputs.
- Location: `agent_ng/_tests/` or relevant `cwd/_tests`.
- Do not test irrelevant patterns (internal state, singletons, framework internals).
- Keep tests DRY with parametrization when validating the same behavior across multiple configs/models.
- Prefer integration tests for endpoint-level behavior; use pytest markers:
  - `python -m pytest -m "not slow"` for fast unit checks.
  - `python -m pytest -m integration` for integration checks.
- When multiple endpoints share the same computation contract, add tests that verify equivalent outcomes.
- References:
  - https://google.github.io/googletest/primer.html
  - https://www.ibm.com/think/insights/unit-testing-best-practices

## Verification Checklist

Before considering work complete:

1. Run `ruff check` on modified files.
2. Run relevant tests (unit/integration as applicable).
3. Confirm no user-facing regressions (non-breaking behavior).
4. Ensure shared logic remains DRY (extract helpers for repeated blocks).
5. Update docs/README when behavior, workflows, or commands changed.

## Documentation Guidelines

- Use clear heading hierarchy (single H1 per file).
- Front-load conclusions and recommendations.
- Use actionable, chunked sections (avoid walls of text).
- Keep source traceability for claims (inline citation where relevant).
- Add a blank line after headings and before lists in Markdown.
- Keep sections action-oriented: each section should clearly imply a decision or next step.
- Documentation files go to `docs/`.
- Progress reports go to `docs/**/progress_reports/` with `YYYYMMDD_` prefix.
- Generate `YYYYMMDD` timestamps with native commands:
  - PowerShell: `Get-Date -Format "yyyyMMdd"`
  - Bash/WSL: `date +%Y%m%d`
  - Python (with active venv): `python -c "from datetime import datetime; print(datetime.now().strftime('%Y%m%d'))"`

## Security & Secrets

- Use `.env` files for local config, never commit secrets.
- Never commit `.env`; use `.env.example` with placeholders only.
- Never commit or include in any code or docs passwords, secrets, keys, or real personal/business/entity names.
- Use agnostic synthetic neutral placeholders for sensitive information.
- Load secrets via dotenv in tests and code. Never hardcode sensitive information.

## Commit Guidelines

- Only create commits when explicitly asked.
- If asked to only draft commit text, generate the message but do not add files, stage, push, or commit.
- Generate commit message text, but do NOT add files, stage, or push unless requested.
- Keep messages concise, structured, and strictly relevant to changes.
- Keep commit message length to the necessary minimum.
- Avoid blabber.

## Work Tracking & Reports

- Plans: create and maintain a concise plan file before implementation.
- Research notes: capture key official references and decisions.
- Progress reports: store under `docs/**/progress_reports/` with `YYYYMMDD_` prefix.
- Documentation files go to `docs/`.

## UI/UX Principles

- **Clarity over clutter:** Remove redundant elements.
- **Maximize data-ink ratio:** Every element adds value.
- **Visual hierarchy:** Group related information consistently.
- **Progressive disclosure:** Essential info prominent, details on demand.
- **Data integrity:** Display zero values when meaningful, not just absence.

## 12-Factor App Principles

Based on https://12factor.net/ and https://github.com/humanlayer/12-factor-agents:

- One codebase, many deploys.
- Declare dependencies explicitly.
- Keep config in environment variables, not hardcoded.
- Treat backing services as attached resources.
- Separate build, release, and run stages.
- Prefer stateless processes with session state in backing services.
- Export services via port binding where applicable.
- Scale via the process model where applicable.
- Optimize disposability (fast startup, graceful shutdown).
- Keep dev/prod parity.
- Treat logs as event streams.
- Run admin tasks as one-off processes.

## Repo-Specific Details

### Agent Behavior

- Reanalyze changes twice for introduced issues.
- Compact/summarize working context proactively during long sessions.

### CMW Platform Terminology

**Critical:** Never expose legacy API terms to LLMs. Use human-readable terms:

| API Term | LLM-Friendly |
|----------|--------------|
| alias | system name |
| instance | record |
| user command | button |
| container | template |
| property | attribute |
| dataset | table |

### CMW Platform Architecture

**Key Concept:** Datasets, Toolbars, and Buttons are separate API entities with different endpoints:

| Entity | Tool to Get | Tool to Edit | Endpoint Pattern |
|--------|-------------|--------------|-------------------|
| Dataset | `get_dataset` | `edit_or_create_dataset` | `webapi/Dataset/{app}/Dataset@{tpl}.{dataset}` |
| Toolbar | `get_toolbar` | `edit_or_create_toolbar` | `webapi/Toolbar/{app}/Toolbar@{tpl}.{toolbar}` |
| Button | `get_button` | `edit_or_create_button` | `webapi/Button/{app}/Button@{tpl}.{button}` |

**Toolbar-Dataset Link:** Toolbars link to datasets via toolbar's `IsDefaultForLists` flag.

### Key Dependencies

- Source of truth: `requirements.txt` (versions may change over time).
- Key packages to be aware of: `langchain`, `gradio`, `pydantic`, `ruff`, `pytest`, `tiktoken`.

---

**Remember:** LangChain-pure, DRY, lean, modular, pythonic patterns. Always research first, plan thoroughly, produce flawless code.

## Agent Environment Notes

Non-obvious caveats. For standard setup, run, lint, and test commands see the **Development Setup** section in `README.md`.

- On startup, the app fetches OpenRouter model pricing via HTTP (~10-15 s of network calls). This is normal, not an error.
- The codebase has ~3 k pre-existing `ruff` findings; many are intentionally unfixable per `pyproject.toml`. Only focus on new findings in files you change.
- On a clean first start with dummy/empty `.env`, some tests may fail or error due to missing credentials or import issues. This is expected; configure real API keys and check test prerequisites before investigating failures.
