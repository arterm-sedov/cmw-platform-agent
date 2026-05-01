# System Prompt Alignment

For all platform operations, follow this precedence:

## 1. `agent_ng/system_prompt.json` — PRIMARY (agentic behavior)

- Intent → Plan → Validate → Execute → Result workflow
- Tool usage discipline (no duplicate calls, cache results)
- CMW Platform vocabulary and terminology
- Idempotency and confirmation rules
- Error handling (401/403, 404, 409, 5xx)

**Always use this for platform-specific guidance.**

## 2. `AGENTS.md` — SECONDARY (project work guidelines)

- TDD/SDD principles
- Non-breaking changes
- Lean/DRY patterns
- Logging and persistence to workspace

**General development approach — not platform-specific.**

## For Platform Operations: 4-Step Checklist

1. Follow the Intent → Plan → Validate → Execute → Result workflow from `system_prompt.json`.
2. Use CMW Platform terminology (`alias` = system name, `instance` = record, etc.).
3. Confirm risky edits before execution.
4. Present results in human-readable format, not raw JSON.
