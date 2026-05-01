# Guiding Principles

Follow these principles for **all** platform operations:

1. **Persist Context** — Always save complete schemas and results to `cmw-platform-workspace/` before making changes. This provides recovery points and prevents context loss.

2. **Read Before Write** — Fetch current state first, save it, then modify. Never edit without reading and persisting first.

3. **Idempotent Operations** — Design operations so running them multiple times produces the same result.

4. **Explicit Over Implicit** — If you provide a value, it overrides. If you don't, the patch preserves existing.

---

→ Save pattern and file naming: [working_files.md](working_files.md)

→ Patch mechanism and validation rules: [edit_or_create.md](edit_or_create.md)
