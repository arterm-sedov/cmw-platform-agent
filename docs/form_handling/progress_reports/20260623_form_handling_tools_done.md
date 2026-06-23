# Form Handling Tools Progress Report

## Changed Behavior

- `edit_or_create_form.widgets` now means desired visible fields to upsert.
- Empty forms are rebuilt with `FieldComponent` nodes instead of returning false success.
- Structural form changes use GET/snapshot/DELETE/POST/GET verification with rollback on POST failure.
- New tools expose copy-form and create-from-attributes workflows.

## Tests Added

- Form structure inspection and stale-token validation.
- Form builder output with `FieldComponent.propertyPath`.
- Delete/post rollback behavior.
- Widget JSON parsing, empty-form upsert, no false success, existing edit, and missing-field append.
- Copy form token rewrite.
- Create-from-attributes system field exclusion.

## Verification Status

- `.venv` was updated from `requirements.txt`; `pip check` passed.
- Focused form tests passed: `15 passed`.
- `python lint.py` passed.
- Explicit ruff check over the changed form files and tests passed.
- `python -m pytest -k "form"` is blocked by unrelated collection-time `experiments/test_video_format.py` dependency on a missing local video file.
- `.venv` does not include `mypy`; `.venv312` can run it, but `mypy agent_ng/` currently reports repo-wide existing typing debt unrelated to this form change.
- Real CMW sandbox and browser visual checks require valid platform credentials and a disposable target template.

## Known Limitations

- Generic field creation is conservative and best suited for simple attributes.
- Complex editor-specific metadata should come from a source/prototype form.
- Browser verification remains a final visual proof step after API verification passes.
