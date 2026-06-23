import json
from pathlib import Path

import pytest

from tools.templates_tools.form_structure import (
    assert_no_stale_tokens,
    assert_references_known_attributes,
    count_field_components,
    list_referenced_attribute_aliases,
)

FIXTURES = Path(__file__).parent / "fixtures" / "forms"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_counts_and_extracts_field_components():
    form = load_fixture("form_with_fields.json")

    assert count_field_components(form) == 1
    assert list_referenced_attribute_aliases(form) == {"BusinessName"}


def test_assert_no_stale_tokens_rejects_source_names():
    form = load_fixture("form_source_copy.json")

    with pytest.raises(ValueError, match="SourceTemplate"):
        assert_no_stale_tokens(form, ["SourceTemplate"])


def test_assert_references_known_attributes_rejects_missing_alias():
    form = load_fixture("form_with_fields.json")

    with pytest.raises(ValueError, match="BusinessName"):
        assert_references_known_attributes(form, {"Comment"})
