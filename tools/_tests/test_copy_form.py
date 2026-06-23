import json
from pathlib import Path

from tools.templates_tools.form_structure import count_field_components
import tools.templates_tools.tool_copy_form as copy_module
from tools.templates_tools.tool_copy_form import (
    build_copied_form,
    copy_form_from_template,
)

FIXTURES = Path(__file__).parent / "fixtures" / "forms"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_build_copied_form_rewrites_source_aliases():
    source = load_fixture("form_source_copy.json")

    copied = build_copied_form(
        source_form=source,
        source_application_system_name="SourceApp",
        source_template_system_name="SourceTemplate",
        target_template_system_name="TargetTemplate",
        target_form_system_name="defaultForm",
        replace_tokens={"Src_": "Tgt_"},
    )
    serialized = json.dumps(copied, ensure_ascii=False)

    assert "SourceTemplate" not in serialized
    assert "Src_" not in serialized
    assert "TargetTemplate" in serialized
    assert "Tgt_Name" in serialized
    assert count_field_components(copied) == count_field_components(source)


def test_copy_form_from_template_exports_tool_path(monkeypatch):
    source = load_fixture("form_source_copy.json")

    def fake_get_form_raw(*_args):
        return source

    monkeypatch.setattr(copy_module, "get_form_raw", fake_get_form_raw)
    monkeypatch.setattr(
        copy_module,
        "replace_form_delete_post",
        lambda **kwargs: {
            "success": True,
            "status_code": 200,
            "data": kwargs["new_form"],
            "error": None,
        },
    )

    result = copy_form_from_template.invoke(
        {
            "source_application_system_name": "SourceApp",
            "source_template_system_name": "SourceTemplate",
            "target_application_system_name": "TargetApp",
            "target_template_system_name": "TargetTemplate",
            "replace_tokens": {"Src_": "Tgt_"},
        }
    )

    assert result["success"] is True
    assert count_field_components(result["data"]) == 1
