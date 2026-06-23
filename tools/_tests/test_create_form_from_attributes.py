import json
from pathlib import Path

from tools.templates_tools.form_attribute_groups import (
    infer_form_groups,
    is_system_attribute,
)
from tools.templates_tools.form_structure import count_field_components
import tools.templates_tools.tool_create_form_from_attributes as create_module
from tools.templates_tools.tool_create_form_from_attributes import (
    create_form_from_attributes,
)

FIXTURES = Path(__file__).parent / "fixtures" / "forms"


def load_fixture(name: str) -> list:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_create_from_attributes_excludes_system_fields():
    attributes = load_fixture("list_attributes_user_and_system.json")
    groups = infer_form_groups(attributes)
    included = [
        attribute["system_name"]
        for group in groups
        for attribute in group["attributes"]
    ]

    assert is_system_attribute(attributes[0]) is True
    assert included == ["BusinessName", "Comment"]


def test_create_form_from_attributes_tool_builds_visible_form(monkeypatch):
    attributes = load_fixture("list_attributes_user_and_system.json")

    class FakeListAttributes:
        @staticmethod
        def invoke(payload):
            _ = payload
            return {
                "success": True,
                "status_code": 200,
                "data": attributes,
                "error": None,
            }

    monkeypatch.setattr(create_module, "list_attributes", FakeListAttributes())
    monkeypatch.setattr(
        create_module,
        "replace_form_delete_post",
        lambda **kwargs: {
            "success": True,
            "status_code": 200,
            "data": kwargs["new_form"],
            "error": None,
        },
    )

    result = create_form_from_attributes.invoke(
        {
            "application_system_name": "App",
            "template_system_name": "TargetTemplate",
            "form_system_name": "defaultForm",
            "form_name": "Default form",
        }
    )

    assert result["success"] is True
    assert result["included_attributes"] == ["BusinessName", "Comment"]
    assert count_field_components(result["data"]) == 2
