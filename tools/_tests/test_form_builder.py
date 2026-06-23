from tools.templates_tools.form_builder import (
    build_field_component,
    build_form_body,
    build_root_from_widgets,
)
from tools.templates_tools.form_structure import (
    count_field_components,
    list_referenced_attribute_aliases,
)


def test_build_root_from_widgets_creates_field_components():
    root = build_root_from_widgets(
        template_system_name="TargetTemplate",
        form_system_name="defaultForm",
        widgets={"BusinessName": {"label": "Business name"}},
    )
    form = {"root": root}

    assert count_field_components(form) == 1
    assert list_referenced_attribute_aliases(form) == {"BusinessName"}


def test_build_field_component_uses_prototype_but_rewrites_attribute():
    prototype = {
        "type": "FieldComponent",
        "propertyPath": [{"type": "Attribute", "owner": "Old", "alias": "OldName"}],
        "editorType": "Text",
    }

    field = build_field_component(
        template_system_name="TargetTemplate",
        form_system_name="defaultForm",
        attribute_system_name="BusinessName",
        label="Business name",
        prototype=prototype,
    )

    assert field["editorType"] == "Text"
    assert field["propertyPath"][0]["owner"] == "TargetTemplate"
    assert field["propertyPath"][0]["alias"] == "BusinessName"


def test_build_form_body_has_form_alias_and_container():
    body = build_form_body(
        template_system_name="TargetTemplate",
        form_system_name="defaultForm",
        name="Default form",
        root={"type": "VerticalLayout", "rows": []},
    )

    assert body["globalAlias"]["alias"] == "defaultForm"
    assert body["container"]["alias"] == "TargetTemplate"
