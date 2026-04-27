"""
Tests for platform_entity_resolver URL parser.

TDD: Tests define behavior contracts before implementation.
"""

import pytest

from tools.platform_entity_resolver import ParsedEntity, _parse_url


class TestParseDesktopUrl:
    def test_desktop_url_no_entities(self):
        parsed = _parse_url("https://bububu.bau.cbap.ru/#desktop/")
        assert parsed.page_type == "desktop"
        assert parsed.entities == []

    def test_desktop_hash_only(self):
        parsed = _parse_url("#desktop/")
        assert parsed.page_type == "desktop"
        assert parsed.entities == []


class TestParseSolutionsUrl:
    def test_solutions_root_no_entities(self):
        parsed = _parse_url("#solutions")
        assert parsed.page_type == "solutions"
        assert parsed.entities == []

    def test_solution_administration(self):
        parsed = _parse_url("#solutions/sln.23/Administration")
        assert parsed.page_type == "solutions"
        assert ParsedEntity("Application", "sln.23") in parsed.entities

    def test_solution_diagram_list(self):
        parsed = _parse_url("#solutions/sln.35/DiagramList/showAll")
        assert parsed.page_type == "solutions"
        assert ParsedEntity("Application", "sln.35") in parsed.entities

    def test_solution_roles(self):
        parsed = _parse_url("#solutions/sln.23/roles")
        assert parsed.page_type == "solutions"
        assert ParsedEntity("Application", "sln.23") in parsed.entities

    def test_solution_role_privileges(self):
        parsed = _parse_url("#solutions/sln.23/roles/role.83/privileges")
        assert ParsedEntity("Application", "sln.23") in parsed.entities
        assert ParsedEntity("Role", "role.83") in parsed.entities

    def test_solution_workspaces(self):
        parsed = _parse_url("#solutions/Workspaces")
        assert parsed.page_type == "solutions"
        assert parsed.entities == []

    def test_solution_workspace(self):
        parsed = _parse_url("#solutions/sln.2/Workspaces/workspace.41")
        assert ParsedEntity("Application", "sln.2") in parsed.entities
        assert ParsedEntity("Workspace", "workspace.41") in parsed.entities

    def test_solution_templates_with_dataset_filter(self):
        parsed = _parse_url(
            "#solutions/sln.23/templates/showall/cmw.container.dataset.dsConfig/"
            "s%3Dcmw.container.dataset.instancesColumnDS%20Desc%20false"
            "%26sk%3D0%26t%3D50"
            "%26f%3D((cmw.container.dataset.solutionColumnDS%20eq%20sln.7)"
            "%20and%20(cmw.container.dataset.solutionColumnDS%20eq%20sln.23))"
        )
        assert ParsedEntity("Application", "sln.23") in parsed.entities
        # sln.* IDs from query filter params
        solution_ids = [
            e.entity_id for e in parsed.entities if e.entity_type == "Application"
        ]
        assert "sln.7" in solution_ids
        assert "sln.23" in solution_ids


class TestParseRecordTypeUrl:
    def test_record_type_administration(self):
        parsed = _parse_url("#RecordType/oa.3/Administration")
        assert parsed.page_type == "RecordType"
        assert ParsedEntity("Template", "oa.3") in parsed.entities

    def test_record_type_context(self):
        parsed = _parse_url("#RecordType/oa.3/Context")
        assert ParsedEntity("Template", "oa.3") in parsed.entities

    def test_record_type_forms(self):
        parsed = _parse_url("#RecordType/oa.3/Forms")
        assert ParsedEntity("Template", "oa.3") in parsed.entities

    def test_record_type_form(self):
        parsed = _parse_url("#RecordType/oa.3/Forms/form.80")
        assert ParsedEntity("Template", "oa.3") in parsed.entities
        assert ParsedEntity("Form", "form.80") in parsed.entities

    def test_record_type_operations(self):
        parsed = _parse_url("#RecordType/oa.3/Operations")
        assert ParsedEntity("Template", "oa.3") in parsed.entities

    def test_record_type_operation(self):
        parsed = _parse_url("#RecordType/oa.3/Operation/event.454")
        assert ParsedEntity("Template", "oa.3") in parsed.entities
        assert ParsedEntity("Button", "event.454") in parsed.entities

    def test_record_type_toolbar(self):
        parsed = _parse_url("#RecordType/oa.3/Toolbar/")
        assert ParsedEntity("Template", "oa.3") in parsed.entities

    def test_record_type_toolbar_settings(self):
        parsed = _parse_url("#RecordType/oa.3/Toolbar/Settings/tb.228")
        assert ParsedEntity("Template", "oa.3") in parsed.entities
        assert ParsedEntity("Toolbar", "tb.228") in parsed.entities

    def test_record_type_card(self):
        parsed = _parse_url("#RecordType/oa.3/Card/")
        assert ParsedEntity("Template", "oa.3") in parsed.entities

    def test_record_type_card_new_card(self):
        parsed = _parse_url("#RecordType/oa.3/Card/Settings/newCard")
        assert ParsedEntity("Template", "oa.3") in parsed.entities

    def test_record_type_card_settings(self):
        parsed = _parse_url("#RecordType/oa.3/Card/Settings/card.148")
        assert ParsedEntity("Template", "oa.3") in parsed.entities
        assert ParsedEntity("Form", "card.148") in parsed.entities

    def test_record_type_lists(self):
        parsed = _parse_url("#RecordType/oa.3/Lists/")
        assert ParsedEntity("Template", "oa.3") in parsed.entities

    def test_record_type_lists_lst(self):
        parsed = _parse_url("#RecordType/oa.3/Lists/lst.81")
        assert ParsedEntity("Template", "oa.3") in parsed.entities
        assert ParsedEntity("Dataset", "lst.81") in parsed.entities

    def test_record_type_csv(self):
        parsed = _parse_url("#RecordType/oa.3/csv")
        assert ParsedEntity("Template", "oa.3") in parsed.entities

    def test_record_type_security(self):
        parsed = _parse_url("#RecordType/oa.3/Security")
        assert ParsedEntity("Template", "oa.3") in parsed.entities

    def test_record_type_documents_templates(self):
        parsed = _parse_url("#RecordType/oa.3/DocumentsTemplates")
        assert ParsedEntity("Template", "oa.3") in parsed.entities


class TestParseRoleTemplateUrl:
    def test_role_template_administration(self):
        parsed = _parse_url("#RecordType/ra.23/Administration")
        assert parsed.page_type == "RecordType"
        assert ParsedEntity("Template", "ra.23") in parsed.entities

    def test_role_template_context(self):
        parsed = _parse_url("#RecordType/ra.23/Context")
        assert ParsedEntity("Template", "ra.23") in parsed.entities

    def test_role_template_toolbar(self):
        parsed = _parse_url("#RecordType/ra.23/Toolbar/")
        assert ParsedEntity("Template", "ra.23") in parsed.entities


class TestParseOrgStructureUrl:
    def test_orgstructure_administration(self):
        parsed = _parse_url("#RecordType/os.23/Administration")
        assert parsed.page_type == "RecordType"
        assert ParsedEntity("Template", "os.23") in parsed.entities

    def test_orgstructure_context(self):
        parsed = _parse_url("#RecordType/os.23/Context")
        assert ParsedEntity("Template", "os.23") in parsed.entities

    def test_orgstructure_toolbar(self):
        parsed = _parse_url("#RecordType/os.23/Toolbar/")
        assert ParsedEntity("Template", "os.23") in parsed.entities


class TestParseProcessTemplateUrl:
    def test_process_template_designer_diagram(self):
        parsed = _parse_url("#ProcessTemplate/pa.77/Designer/Revision/diagram.315")
        assert parsed.page_type == "ProcessTemplate"
        assert ParsedEntity("ProcessTemplate", "pa.77") in parsed.entities
        assert ParsedEntity("ProcessDiagram", "diagram.315") in parsed.entities

    def test_process_template_lists(self):
        parsed = _parse_url("#ProcessTemplate/pa.77/Lists/")
        assert ParsedEntity("ProcessTemplate", "pa.77") in parsed.entities

    def test_process_template_lists_lst(self):
        parsed = _parse_url("#ProcessTemplate/pa.77/Lists/lst.2741")
        assert ParsedEntity("ProcessTemplate", "pa.77") in parsed.entities
        assert ParsedEntity("Dataset", "lst.2741") in parsed.entities

    def test_process_template_toolbar(self):
        parsed = _parse_url("#ProcessTemplate/pa.77/Toolbar/tb.8215")
        assert ParsedEntity("ProcessTemplate", "pa.77") in parsed.entities
        assert ParsedEntity("Toolbar", "tb.8215") in parsed.entities

    def test_process_template_operation(self):
        parsed = _parse_url("#ProcessTemplate/pa.77/Operation/event.15193")
        assert ParsedEntity("ProcessTemplate", "pa.77") in parsed.entities
        assert ParsedEntity("Button", "event.15193") in parsed.entities


class TestParseDataViewUrl:
    def test_data_view_with_dataset_query(self):
        parsed = _parse_url(
            "#data/oa.26/lst.137/s%3Dds.5615%20Asc%20false%26sk%3D0%26t%3D50"
        )
        assert ParsedEntity("Template", "oa.26") in parsed.entities
        assert ParsedEntity("Dataset", "lst.137") in parsed.entities
        # ds.* from query params
        dataset_ids = [
            e.entity_id for e in parsed.entities if e.entity_type == "Dataset"
        ]
        assert "ds.5615" in dataset_ids

    def test_data_view_simple(self):
        parsed = _parse_url("#data/oa.163/lst.1097/sk%3D0%26t%3D50")
        assert ParsedEntity("Template", "oa.163") in parsed.entities
        assert ParsedEntity("Dataset", "lst.1097") in parsed.entities


class TestParseFormViewUrl:
    def test_form_view_with_record(self):
        parsed = _parse_url("#form/oa.3/form.80/55")
        assert ParsedEntity("Template", "oa.3") in parsed.entities
        assert ParsedEntity("Form", "form.80") in parsed.entities
        assert ParsedEntity("Record", "55") in parsed.entities


class TestParseAppViewUrl:
    def test_app_list(self):
        parsed = _parse_url("#app/FacilityManagement/list/MaintenancePlans")
        assert parsed.page_type == "app"
        assert ParsedEntity("App", "FacilityManagement") in parsed.entities
        assert ParsedEntity("Template", "MaintenancePlans") in parsed.entities

    def test_app_view_with_record(self):
        parsed = _parse_url("#app/FacilityManagement/view/MaintenancePlans/15199")
        assert ParsedEntity("App", "FacilityManagement") in parsed.entities
        assert ParsedEntity("Template", "MaintenancePlans") in parsed.entities
        assert ParsedEntity("Record", "15199") in parsed.entities


class TestParseSettingsUrl:
    def test_settings_groups(self):
        parsed = _parse_url("#Settings/Groups")
        assert parsed.page_type == "Settings"
        assert parsed.entities == []

    def test_settings_support_channels(self):
        parsed = _parse_url("#Settings/support/channels")
        assert parsed.page_type == "Settings"
        assert parsed.entities == []

    def test_settings_support_routes(self):
        parsed = _parse_url("#Settings/support/routes")
        assert parsed.page_type == "Settings"
        assert parsed.entities == []

    def test_settings_global_security(self):
        parsed = _parse_url("#Settings/globalSecurity")
        assert parsed.page_type == "Settings"
        assert parsed.entities == []

    def test_settings_global_security_role(self):
        parsed = _parse_url("#Settings/globalSecurity/role.9/privileges")
        assert parsed.page_type == "Settings"
        assert ParsedEntity("Role", "role.9") in parsed.entities

    def test_settings_global_security_role_no_privileges(self):
        parsed = _parse_url("#Settings/globalSecurity/role.9")
        assert ParsedEntity("Role", "role.9") in parsed.entities

    def test_settings_global_security_role_by_name(self):
        parsed = _parse_url(
            "#Settings/globalSecurity/cmw.role.SysInfrastructureAdminRole/privileges"
        )
        # System name role reference — not an internal ID, skip
        assert parsed.page_type == "Settings"


class TestParseResolverUrl:
    def test_resolver_template(self):
        parsed = _parse_url("#Resolver/oa.193")
        assert parsed.page_type == "Resolver"
        assert ParsedEntity("Template", "oa.193") in parsed.entities

    def test_resolver_raw_id(self):
        parsed = _parse_url("#Resolver/event.15199")
        assert parsed.page_type == "Resolver"
        assert ParsedEntity("Button", "event.15199") in parsed.entities


class TestParseFullUrl:
    def test_full_url_with_base(self):
        parsed = _parse_url(
            "https://bububu.bau.cbap.ru/#RecordType/oa.3/Operation/event.454"
        )
        assert parsed.page_type == "RecordType"
        assert ParsedEntity("Template", "oa.3") in parsed.entities
        assert ParsedEntity("Button", "event.454") in parsed.entities

    def test_full_url_desktop(self):
        parsed = _parse_url("https://bububu.bau.cbap.ru/#desktop/")
        assert parsed.page_type == "desktop"
        assert parsed.entities == []

    def test_full_url_data_view(self):
        parsed = _parse_url(
            "https://bububu.bau.cbap.ru/#data/oa.3/lst.81/sk%3D0%26t%3D50"
        )
        assert ParsedEntity("Template", "oa.3") in parsed.entities
        assert ParsedEntity("Dataset", "lst.81") in parsed.entities


class TestParseRawId:
    def test_raw_template_id(self):
        parsed = _parse_url("oa.193")
        assert ParsedEntity("Template", "oa.193") in parsed.entities

    def test_raw_process_template_id(self):
        parsed = _parse_url("pa.77")
        assert ParsedEntity("ProcessTemplate", "pa.77") in parsed.entities

    def test_raw_role_template_id(self):
        parsed = _parse_url("ra.23")
        assert ParsedEntity("Template", "ra.23") in parsed.entities

    def test_raw_orgstructure_id(self):
        parsed = _parse_url("os.23")
        assert ParsedEntity("Template", "os.23") in parsed.entities

    def test_raw_solution_id(self):
        parsed = _parse_url("sln.23")
        assert ParsedEntity("Application", "sln.23") in parsed.entities

    def test_raw_button_id(self):
        parsed = _parse_url("event.15199")
        assert ParsedEntity("Button", "event.15199") in parsed.entities

    def test_raw_form_id(self):
        parsed = _parse_url("form.80")
        assert ParsedEntity("Form", "form.80") in parsed.entities

    def test_raw_toolbar_id(self):
        parsed = _parse_url("tb.228")
        assert ParsedEntity("Toolbar", "tb.228") in parsed.entities

    def test_raw_list_id(self):
        parsed = _parse_url("lst.81")
        assert ParsedEntity("Dataset", "lst.81") in parsed.entities

    def test_raw_diagram_id(self):
        parsed = _parse_url("diagram.315")
        assert ParsedEntity("ProcessDiagram", "diagram.315") in parsed.entities

    def test_raw_role_id(self):
        parsed = _parse_url("role.83")
        assert ParsedEntity("Role", "role.83") in parsed.entities

    def test_raw_workspace_id(self):
        parsed = _parse_url("workspace.41")
        assert ParsedEntity("Workspace", "workspace.41") in parsed.entities

    def test_raw_record_id(self):
        parsed = _parse_url("15199")
        assert ParsedEntity("Record", "15199") in parsed.entities


class TestParseEdgeCases:
    def test_empty_string(self):
        parsed = _parse_url("")
        assert parsed.page_type == "unknown"
        assert parsed.entities == []

    def test_whitespace_only(self):
        parsed = _parse_url("   ")
        assert parsed.page_type == "unknown"
        assert parsed.entities == []

    def test_url_without_hash(self):
        parsed = _parse_url("https://bububu.bau.cbap.ru/")
        assert parsed.page_type == "unknown"
        assert parsed.entities == []

    def test_url_with_hash_only(self):
        parsed = _parse_url("#")
        assert parsed.page_type == "unknown"
        assert parsed.entities == []

    def test_multiple_entities_in_one_url(self):
        parsed = _parse_url("#RecordType/oa.3/Lists/lst.81")
        assert len(parsed.entities) == 2
        assert ParsedEntity("Template", "oa.3") in parsed.entities
        assert ParsedEntity("Dataset", "lst.81") in parsed.entities

    def test_complex_data_view_multiple_datasets(self):
        parsed = _parse_url(
            "#data/oa.3/lst.81/s%3Dds.5615%20Asc%20false"
            "%26f%3D(cmw.container.dataset.solutionColumnDS%20eq%20sln.23)"
        )
        assert ParsedEntity("Template", "oa.3") in parsed.entities
        assert ParsedEntity("Dataset", "lst.81") in parsed.entities
        assert ParsedEntity("Application", "sln.23") in parsed.entities
