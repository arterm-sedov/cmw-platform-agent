import json
from tools.templates_tools.tool_list_records import list_template_records


def run_case(application_system_name: str, template_system_name: str, attributes, filters):
    params = {
        "application_system_name": application_system_name,
        "template_system_name": template_system_name,
        "attributes": attributes,
        "filters": filters,
        "limit": 10,
        "offset": 0,
        "sort_by": None,
        "sort_desc": False,
    }
    res = list_template_records.invoke(params)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Target case from chat
    print("\n=== Case 1: attributes=[], filters={} ===")
    run_case("Управлениеавтопарком", "Заявкинаавтомобили", attributes=[], filters={})

    print("\n=== Case 2: attributes=None, filters=None ===")
    run_case("Управлениеавтопарком", "Заявкинаавтомобили", attributes=None, filters=None)


