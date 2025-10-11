import json
from tools.templates_tools.tool_list_attributes import list_attributes


if __name__ == "__main__":
    params = {
        "application_system_name": "Управлениеавтопарком",
        "template_system_name": "Заявкинаавтомобили",
    }
    res = list_attributes.invoke(params)
    print(json.dumps(res, ensure_ascii=False, indent=2))


