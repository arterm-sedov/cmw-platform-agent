from ..tool_utils import *
import requests
from typing import Dict, Any, List

# Эндпоинты (имена как в вашем первоначальном коде)
ENDPOINT0 = "api/public/system/Process/ProcessAppService/List"      # ← Шаг 1: получить все системы
ENDPOINT1 = "api/public/system/Process/ProcessAppService/Get"         # ← Шаг 2: получить ActiveDiagramId по system_id
ENDPOINT2 = "api/public/system/Base/OntologyService/GetAxioms"        # ← Шаг 3: получить элементы по diagram_id
ENDPOINT3 = "api/public/system/Base/TriggerService/Get"               # ← Для триггеров (проверьте путь!)

def get_element_type(rdf_types: List[str]) -> str:
    for t in rdf_types:
        if t.startswith("cmw.process.diagram.activity."):
            short = t.split(".")[-1]
            if short == "StartEvent":
                return "start_event"
            elif short == "EndEvent":
                return "end_event"
            elif short == "Gate":
                return "gateway"
            elif short == "ExecutionFlow":
                return "sequence_flow"
            elif short == "SubProcess":
                return "sub_process"
            elif "Task" in short or short == "SomeActivity":
                return "task"
            else:
                return "activity"
    return "unknown"


def fetch_trigger_name(tid: str) -> str:
    try:
        cfg = requests_._load_server_config()
        base_url = cfg.base_url.rstrip("/")
        headers = requests_._basic_headers()
        url = f"{base_url}/{ENDPOINT3}"

        resp = requests.post(url, headers=headers, json=tid, timeout=cfg.timeout)
        resp.raise_for_status()
        data = resp.json()
        names = data.get("cmw.trigger.name", [])
        return names[0] if names else ""
    except (requests.RequestException, ValueError, KeyError, IndexError) as e:
        print(f"⚠️ Ошибка при получении триггера {tid}: {e}")
        return ""


@tool("audit_process_schema", return_direct=False)
def audit_process_schema(process_template_system_name: str) -> List[Dict[str, Any]]:
    """
    Аудит схемы процесса по system name шаблона.
    
    Последовательность:
      1. endpoint0 (без тела) → список систем → найти id по alias
      2. endpoint1 (тело = system_id) → {"ActiveDiagramId": "..."}
      3. endpoint2 (тело = ActiveDiagramId) → получить элементы схемы
    """
    cfg = requests_._load_server_config()
    base_url = cfg.base_url.rstrip("/")
    headers = requests_._basic_headers()

    endpoint0 = f"{base_url}/{ENDPOINT0}"
    endpoint1 = f"{base_url}/{ENDPOINT1}"
    endpoint2 = f"{base_url}/{ENDPOINT2}"

    # --- Шаг 1: Получить все системы ---
    try:
        resp0 = requests.post(endpoint0, headers=headers, timeout=cfg.timeout)
        if resp0.status_code != 200:
            raise ValueError(f"Ошибка запроса к {endpoint0}: HTTP {resp0.status_code}, тело: {resp0.text}")

        systems = resp0.json()
        if not isinstance(systems, list):
            raise ValueError(f"Ожидался список от {endpoint0}, получено: {type(systems)}")

        target_system = next((s for s in systems if s.get("alias") == process_template_system_name), None)
        if not target_system:
            raise ValueError(f"Система с alias '{process_template_system_name}' не найдена.")

        system_id = target_system.get("id")
        if not system_id:
            raise ValueError(f"У системы отсутствует 'id'.")

    except Exception as e:
        raise RuntimeError(f"Шаг 1 (получение system_id) завершился ошибкой: {e}")

    # --- Шаг 2: Получить ActiveDiagramId ---
    try:
        resp1 = requests.post(endpoint1, headers=headers, data=f"{system_id}", timeout=cfg.timeout)
        if resp1.status_code != 200:
            raise ValueError(f"Ошибка запроса к {endpoint1}: HTTP {resp1.status_code}, тело: {resp1.text}")

        data1 = resp1.json()
        active_diagram_id = data1.get("activeDiagramId")
        if not active_diagram_id:
            raise ValueError(f"Ответ от {endpoint1} не содержит 'ActiveDiagramId'. Получено: {data1}")

    except Exception as e:
        raise RuntimeError(f"Шаг 2 (получение ActiveDiagramId) завершился ошибкой: {e}")

    # --- Шаг 3: Получить все элементы схемы по diagram_id ---
    try:
        resp2 = requests.post(endpoint2, headers=headers, data=active_diagram_id, timeout=cfg.timeout)
        if resp2.status_code != 200:
            raise ValueError(f"Ошибка запроса к {endpoint2}: HTTP {resp2.status_code}, тело: {resp2.text}")

        data2 = resp2.json()
        all_ids = data2.get("cmw.process.diagram.activities", [])
    except Exception as e:
        raise RuntimeError(f"Шаг 3 (получение списка элементов) завершился ошибкой: {e}")

    # --- Шаг 4: Подробно загрузить каждый элемент ---
    elements = []
    for elem_id in all_ids:
        try:
            resp_elem = requests.post(endpoint2, headers=headers, data=elem_id, timeout=cfg.timeout)
            if resp_elem.status_code != 200:
                print(f"⚠️ Пропущен элемент {elem_id}: HTTP {resp_elem.status_code}")
                continue
            data = resp_elem.json()
        except (requests.RequestException, ValueError) as e:
            print(f"⚠️ Ошибка при получении данных элемента {elem_id}: {e}")
            continue

        rdf_types = data.get("http://www.w3.org/1999/02/22-rdf-syntax-ns#type", [])
        elem_type = get_element_type(rdf_types)
        name = data.get("cmw.process.diagram.activity.title", [""])[0]

        element = {
            "id": elem_id,
            "type": elem_type,
            "name": name
        }

        if elem_type != "sequence_flow":
            trigger_ids = data.get("cmw.process.diagram.activity.exitTrigger", [])
            element["on_exit_trigger_names"] = [
                fetch_trigger_name(tid) for tid in trigger_ids
            ]

        if elem_type == "gateway":
            kinds = data.get("cmw.process.diagram.activity.gate.kind", [])
            if "cmw.process.diagram.activity.gate.kind.Exclusive" in kinds:
                element["gateway_type"] = "exclusive"
            elif "cmw.process.diagram.activity.gate.kind.Parallel" in kinds:
                element["gateway_type"] = "parallel"
            else:
                element["gateway_type"] = "unknown"

        if elem_type == "sequence_flow":
            element["source"] = data.get("cmw.process.diagram.activity.flow.source", [None])[0]
            element["target"] = data.get("cmw.process.diagram.activity.flow.target", [None])[0]

        elements.append(element)

    return elements


if __name__ == "__main__":
    try:
        results = audit_process_schema("TestProcess")
        print("✅ Успешно получены элементы:")
        for r in results:
            print(r)
    except Exception as e:
        print(f"❌ Ошибка: {e}")