from ..tool_utils import *
import requests
from typing import Dict, Any, List

# Эндпоинты (имена как в вашем первоначальном коде)
ENDPOINT0 = "api/public/system/Process/ProcessAppService/List"      # ← Шаг 1: получить все системы
ENDPOINT1 = "api/public/system/Process/ProcessAppService/Get"         # ← Шаг 2: получить ActiveDiagramId по system_id
ENDPOINT2 = "api/public/system/Base/OntologyService/GetAxioms"        # ← Шаг 3: получить элементы по diagram_id

def get_element_subtype(data: Dict[str, Any], element_type: str) -> str:
    """
    Возвращает подтип элемента в человекочитаемом виде
    """
    if element_type == "gateway":
        gate_kinds = data.get("cmw.process.diagram.activity.gate.kind", [])
        if "cmw.process.diagram.activity.gate.kind.Exclusive" in gate_kinds:
            return "И"
        elif "cmw.process.diagram.activity.gate.kind.Parallel" in gate_kinds:
            return "ИЛИ"
        else:
            return "неизвестный"
    
    elif element_type == "sequence_flow":
        # Определяем подтипы потоков
        flow_types = data.get("http://www.w3.org/1999/02/22-rdf-syntax-ns#type", [])
        if "cmw.process.diagram.activity.ExecutionFlow" in flow_types:
            return "execution"
        elif "cmw.process.diagram.activity.SequenceFlow" in flow_types:
            return "sequence"
        else:
            return "неизвестный"
    
    elif element_type in ["start_event", "end_event", "intermediate_event"]:
        event_kinds = data.get("cmw.process.diagram.activity.event.kind", [])
        if not event_kinds:
            return "Обычное"
        
        event_kind = event_kinds[0] if event_kinds else ""
        
        # Определяем подтипы событий
        if "None" in event_kind:
            return "Обычное"
        elif "Timer" in event_kind:
            return "Таймер"
        elif "CatchingMessage" in event_kind:
            return "Получение сообщения"
        elif "ThrowingMessage" in event_kind:
            return "Отправка сообщения"
        elif "Terminate" in event_kind:
            return "Терминатор"
        else:
            return "неизвестный"
    
    elif element_type == "task":
        task_kinds = data.get("cmw.process.diagram.activity.task.kind", [])
        if "User" in task_kinds:
            return "Пользовательская задача"
        elif "Script" in task_kinds:
            return "Выполнение скрипта"
        elif "Service" in task_kinds:
            return "Вызов сервиса"
        elif "SubProcess" in task_kinds:
            return "Вызов подпроцесса"
        else:
            return "неизвестный"
    
    return ""

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
            elif short == "SequenceFlow":
                return "sequence_flow"
            elif short == "SubProcess" or short == "EmbeddedProcess":
                return "sub_process"
            elif short == "Task":
                return "task"
            elif short == "Pool":
                return "pool"
            elif short == "Lane":
                return "lane"
            elif short == "IntermediateEvent":
                return "intermediate_event"
            elif short == "Event":
                return "intermediate_event"  # Для промежуточных событий
            else:
                return "activity"
    return "unknown"

def get_event_definitions(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Извлекает определения событий (таймер, сообщение и т.д.)
    """
    definitions = {}
    
    # Таймер
    timer_def = data.get("cmw.process.diagram.activity.event.timerDefinition", [])
    if timer_def:
        definitions["timer_definition"] = timer_def[0]
    
    # Сообщение
    message_def = data.get("cmw.process.diagram.activity.event.messageDefinition", [])
    if message_def:
        definitions["message_definition"] = message_def[0]
    
    # Форма (для задач)
    form_def = data.get("cmw.process.diagram.activity.task.form", [])
    if form_def:
        definitions["form_definition"] = form_def[0]
    
    # Скрипт (для задач)
    script_def = data.get("cmw.process.diagram.activity.task.scriptDefinition", [])
    if script_def:
        definitions["script_definition"] = script_def[0]
    
    # Пользовательская задача
    user_def = data.get("cmw.process.diagram.activity.task.userDefinition", [])
    if user_def:
        definitions["user_definition"] = user_def[0]
    
    # Поток последовательности
    flow_def = data.get("cmw.process.diagram.activity.flow.sequenceFlowDefinition", [])
    if flow_def:
        definitions["flow_definition"] = flow_def[0]
    
    return definitions

def fetch_trigger_name(tid: str) -> str:
    try:
        cfg = requests_._load_server_config()
        base_url = cfg.base_url.rstrip("/")
        headers = requests_._basic_headers()
        url = f"{base_url}/{ENDPOINT2}"

        resp = requests.post(url, headers=headers, json=tid, timeout=cfg.timeout)
        resp.raise_for_status()
        data = resp.json()
        names = data.get("cmw.trigger.name", [])
        return names[0] if names else ""
    except (requests.RequestException, ValueError, KeyError, IndexError) as e:
        print(f"⚠️ Ошибка при получении сценария {tid}: {e}")
        return ""


@tool("get_process_schema", return_direct=False)
def get_process_schema(process_template_system_name: str) -> List[Dict[str, Any]]:
    """
    Получает BPMN-диаграмму процесса по имени шаблона процесса. 
    Возвращает структурированный список элементов BPMN диаграммы с их свойствами, 
    включая пулы, дорожки, шлюзы, события, задачи и потоки управления.
    
    Параметры:
    - process_template_system_name: системное имя шаблона процесса для аудита
    
    Возвращаемые данные:
    - id: идентификатор элемента
    - type: тип элемента (start_event, end_event, gateway, task, sequence_flow, pool, lane и др.)
    - subtype: подтип элемента (например, "И", "ИЛИ" для шлюзов, "execution" для flow и т.д.)
    - name: название элемента
    - lid: локальный идентификатор элемента
    - systemName: системное имя элемента
    - owner: владелец/системное имя
    - x, y: координаты элемента
    - width, height: размеры элемента
    - source, target: для потоков управления - источник и цель
    - pointsX, pointsY: для потоков управления - массивы координат точек
    - definitions: словарь с определениями (таймер, сообщение, форма и т.д.)
    - is_interrupting: флаг прерывания (для событий)
    - lane_index: индекс дорожки (для дорожек)
    - on_exit_trigger_names: имена сценариев на выходе из элементов (для элементов кроме потоков)
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
            "name": name,
            "lid": data.get("cmw.process.diagram.activity.lid", [None])[0],
            "systemName": data.get("cmw.process.diagram.activity.systemName", [None])[0],
            "owner": data.get("cmw.process.diagram.activity.owner", [None])[0],
            "x": data.get("cmw.process.diagram.activity.x", [None])[0],
            "y": data.get("cmw.process.diagram.activity.y", [None])[0],
            "width": data.get("cmw.process.diagram.activity.width", [None])[0],
            "height": data.get("cmw.process.diagram.activity.height", [None])[0]
        }

        # Определяем подтип элемента
        element_subtype = get_element_subtype(data, elem_type)
        if element_subtype:
            element["subtype"] = element_subtype

        # Добавляем определения (таймеры, сообщения и т.д.)
        definitions = get_event_definitions(data)
        if definitions:
            element["definitions"] = definitions

        # Для событий добавляем флаг прерывания
        if elem_type in ["start_event", "end_event", "intermediate_event"]:
            is_interrupting = data.get("cmw.process.diagram.activity.event.isInterrrupting", [None])[0]
            if is_interrupting is not None:
                element["is_interrupting"] = is_interrupting

        # Для дорожек добавляем индекс
        if elem_type == "lane":
            lane_index = data.get("cmw.process.diagram.activity.lane.index", [None])[0]
            if lane_index is not None:
                element["lane_index"] = lane_index

        if elem_type != "sequence_flow":
            trigger_ids = data.get("cmw.process.diagram.activity.exitTrigger", [])
            element["on_exit_trigger_names"] = [
                fetch_trigger_name(tid) for tid in trigger_ids
            ]

        if elem_type == "sequence_flow":
            element["source"] = data.get("cmw.process.diagram.activity.flow.source", [None])[0]
            element["target"] = data.get("cmw.process.diagram.activity.flow.target", [None])[0]
            element["pointsX"] = data.get("cmw.process.diagram.activity.flow.pointsX", [None])[0]
            element["pointsY"] = data.get("cmw.process.diagram.activity.flow.pointsY", [None])[0]

        elements.append(element)

    return elements


if __name__ == "__main__":
    try:
        results = get_process_schema("TestProcess")
        print("✅ Успешно получены элементы:")
        for r in results:
            print(r)
    except Exception as e:
        print(f"❌ Ошибка: {e}")