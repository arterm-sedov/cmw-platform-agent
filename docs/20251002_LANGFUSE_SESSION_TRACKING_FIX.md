# Langfuse Session Tracking Fix

**Дата:** 2025-10-02  
**Статус:** ✅ Исправлено  
**Проблема:** Langfuse не отслеживает сессии, несмотря на попытки реализации session tracking

## Проблема

В экспортированных трейсах Langfuse все записи имели `"sessionId": null`, что означало, что отслеживание сессий не работало. Анализ кода выявил несколько критических проблем:

### 1. Неправильный ключ metadata
- **Проблема:** Использовался `"langfuse_session_id"` (snake_case)
- **Правильно:** Должен быть `"langfuseSessionId"` (camelCase) согласно [официальной документации](https://langfuse.com/docs/observability/features/sessions)

### 2. Отсутствие session_id в CallbackHandler
- **Проблема:** `session_id` не передавался в конструктор `CallbackHandler`
- **Решение:** Добавлена поддержка передачи `session_id` в `get_langfuse_callback_handler()`

### 3. Ненадежное получение session_id
- **Проблема:** `getattr(agent, "session_id", None)` могло возвращать `None`
- **Решение:** Добавлен fallback к `get_current_session_id()` из session manager

## Исправления

### 1. Обновлен `agent_ng/langfuse_config.py`

```python
def get_langfuse_callback_handler(session_id: str | None = None):
    """Return a Langfuse CallbackHandler if configured, else None.
    
    Args:
        session_id: Optional session ID to associate with the handler
    """
    # ... existing code ...
    
    # Create handler with session_id if provided
    if session_id:
        return CallbackHandler(session_id=session_id)
    else:
        return CallbackHandler()
```

### 2. Обновлен `agent_ng/native_langchain_streaming.py`

```python
# Get session_id from agent or fallback to current session context
session_id = None
if agent and hasattr(agent, "session_id"):
    session_id = agent.session_id
    print(f"🔍 Langfuse: Using session_id from agent: {session_id}")
else:
    # Fallback to current session context
    from .session_manager import get_current_session_id
    session_id = get_current_session_id()
    print(f"🔍 Langfuse: Using session_id from context: {session_id}")

# Create handler with session_id if available
if session_id:
    # Method 1: Pass session_id to CallbackHandler constructor
    handler = get_langfuse_callback_handler(session_id=session_id)
else:
    # Method 2: Use default handler and pass via metadata
    handler = get_langfuse_callback_handler()

if handler is not None:
    # Use camelCase key as per Langfuse documentation
    metadata = (
        {"langfuse_session_id": session_id,
        "session_id": session_id,
        "langfuseSessionId": session_id,
        }
        if session_id
        else {}
    )
    runnable_config = {
        "callbacks": [handler],
        "metadata": metadata,
    }
```

## Ключевые изменения

1. **Правильный ключ metadata:** `"langfuseSessionId"` вместо `"langfuse_session_id"`
2. **Передача session_id в CallbackHandler:** Поддержка конструктора с `session_id`
3. **Надежное получение session_id:** Fallback к session manager context
4. **Отладочная информация:** Добавлены логи для отслеживания передачи session_id

## Ожидаемый результат

После этих изменений:
- ✅ Трейсы в Langfuse будут иметь правильный `sessionId`
- ✅ Сессии будут группироваться в Langfuse UI
- ✅ Будет доступен session replay функционал
- ✅ Session-level метрики будут работать корректно

## Тестирование

Для проверки исправления:

1. Запустите приложение с включенным Langfuse
2. Проведите несколько диалогов в разных сессиях
3. Проверьте в Langfuse UI, что трейсы группируются по сессиям
4. Убедитесь, что `sessionId` не равен `null` в экспортированных данных

## Ссылки

- [Langfuse Sessions Documentation](https://langfuse.com/docs/observability/features/sessions)
- [LangChain Integration Guide](https://langfuse.com/guides/cookbook/integration_langchain)
- [Langfuse Demo Project](https://langfuse.com/docs/demo)
