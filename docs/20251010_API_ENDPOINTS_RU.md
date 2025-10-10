# CMW Platform Agent API - Streaming Endpoint (ask_stream)

**Дата:** 2025-10-10  
**Версия:** 1.0  
**Статус:** Готово к продакшену

## Обзор

CMW Platform Agent предоставляет REST API endpoint `/ask_stream` для потокового получения ответов ассистента в реальном времени. Этот endpoint идеально подходит для создания интерактивных чат-интерфейсов.

## Базовый URL

```
http://10.9.7.7:7860/
```

или

```
https://huggingface.co/spaces/arterm-sedov/cmw-agent
```

## Аутентификация

Аутентификация не требуется. Все запросы обрабатываются с изоляцией сессий.

## Endpoint `/ask_stream`

### Описание

Возвращает инкрементальные части ответа ассистента по мере их генерации.

**Метод:** `POST`  
**Путь:** `/call/ask_stream`  
**Content-Type:** `application/json`

### Формат запроса

```json
{
  "data": ["Ваш вопрос здесь", "username", "password", "base_url"],
  "session_hash": "опциональный-id-сессии"
}
```

### Параметры

- `data[0]` (string, обязательный): Вопрос пользователя
- `data[1]` (string, опциональный): Имя пользователя для аутентификации в Comindware Platform
- `data[2]` (string, опциональный): Пароль для аутентификации в Comindware Platform  
- `data[3]` (string, опциональный): Базовый URL платформы Comindware (например, "https://your-platform.com")
- `session_hash` (string, опциональный): Идентификатор сессии для многоходовых диалогов

### Формат ответа

**Успешный ответ:**
```json
{
  "event_id": "уникальный-id-события"
}
```

**Потоковые результаты (через GET):**
```
event: generating
data: ["Привет"]

event: generating
data: ["Привет, к"]

event: generating
data: ["Привет, как"]

event: generating
data: ["Привет, как дела?"]

event: complete
data: ["Привет, как дела?"]
```

## Примеры использования cURL

### Базовый пример

```bash
# Отправка вопроса
curl -X POST http://localhost:7860/call/ask_stream \
  -H "Content-Type: application/json" \
  -d '{"data": ["Привет, как дела?"]}'

# Получение результата (замените EVENT_ID на реальный ID)
curl -N http://localhost:7860/call/ask_stream/EVENT_ID
```

### С аутентификацией

```bash
# Отправка вопроса с учетными данными
curl -X POST http://localhost:7860/call/ask_stream \
  -H "Content-Type: application/json" \
  -d '{"data": ["Покажи мои задачи", "myuser", "mypass", "https://my-platform.com"]}'

# Получение результата
curl -N http://localhost:7860/call/ask_stream/EVENT_ID
```

### С сессией

```bash
# Первое сообщение в сессии
curl -X POST http://localhost:7860/call/ask_stream \
  -H "Content-Type: application/json" \
  -d '{"data": ["Что такое 2+2?"], "session_hash": "my-session-123"}'

# Следующее сообщение в той же сессии
curl -X POST http://localhost:7860/call/ask_stream \
  -H "Content-Type: application/json" \
  -d '{"data": ["А что такое 3+3?"], "session_hash": "my-session-123"}'
```

### Комбинированная команда (отправка + получение)

```bash
# Автоматическое получение EVENT_ID и запрос результата
curl -X POST http://localhost:7860/call/ask_stream \
  -H "Content-Type: application/json" \
  -d '{"data": ["Стрими это пожалуйста"]}' \
  | awk -F'"' '{ print $4}' \
  | read EVENT_ID; curl -N http://localhost:7860/call/ask_stream/$EVENT_ID
```

## Управление сессиями

### Многоходовые диалоги

Оба endpoint поддерживают сохранение состояния сессии через параметр `session_hash`:

- **С session_hash:** Сообщения являются частью одного контекста разговора
- **Без session_hash:** Каждый запрос обрабатывается как новый разговор
- **Изоляция сессий:** Разные session_hash поддерживают отдельные истории диалогов

## Обработка ошибок

### Ошибки в потоке

Для потоковых endpoint ошибки возвращаются как события:

```
event: error
data: ["Сообщение об ошибке здесь"]
```

### Типичные ошибки

**Ошибка соединения:**
```json
{
  "error": "Connection refused"
}
```

**Ошибка таймаута:**
```json
{
  "error": "Request timeout"
}
```

## Управление сессиями

- **Всегда используйте session_hash** для многоходовых диалогов
- **Генерируйте уникальные session ID** для разных пользователей/диалогов
- **Переиспользуйте session_hash** в рамках одного потока диалога

## Тестирование

### Ручное тестирование

1. **Запустите агента:**
   ```bash
   python -m agent_ng.app_ng_modular
   ```

2. **Тестируйте потоковый endpoint:**
   ```bash
   curl -X POST http://localhost:7860/call/ask_stream \
     -H "Content-Type: application/json" \
     -d '{"data": ["Стрими это"]}'
   ```

## Производительность

- **Время первого чанка:** 1-2 секунды
- **Инкрементальные обновления:** По мере генерации
- **Одновременные запросы:** Ограничены системой очередей Gradio (по умолчанию: 1)
- **Таймаут очереди:** 30 секунд на запрос

## Устранение неполадок

### Частые проблемы

1. **"Application is initializing..."**
   - Дождитесь полной инициализации агента
   - Проверьте логи на ошибки инициализации

2. **Connection refused**
   - Убедитесь, что агент запущен на правильном порту
   - Проверьте настройки файрвола

3. **Пустые ответы**
   - Убедитесь, что вопрос не пустой
   - Проверьте конфигурацию агента

