"""
Internationalization (i18n) Translations for CMW Platform Agent
================================================================

Provides Russian and English translations for all UI text in the Gradio application.
Uses Gradio's built-in I18n class for seamless localization support.

Based on Gradio's internationalization documentation:
https://www.gradio.app/guides/internationalization
"""

import gradio as gr
from typing import Dict, Any

# Russian translations for all UI text
RUSSIAN_TRANSLATIONS = {
    # App title and header
    "app_title": "Ассистент аналитика Comindware",
    "hero_title": "Ассистент аналитика",
    
    # Tab labels
    "tab_chat": "💬 Чат",
    "tab_logs": "📜 Журналы", 
    "tab_stats": "📊 Статистика",
    
    # Chat tab content
    "welcome_title": "💬 Добро пожаловать!",
    "welcome_description": """
    Ассистен аналитика Comindware фокусируется на операциях с сущностями **Comindware Platform** (приложения, шаблоны, атрибуты) и использует детерминированные инструменты для выполнения точных изменений.

    - **Операции с Comindware Platform в приоритете**: помогает выполнять операции с помощью инструментов для изменения сущностей (например, создание/редактирование атрибутов)
    - **Оркестрация нескольких моделей**: позволяет использовать различных поставщиков LLM
    - **Компактный структурированный вывод**: Намерение → План → Проверка → Выполнение → Результат
    """,
    
    "try_asking_title": "❓ Варианты запросов:",
    "try_asking_examples": """
    - Перечисли все приложения в платформе в удобном списке
    - Покажи все шаблоны записей в приложении "ERP". Отформатируй красиво используя Markdown.
    - Выдай список всех атрибутов шаблона "Контрагенты", приложение "ERP"
    - Создать текстовый атрибут "Комментарий", приложение "HR", шаблон "Кандидаты"
    - Создай текстовый атрибут "ID клиента", приложение "ERP", шаблон "Контрагенты", особая маска ввода: ([0-9]{10}|[0-9]{12})
    - Для атрибута "Контактный телефон" в приложении "CRM", шаблон "Лиды", смени формат отображения на российский телефон
    - Получи атрибут: системное имя "Комментарий", приложение "HR", шаблон "Кандидаты"
    - Архивируй/разархивируй атрибут, системное имя "Комментарий", приложение "HR", шаблон "Кандидаты"
    """,
    
    # Quick actions
    "quick_actions_title": "⚡ Заготовки",
    "quick_list_apps": "🔎 Список всех приложений",
    "quick_create_attr": "🧩 Создать текстовый атрибут",
    "quick_edit_mask": "🛠️ Редактировать маску телефона",
    "quick_math": "🧮 15 * 23 + 7 = ?",
    "quick_code": "💻 Функция проверки простых чисел на Python",
    "quick_explain": "💭 Объяснить ML кратко",
    
    # Chat interface
    "chat_label": "Чат с агентом",
    "message_label": "Ваше сообщение",
    "message_placeholder": "Введите ваше сообщение...",
    "send_button": "Отправить",
    "clear_button": "Очистить чат",
    
    # Status section
    "status_title": "🤖 Статус",
    "status_initializing": "🟡 Инициализация...",
    "status_ready": "✅ Готов",
    "progress_title": "📊 Прогресс",
    "progress_ready": "Готов обработать ваш запрос...",
    
    # Logs tab
    "logs_title": "Журналы инициализации",
    "logs_initializing": "🟡 Идёт инициализация...",
    "refresh_logs_button": "🔄 Обновить журналы",
    "clear_logs_button": "🗑️ Очистить журналы",
    "logs_cleared": "Журналы очищены.",
    "logs_not_available": "Журналы недоступны — основное приложение не подключено",
    
    # Stats tab
    "stats_title": "Статистика агента",
    "stats_loading": "Загрузка статистики...",
    "refresh_stats_button": "🔄 Обновить статистику",
    "agent_not_available": "Агент недоступен",
    "error_loading_stats": "Ошибка загрузки статистики",
    
    # Status messages
    "agent_ready": "✅ **Агент готов**",
    "agent_initializing": "🟡 **Инициализация агента...**",
    "agent_not_ready": "❌ **Агент не готов. Пожалуйста, дождитесь завершения инициализации.**",
    
    # Error messages
    "error_processing": "❌ **Ошибка обработки сообщения: {error}**",
    "error_streaming": "❌ **Ошибка потоковой передачи сообщения: {error}**",
    "error_agent_timeout": "❌ **Таймаут инициализации агента**",
    "error_initialization_failed": "❌ **Ошибка инициализации: {error}**",
    
    # Token and execution info
    "prompt_tokens": "**Токены запроса:** {tokens}",
    "api_tokens": "**API токены:** {tokens}",
    "execution_time": "**Время выполнения:** {time:.2f}с",
    "deduplication": "**Дедупликация:** {duplicates} дублирующих вызовов предотвращено ({breakdown})",
    "total_tool_calls": "**Всего вызовов инструментов:** {calls}",
    
    # Agent status details
    "agent_status_ready": "✅ **Агент готов**",
    "agent_status_initializing": "🟡 **Инициализация агента**",
    "provider_label": "**Провайдер:** {provider}",
    "model_label": "**Модель:** {model}",
    "status_label": "**Статус:** {status}",
    "tools_label": "**Инструменты:** {count} доступно",
    "last_used_label": "**Последнее использование:** {time}",
    "healthy_status": "✅ Исправен",
    "unhealthy_status": "❌ Неисправен",
    
    # Statistics labels
    "agent_status_section": "**Статус агента:**",
    "conversation_section": "**Диалог:**",
    "tools_section": "**Инструменты:**",
    "token_usage_section": "**Расход токенов:**",
    "messages_label": "Сообщения",
    "user_messages_label": "Пользователь",
    "assistant_messages_label": "Ассистент",
    "total_messages_label": "Всего сообщений",
    "available_label": "Доступно",
    "used_label": "Использовано",
    "total_persistent_label": "Всего",
    "current_conversation_label": "Текущий диалог",
    "average_per_message_label": "Среднее на сообщение",
    "tokens_label": "токенов",
    "unique_tools_label": "уникальных инструментов",
    "total_calls_label": "Всего вызовов",
    "tools_used_label": "Использовано инструментов",
    
    # Quick action messages
    "quick_math_message": "Сколько будет 15 * 23 + 7? Покажите работу пошагово.",
    "quick_code_message": "Напиши функцию на Python проверяющую, является ли число простым. Напиши и запусти тесты.",
    "quick_explain_message": "Объясни концепцию машинного обучения простыми словами.",
    "quick_create_attr_message": (
        "Составь план для создания текстового атрибута \"ID клиента\" в приложении \"ERP\", шаблон \"Контрагенты\" "
        "с display_format=CustomMask и маской ([0-9]{{10}}|[0-9]{{12}}), system_name=CustomerID. "
        "Представь: Намерение, План, Проверку и предварительный просмотр (DRY-RUN) аргументов (компактный JSON) для вызова инструмента, "
        "Но не выполняй никаких изменений пока. Жди моего подтверждения."
    ),
    "quick_edit_mask_message": (
        "Подготовь безопасный план редактирования атрибута \"Контактный телефон\" (system_name=ContactPhone) в приложении \"CRM\", шаблон \"Лиды\". "
        "Измени display_format на PhoneRuMask. Представь: Намерение, План, Контрольный список проверки (заметки о рисках) и предварительный просмотр запроса (DRY-RUN). "
        "Не выполняй изменения, ожидай моего одобрения."
    ),
    "quick_list_apps_message": (
        "Покажи список всех приложений в Comindware Platform. "
        "Отформатируй красиво в Markdown. "
        "Покажи системные имена, описания, ссылки, если есть."
    ),
    
    # Query example buttons (converted from try_asking_examples)
    "quick_edit_enum": "📝 Редактировать «Список значений»",
    "quick_edit_enum_message": "Получи атрибут типа enum \"Статус\" из приложения \"CRM\", шаблон \"Лиды\", затем добавь к нему новое значение \"В работе\" (system_name: in_progress, color: #FF9800) и обнови атрибут",
    
    "quick_templates_erp": "📄 Шаблоны ERP",
    "quick_templates_erp_message": "Покажи все шаблоны записей в приложении \"ERP\". Отформатируй красиво используя Markdown.",
    
    "quick_attributes_contractors": "🏷️ Атрибуты контрагентов",
    "quick_attributes_contractors_message": "Выдай список всех атрибутов шаблона \"Контрагенты\", приложение \"ERP\"",
    
    "quick_create_comment_attr": "💬 Создать атрибут комментария",
    "quick_create_comment_attr_message": "Создать текстовый атрибут \"Комментарий\", приложение \"HR\", шаблон \"Кандидаты\"",
    
    "quick_create_id_attr": "🆔 Создать атрибут ID",
    "quick_create_id_attr_message": "Создай текстовый атрибут \"ID клиента\", приложение \"ERP\", шаблон \"Контрагенты\", особая маска ввода: ([0-9]{10}|[0-9]{12})",
    
    "quick_edit_phone_mask": "📞 Редактировать маску телефона",
    "quick_edit_phone_mask_message": "Для атрибута \"Контактный телефон\" в приложении \"CRM\", шаблон \"Лиды\", смени формат отображения на российский телефон",
    
    "quick_get_comment_attr": "🔍 Получить атрибут комментария",
    "quick_get_comment_attr_message": "Получи атрибут: системное имя \"Комментарий\", приложение \"HR\", шаблон \"Кандидаты\"",
    
    "quick_edit_date_time": "📅 Настроить дату/время",
    "quick_edit_date_time_message": "Создай атрибут даты/времени \"Дата создания заявки\" в приложении \"CRM\", шаблон \"Лиды\" с форматом отображения LongDateLongTime и используй его как заголовок записи для автоматической сортировки по времени",
    
    "quick_archive_attr": "📦 Архивировать атрибут",
    "quick_archive_attr_message": "Архивируй/разархивируй атрибут, системное имя \"Комментарий\", приложение \"HR\", шаблон \"Кандидаты\"",
    
    # Status messages
    "processing_complete": "🎉 Обработка завершена",
    "response_completed": "Ответ завершен",
    "processing_failed": "Обработка не удалась",
    
    # Iteration messages
    "iteration_processing": "Итерация {iteration}/{max_iterations} - Обработка...",
    "iteration_finished": "Итерация {iteration}/{max_iterations} - Завершена",
    "iteration_completed": "Итерация {iteration} завершена - Продолжение...",
    "iteration_max_reached": "Итерация {iteration}/{max_iterations} - Завершена (достигнут максимум)",
    "max_iterations_warning": "⚠️ Достигнут лимит итераций ({max_iterations}), диалог может быть неполным",
    
    # Tool messages
    "tool_called": "🔧 Вызван инструмент: {tool_name}",
    "call_count": "Количество вызовов: {total_calls}",
    "result": "**Результат:** {tool_result}",
    "tool_error": "❌ **Ошибка инструмента: {error}**",
    "unknown_tool": "❌ **Неизвестный инструмент: {tool_name}**",
    
    # Error messages
    "error": "❌ **Ошибка: {error}**"
}

# English translations (fallback)
ENGLISH_TRANSLATIONS = {
    # App title and header
    "app_title": "Comindware Analyst Copilot",
    "hero_title": "Analyst Copilot",
    
    # Tab labels
    "tab_chat": "💬 Chat",
    "tab_logs": "📜 Logs", 
    "tab_stats": "📊 Statistics",
    
    # Chat tab content
    "welcome_title": "💬 Welcome!",
    "welcome_description": """
    The Comindware Analyst Copilot focuses on the **Comindware Platform** entity operations (applications, templates, attributes) and uses deterministic tools to execute precise changes.

    - **Platform operations first**: Validates your intent and executes tools for entity changes (e.g., create/edit attributes)
    - **Multi-model orchestration**: Supports multiple LLM providers
    - **Compact structured output**: Intent → Plan → Validate → Execute → Result
    """,
    
    "try_asking_title": "❓ Try asking",
    "try_asking_examples": """
    - List all applications in the platform. Format nicely using Markdown
    - List all record templates in app \"ERP\". Format as a list
    - List all attributes in template \"Counterparties\", app \"ERP\"
    - Create plain text attribute \"Comment\", app \"HR\", template \"Candidates\"
    - Create \"Customer ID\" text attribute, app \"ERP\", template \"Counterparties\", custom input mask ([0-9]{10}|[0-9]{12})
    - For attribute \"Contact Phone\" in app \"CRM\", template \"Leads\", change display format to Russian phone
    - Fetch attribute: system name \"Comment\", app \"HR\", template \"Candidates\"
    - Archive/unarchive attribute, system name \"Comment\", app \"HR\", template \"Candidates\"
    """,
    
    # Quick actions
    "quick_actions_title": "⚡ Quick Actions",
    "quick_list_apps": "🔎 List all apps",
    "quick_create_attr": "🧩 Create text attribute",
    "quick_edit_mask": "🛠️ Edit phone mask",
    "quick_math": "🧮 15 * 23 + 7 = ?",
    "quick_code": "💻 Python prime check function",
    "quick_explain": "💭 Explain ML briefly",
    
    # Chat interface
    "chat_label": "Chat with the Agent",
    "message_label": "Your Message",
    "message_placeholder": "Type your message here",
    "send_button": "Send",
    "clear_button": "Clear chat",
    
    # Status section
    "status_title": "🤖 Status",
    "status_initializing": "🟡 Initializing...",
    "status_ready": "✅ Ready",
    "progress_title": "📊 Progress",
    "progress_ready": "Ready to process your request...",
    
    # Logs tab
    "logs_title": "Initialization Logs",
    "logs_initializing": "🟡 Starting initialization...",
    "refresh_logs_button": "🔄 Refresh Logs",
    "clear_logs_button": "🗑️ Clear Logs",
    "logs_cleared": "Logs cleared.",
    "logs_not_available": "Logs not available - main app not connected",
    
    # Stats tab
    "stats_title": "Agent Statistics",
    "stats_loading": "Loading statistics...",
    "refresh_stats_button": "🔄 Refresh Stats",
    "agent_not_available": "Agent not available",
    "error_loading_stats": "Error loading statistics",
    
    # Status messages
    "agent_ready": "✅ **Agent Ready**",
    "agent_initializing": "🟡 **Agent Initializing**",
    "agent_not_ready": "❌ **Agent not ready. Please wait for initialization to complete.**",
    
    # Error messages
    "error_processing": "❌ **Error processing message: {error}**",
    "error_streaming": "❌ **Error streaming message: {error}**",
    "error_agent_timeout": "❌ **Agent initialization timeout**",
    "error_initialization_failed": "❌ **Initialization failed: {error}**",
    
    # Token and execution info
    "prompt_tokens": "**Prompt tokens:** {tokens}",
    "api_tokens": "**API tokens:** {tokens}",
    "execution_time": "**Execution time:** {time:.2f}s",
    "deduplication": "**Deduplication:** {duplicates} duplicate calls prevented ({breakdown})",
    "total_tool_calls": "**Total tool calls:** {calls}",
    
    # Agent status details
    "agent_status_ready": "✅ **Agent Ready**",
    "agent_status_initializing": "🟡 **Agent Initializing**",
    "provider_label": "**Provider:** {provider}",
    "model_label": "**Model:** {model}",
    "status_label": "**Status:** {status}",
    "tools_label": "**Tools:** {count} available",
    "last_used_label": "**Last Used:** {time}",
    "healthy_status": "✅ Healthy",
    "unhealthy_status": "❌ Unhealthy",
    
    # Statistics labels
    "agent_status_section": "**Agent Status:**",
    "conversation_section": "**Conversation:**",
    "tools_section": "**Tools:**",
    "token_usage_section": "**Token Usage:**",
    "messages_label": "Messages",
    "user_messages_label": "User",
    "assistant_messages_label": "Copilot",
    "total_messages_label": "Total Messages",
    "available_label": "Available",
    "used_label": "Used",
    "total_persistent_label": "Total (Persistent)",
    "current_conversation_label": "Current Conversation",
    "average_per_message_label": "Average per Message",
    "tokens_label": "tokens",
    "unique_tools_label": "unique tools",
    "total_calls_label": "Total Calls",
    "tools_used_label": "Used tools",
    
    # Quick action messages
    "quick_math_message": "What is 15 * 23 + 7? Please show your work step by step.",
    "quick_code_message": "Write a Python function to check if a number is prime. Include tests.",
    "quick_explain_message": "Explain the concept of machine learning in simple terms.",
    "quick_create_attr_message": (
        "Draft a plan to CREATE a text attribute \"Customer ID\" in application \"ERP\", template \"Counterparties\" "
        "with display_format=CustomMask and mask ([0-9]{{10}}|[0-9]{{12}}), system_name=CustomerID. "
        "Provide Intent, Plan, Validate, and a DRY-RUN payload preview (compact JSON) for the tool call, "
        "but DO NOT execute any changes yet. Wait for my confirmation."
    ),
    "quick_edit_mask_message": (
        "Prepare a safe EDIT plan for attribute \"Contact Phone\" (system_name=ContactPhone) in application \"CRM\", template \"Leads\" "
        "to change display_format to PhoneRuMask. Provide Intent, Plan, Validate checklist (risk notes), and a DRY-RUN payload preview. "
        "Do NOT execute changes yet—await my approval."
    ),
    "quick_list_apps_message": (
        "List all applications in the platform. "
        "Format nicely using Markdown. "
        "Show system names and descriptions if any."
    ),
    
    # Query example buttons (converted from try_asking_examples)
    "quick_edit_enum": "📝 Edit Enum",
    "quick_edit_enum_message": "Get the enum attribute \"Status\" from application \"CRM\", template \"Leads\", then add a new value \"In Progress\" (system_name: in_progress, color: #FF9800) and update the attribute",
    
    "quick_templates_erp": "📄 ERP Templates",
    "quick_templates_erp_message": "Show all record templates in the \"ERP\" application. Format nicely using Markdown.",
    
    "quick_attributes_contractors": "🏷️ Contractor Attributes",
    "quick_attributes_contractors_message": "Get a list of all attributes of the \"Counterparties\" template, application \"ERP\"",
    
    "quick_create_comment_attr": "💬 Create Comment Attribute",
    "quick_create_comment_attr_message": "Create a text attribute \"Comment\", application \"HR\", template \"Candidates\"",
    
    "quick_create_id_attr": "🆔 Create ID Attribute",
    "quick_create_id_attr_message": "Create a text attribute \"Customer ID\", application \"ERP\", template \"Counterparties\", special input mask: ([0-9]{10}|[0-9]{12})",
    
    "quick_edit_phone_mask": "📞 Edit Phone Mask",
    "quick_edit_phone_mask_message": "For the \"Contact Phone\" attribute in application \"CRM\", template \"Leads\", change the display format to Russian phone",
    
    "quick_get_comment_attr": "🔍 Get Comment Attribute",
    "quick_get_comment_attr_message": "Get attribute: system name \"Comment\", application \"HR\", template \"Candidates\"",
    
    "quick_edit_date_time": "📅 Configure Date/Time",
    "quick_edit_date_time_message": "Create a date/time attribute \"Lead Creation Date\" in application \"CRM\", template \"Leads\" with LongDateLongTime display format and use it as record title for automatic time-based sorting",
    
    "quick_archive_attr": "📦 Archive Attribute",
    "quick_archive_attr_message": "Archive/unarchive attribute, system name \"Comment\", application \"HR\", template \"Candidates\"",
    
    # Status messages
    "processing_complete": "🎉 Processing complete",
    "response_completed": "Response completed",
    "processing_failed": "Processing failed",
    
    # Iteration messages
    "iteration_processing": "Iteration {iteration}/{max_iterations} - Processing...",
    "iteration_finished": "Iteration {iteration}/{max_iterations} - Finished",
    "iteration_completed": "Iteration {iteration} completed - Continuing...",
    "iteration_max_reached": "Iteration {iteration}/{max_iterations} - Finished (Max Reached)",
    "max_iterations_warning": "⚠️ Reached maximum iterations ({max_iterations}), conversation may be incomplete",
    
    # Tool messages
    "tool_called": "🔧 Tool called: {tool_name}",
    "call_count": "Call count: {total_calls}",
    "result": "**Result:** {tool_result}",
    "tool_error": "❌ **Tool error: {error}**",
    "unknown_tool": "❌ **Unknown tool: {tool_name}**",
    
    # Error messages
    "error": "❌ **Error: {error}**"
}

def create_i18n_instance(language: str = "en") -> gr.I18n:
    """
    Create a Gradio I18n instance with translations for all supported languages.
    
    Args:
        language: Language code ('en' or 'ru') - used for default language selection
    
    Returns:
        Gradio I18n instance with both English and Russian translations
    """
    return gr.I18n(
        en=ENGLISH_TRANSLATIONS,
        ru=RUSSIAN_TRANSLATIONS
    )

def get_translation_key(key: str, language: str = "en") -> str:
    """
    Get a translation for a specific key in the specified language.
    
    Args:
        key: Translation key
        language: Language code ('en' or 'ru')
        
    Returns:
        Translated string
    """
    if language.lower() == "ru":
        return RUSSIAN_TRANSLATIONS.get(key, ENGLISH_TRANSLATIONS.get(key, key))
    else:
        return ENGLISH_TRANSLATIONS.get(key, key)

def format_translation(key: str, language: str = "en", **kwargs) -> str:
    """
    Get a formatted translation for a specific key with variable substitution.
    
    Args:
        key: Translation key
        language: Language code ('en' or 'ru')
        **kwargs: Variables to substitute in the translation
        
    Returns:
        Formatted translated string
    """
    template = get_translation_key(key, language)
    try:
        return template.format(**kwargs)
    except KeyError as e:
        print(f"Warning: Missing format variable {e} for key '{key}' in language '{language}'")
        return template
