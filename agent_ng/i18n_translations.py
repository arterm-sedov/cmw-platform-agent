"""
Internationalization (i18n) Translations for CMW Platform Agent
================================================================

Provides Russian and English translations for all UI text in the Gradio application.
Uses Gradio's built-in I18n class for seamless localization support.

Based on Gradio's internationalization documentation:
https://www.gradio.app/guides/internationalization
"""

from typing import Any

import gradio as gr

# Russian translations for all UI text
RUSSIAN_TRANSLATIONS = {
    # Language detection resource (fake resource for Gradio I18n)
    "language": "ru",
    # App title and header
    "app_title": "Ассистент аналитика Comindware",
    "hero_title": "Ассистент аналитика",
    # Tab labels
    "tab_home": "🏠 Главная",
    "tab_chat": "💬 Чат",
    "tab_logs": "📜 Журналы",
    "tab_stats": "📊 Статистика",
    "tab_config": "⚙️ Настройки",
    "tab_sidebar": "⚙️ Настройки и управление",
    "tab_downloads": "📥 Загрузки",
    # Home tab content
    "welcome_title": "Добро пожаловать!",
    "welcome_description": """
    **Ассистент аналитика Comindware** предназначен для работы с сущностями **Comindware Platform**, такими как приложения, шаблоны, атрибуты.

    Ассистент использует детерминированные инструменты, не полагающиеся на языковую модель, а взаимодействующие с API **Comindware Platform**.

    **Основные возможности:**
    - **Анализ сущностей**: глубокий анализ приложений, шаблонов и атрибутов в **Comindware Platform**.
    - **Работа с атрибутами**: создание, редактирование, удаление атрибутов всех типов.
    - **Локализация**: агент отвечает на языке вопроса, но может поддерживать разговор на любом языке, который поддерживает выбранная модель. Внутренние размышления агент выполняет на английском языке. Агент может создавать имена сущностей на любых языках. Интерфейс переведён на английский и русский языки.
    """,
    "quick_start_title": "Быстрый старт",
    "quick_start_description": """
    1. Настройте подключение к **Comindware Platform** на вкладке «**Настройки**».
    2. Перейдите на вкладку «**Чат**» для начала разговора.
    3. Выберите модель в боковой панели.
    4. Введите свой вопрос или выберите **заготовку** в левой панели и отредактируйте её, например:
        - Что ты умеешь?
        - Чего ты не умеешь?
        - Перечисли все приложения в платформе в удобном списке.
        - Дай полный аудит всех приложений, шаблонов и атрибутов в системе.
        - Покажи все шаблоны записей в приложении "ERP".
        - Выдай список всех атрибутов шаблона "Контрагенты", приложение "ERP".
        - Создай текстовый атрибут "Комментарий", приложение "HR", шаблон "Кандидаты".
        - Создай текстовый атрибут "ID клиента", приложение "ERP", шаблон "Контрагенты", особая маска ввода: ([0-9]{10}|[0-9]{12}).
        - Для атрибута "Контактный телефон" в приложении "CRM", шаблон "Лиды", смени формат отображения на российский телефон.
        - Получи атрибут: системное имя "Комментарий", приложение "HR", шаблон "Кандидаты".
        - Архивируй/разархивируй атрибут: системное имя "Комментарий", приложение "HR", шаблон "Кандидаты".
    5. Нажмите кнопку «**Отправить**» и дождитесь ответа агента.
    6. Агент может отвечать некоторое время, особенно если требуется обращение к **Comindware Platform**.  Прогресс отображается в левой панели.
    7. По окончании работы агент выдаст сообщение «Обработка завершена» в левой панели.
    """,
    # Quick actions
    "quick_actions_title": "Заготовки",
    "quick_actions_dropdown_placeholder": "Выберите быструю команду...",
    "quick_actions_dropdown_label": "Быстрые команды",
    # History compression
    "compression_enabled_label": "При переполнении сжимать диалог",
    "quick_list_apps": "🔎 Список всех приложений",
    "quick_create_attr": "🧩 Создать текстовый атрибут",
    "quick_edit_mask": "🛠️ Редактировать маску телефона",
    "quick_math": "🧮 15 * 23 + 7 = ?",
    "quick_code": "💻 Функция проверки простых чисел на Python",
    "quick_explain": "💭 Объяснить ML кратко",
    "quick_full_audit": "🧾 Полный аудит системы",
    # Chat interface
    "chat_label": "Диалог с агентом",
    "message_label": "Ваше сообщение",
    "message_placeholder": "Введите ваше сообщение...",
    "send_button": "Отправить",
    "stop_button": "⏹️ Остановить",
    "clear_button": "Очистить диалог",
    "download_button": "📥 Скачать диалог (Markdown)",
    "download_html_button": "🌐 Скачать диалог (HTML)",
    "download_file_label": "Скачать Markdown",
    # LLM Selection section
    "llm_selection_title": "Выбор LLM",
    "provider_label": "Провайдер",
    "model_label": "Модель",
    "provider_model_label": "Провайдер / модель",
    "apply_llm_button": "Применить",
    "llm_apply_success": "✅ LLM изменена: {provider} - {model}",
    "llm_apply_error": "❌ Ошибка применения LLM",
    # Mistral switching confirmation
    "mistral_switch_title": "⚠️ Внимание! Переключение на Mistral",
    "mistral_switch_warning": """
    Идёт переключение на {provider} / {model}

    Mistral не совместима с другими нейросетями.
    Для продолжения работы чат был очищен.
    """,
    "mistral_chat_cleared": "Чат очищен для совместимости с Mistral.",
    "mistral_switch_cancelled": "❌ Переключение на Mistral отменено",
    "current_provider": "Провайдер: {provider}",
    "current_model": "Модель: {model}",
    "model_pricing_label": "Цена",
    "model_pricing_input_label": "Ввод",
    "model_pricing_output_label": "Вывод",
    # Status section
    "status_title": "Статус",
    "status_initializing": "🟡 Инициализация...",
    "status_ready": "Готов",
    "status_ready_true": "Готов: да ✅",
    "status_ready_false": "Готов: нет ❌",
    "token_budget_title": "Расход токенов",
    "token_budget_initializing": "🟡 Загрузка...",
    "token_statistics_title": "📊 Статистика",
    # Token usage components (separated for flexibility)
    "token_usage_header": "Расход токенов:",
    "token_usage_total": "Всего: {total_tokens:,}",
    "token_usage_conversation": "Диалог: {conversation_tokens:,}",
    "token_usage_estimate": "Прогноз: {estimated_tokens:,}",
    "token_usage_last_message": "Сообщение:",
    "token_usage_average": "Сред. сообщение: {avg_tokens:,}",
    "token_breakdown_context": "Контекст: {conv_tokens:,}",
    "token_breakdown_tools": "Инструменты: {tool_tokens:,}",
    "token_breakdown_overhead": "Накладные: {overhead_tokens:,}",
    "token_message_context": "Контекст: {percentage}% ({used:,}/{context_window:,}) {status_icon}",
    "token_message_input": "Входные: {tokens:,}",
    "token_message_output": "Выходные: {tokens:,}",
    "token_message_cached_tokens": "Кэш (чтение): {tokens:,}",
    "token_message_cache_write_tokens": "Кэш (запись): {tokens:,}",
    "token_message_cost": "Стоимость: {cost}",
    # Status icons for token usage
    "token_status_good": "🟢",
    "token_status_moderate": "🟡",
    "token_status_warning": "🟠",
    "token_status_critical": "🔴",
    "token_status_unknown": "❓",
    # Legacy combined format (for backward compatibility)
    "token_budget_detailed": """Расход токенов:
- Всего: {total_tokens:,}
- Диалог: {conversation_tokens:,}
- Сообщение {percentage}% ({used:,}/{context_window:,}) {status_icon}
- Среднее на сообщение: {avg_tokens:,}""",
    "token_budget_unknown": "❓ Неизвестно",
    "progress_title": "Прогресс",
    "progress_ready": "Готов к работе…",
    "progress_processing": "Обработка запроса...",
    # Logs tab
    "logs_title": "Журналы инициализации",
    "logs_initializing": "🟡 Идёт инициализация...",
    "refresh_logs_button": "🔄 Обновить журналы",
    "clear_logs_button": "🗑️ Очистить журналы",
    "logs_cleared": "Журналы очищены.",
    "logs_not_available": "Журналы недоступны — основное приложение не подключено",
    # Turn summary translations
    "conversation_summary": "Сводка диалога",
    "roles_sequence": "Роли",
    "tools_used_total": "Инструменты",
    "providers_models_total": "Провайдеры / модели",
    # Stats tab
    "stats_title": "Статистика агента",
    "stats_loading": "Загрузка статистики...",
    "refresh_stats_button": "🔄 Обновить статистику",
    "clear_stats_button": "🗑️ Очистить статистику",
    "stats_cleared": "Статистика очищена",
    "agent_not_available": "Агент недоступен",
    "stats_auto_refresh_message": "📊 Статистика обновляется автоматически. Нажмите кнопку обновления для просмотра данных сессии.",
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
    "turn_cost": "**Стоимость запроса:** {cost}",
    "conversation_cost": "**Стоимость диалога:** {cost}",
    "total_cost": "**Итого:** {cost}",
    "cost_label": "стоимость: {cost}",
    "last_turn_cost": "Последний запрос: {cost}",
    "input_tokens_label": "**Входные токены:** {tokens:,}",
    "output_tokens_label": "**Выходные токены:** {tokens:,}",
    "execution_time": "**Время выполнения:** {time:.2f}с",
    "provider_model": "**Провайдер / модель:** {provider} / {model}",
    "deduplication": "**Дедупликация:** {duplicates} дублирующих вызовов предотвращено ({breakdown})",
    "total_tool_calls": "**Всего вызовов инструментов:** {calls}",
    "no_content_response": "⚠️ **Нет ответа от агента.** Попробуйте ещё раз, измените формулировку или переключите модель.",
    # Agent status details
    "agent_status_ready": "✅ **Агент готов**",
    "agent_status_initializing": "🟡 **Инициализация агента**",
    "provider_info": "Провайдер: {provider}",
    "model_info": "**Модель:** {model}",
    "status_label": "**Статус:** {status}",
    "tools_count_label": "**Инструменты:** {count} доступно",
    "last_used_label": "**Последнее использование:** {time}",
    "healthy_status": "✅ Исправен",
    "unhealthy_status": "❌ Неисправен",
    # Config tab
    "config_title": "Подключение к Comindware Platform",
    "config_platform_url": "Адрес сервера",
    "config_username": "Имя пользователя",
    "config_password": "Пароль",
    "config_save_button": "💾 Сохранить в браузере",
    "config_load_button": "🔄 Загрузить из браузера",
    "config_save_success_session": "✅ Настройки применены для текущего сеанса",
    "config_save_error": "❌ Ошибка сохранения настроек",
    "config_load_success": "✅ Настройки загружены",
    "config_load_error": "❌ Ошибка загрузки настроек",
    "config_clear_storage_button": "🧹 Очистить хранилище браузера",
    "config_clear_success": "✅ Хранилище браузера очищено",
    "config_clear_error": "❌ Не удалось очистить хранилище браузера",
    "config_platform_dotenv_notice": (
        "Используется преднастроенное подключение к Comindware Platform / "
        "Using the preconfigured connection to the Comindware Platform."
    ),
    "config_help": (
        """
        Задайте параметры подключения к **Comindware Platform**:

        - **Адрес сервера** — URL вашего сайта **Comindware Platform**,
        например `https://your-host`.
        - **Имя пользователя** и **Пароль** — учетные данные для
        использования API. **Обычный аккаунт не будет работать.**
        - Нажмите «**Сохранить в браузере**», чтобы применить параметры к
        текущему сеансу.
        - Кнопка «**Загрузить из браузера**» восстановит ранее сохранённые
        значения.
        - **Очистить хранилище браузера** удалит сохранённые параметры и
          очистит поля.

        Данные сохраняются в браузере и применяются только в
        пределах текущего сеанса.
        """
    ),
    # LLM override fields
    "config_llm_section": "Подключение к LLM",
    "config_llm_provider_label": "Провайдер",
    "config_llm_api_key_label": "Ключ API",
    "config_llm_api_keys_table_label": "Ключи API провайдеров",
    "config_llm_empty_means_default": "Оставьте пустым, чтобы использовать настройки по умолчанию",
    # Statistics labels
    "agent_status_section": "**Агент:**",
    "conversation_section": "**Диалог:**",
    "token_usage_section": "**Использование токенов:**",
    "cost_section": "**Стоимость:**",
    "token_usage_overall": "Всего (все диалоги): {total_tokens:,}",
    "avg_tokens_per_message_label": "Среднее на сообщение: {avg:,}",
    "tools_section": "**Инструменты:**",
    "messages_label": "Сообщения",
    "user_messages_label": "Пользователь",
    "assistant_messages_label": "Ассистент",
    "total_messages_label": "Всего сообщений",
    "available_label": "Доступно",
    "used_label": "Использовано",
    "unique_tools_label": "уникальных",
    "total_calls_label": "Инструменты",
    "tools_used_label": "Использовано инструментов",
    "tools_label": "Инструментов",
    "system_prompt_label": "Система",
    "memory_entries": "Записей в памяти: {count}",
    # Quick action messages
    "quick_math_message": "Сколько будет 15 * 23 + 7? Покажите работу пошагово.",
    "quick_code_message": "Напиши функцию на Python проверяющую, является ли число простым. Напиши и запусти тесты.",
    "quick_explain_message": "Поищи в интернете. Объясни концепцию машинного обучения простыми словами.",
    "quick_create_attr_message": (
        'Составь план для создания текстового атрибута "ID клиента" в приложении "ERP", шаблон "Контрагенты" '
        "с display_format=CustomMask и маской ([0-9]{{10}}|[0-9]{{12}}), system_name=CustomerID. "
        "Представь: Намерение, План, Проверку и предварительный просмотр (DRY-RUN) аргументов (компактный JSON) для вызова инструмента, "
        "Но не выполняй никаких изменений пока. Жди моего подтверждения."
    ),
    "quick_edit_mask_message": (
        'Подготовь безопасный план редактирования атрибута "Контактный телефон" (system_name=ContactPhone) в приложении "CRM", шаблон "Лиды". '
        "Измени display_format на PhoneRuMask. Представь: Намерение, План, Контрольный список проверки (заметки о рисках) и предварительный просмотр запроса (DRY-RUN). "
        "Не выполняй изменения, ожидай моего одобрения."
    ),
    "quick_list_apps_message": (
        "Покажи список всех приложений в Comindware Platform. Отформатируй красиво в Markdown."
    ),
    # Query example buttons (converted from try_asking_examples)
    "quick_edit_enum": "📝 Редактировать «Список значений»",
    "quick_edit_enum_message": 'Получи атрибут типа enum "Статус" из приложения "CRM", шаблон "Лиды", затем добавь к нему новое значение "В работе" (system_name: in_progress, color: #FF9800) и обнови атрибут',
    "quick_templates_erp": "📄 Шаблоны ERP",
    "quick_templates_erp_message": 'Покажи все шаблоны записей в приложении "ERP". Отформатируй красиво используя Markdown.',
    "quick_attributes_contractors": "🏷️ Атрибуты контрагентов",
    "quick_attributes_contractors_message": 'Выдай список всех атрибутов шаблона "Контрагенты", приложение "ERP"',
    "quick_create_comment_attr": "💬 Создать атрибут комментария",
    "quick_create_comment_attr_message": 'Создать текстовый атрибут "Комментарий", приложение "HR", шаблон "Кандидаты"',
    "quick_create_id_attr": "🆔 Создать атрибут ID",
    "quick_create_id_attr_message": 'Создай текстовый атрибут "ID клиента", приложение "ERP", шаблон "Контрагенты", особая маска ввода: ([0-9]{10}|[0-9]{12})',
    "quick_edit_phone_mask": "📞 Редактировать маску телефона",
    "quick_edit_phone_mask_message": 'Для атрибута "Контактный телефон" в приложении "CRM", шаблон "Лиды", смени формат отображения на российский телефон',
    "quick_get_comment_attr": "🔍 Получить атрибут комментария",
    "quick_get_comment_attr_message": 'Получи атрибут: системное имя "Комментарий", приложение "HR", шаблон "Кандидаты"',
    "quick_edit_date_time": "📅 Настроить дату/время",
    "quick_edit_date_time_message": 'Создай атрибут даты/времени "Дата создания заявки" в приложении "CRM", шаблон "Лиды" с форматом отображения LongDateLongTime и используй его как заголовок записи для автоматической сортировки по времени',
    "quick_archive_attr": "📦 Архивировать атрибут",
    "quick_archive_attr_message": 'Архивируй/разархивируй атрибут, системное имя "Комментарий", приложение "HR", шаблон "Кандидаты"',
    "quick_what_can_do": "❓ Что ты умеешь?",
    "quick_what_can_do_message": "Что ты умеешь?",
    "quick_what_cannot_do": "❌ Чего ты не умеешь?",
    "quick_what_cannot_do_message": "Чего ты не умеешь?",
    "quick_full_audit_message": "Дай полный аудит всех приложений, шаблонов и атрибутов в системе.",
    # Status messages
    "processing_complete": "🎉Обработка завершена",
    "response_completed": "Ответ завершен",
    "processing_failed": "Обработка не удалась",
    # Initialization messages
    "session_manager_ready": "Менеджер сессий готов",
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
    "tools_available": "🔧 Доступно инструментов: {count}",
    "tool_calls_made": "Вызовы инструментов: {tool_names}",
    # Error messages
    "error": "❌ **Ошибка: {error}**",
    # Provider availability messages
    "no_providers_available": "❌ Нет доступных провайдеров",
    "no_models_available": "❌ Нет доступных моделей",
    "error_loading_providers": "❌ Ошибка загрузки провайдеров",
    # History compression
    "history_compression_title": "📦 История диалога сжата",
    "history_compression_info": "Сохранено ~{tokens_saved:,} токенов. Использование: {previous_pct:.1f}% → {current_pct:.1f}%. Сжатий: {compression_count}.",
    "history_compression_info_before": "Сжимаю диалог... ({previous_pct:.1f}% использовано)",
    "history_compression_reason_critical": "Критическое использование токенов (≥90%)",
    "history_compression_reason_proactive": "Сжимаю диалог для предотвращения переполнения",
    "history_compression_reason_interrupted": "Оборванный ход с критическим статусом",
    "compression_stats_label": "Статистика сжатия",
    "compression_count_label": "Сжатий: {count}",
    "compression_tokens_saved_label": "Токенов сохранено: {tokens:,}",
    "use_fallback_model_label": "При переполнении сменить модель",
    # UI Icons
    "clock_icons": [
        "🕐",
        "🕑",
        "🕒",
        "🕓",
        "🕔",
        "🕕",
        "🕖",
        "🕗",
        "🕘",
        "🕙",
        "🕚",
        "🕛",
    ],
    "finish_icons": ["🎉", "🏁", "✨", "🎯"],
    "completion_icons": ["✅", "✔️", "🎯", "✨"],
    "max_icons": ["⚠️", "⏰", "🔄", "⚡"],
    "completion_final_icons": ["✅", "🎯", "✨", "🏆"],
    "error_icons": ["❌", "💥", "⚠️", "🚫"],
}

# English translations (fallback)
ENGLISH_TRANSLATIONS = {
    # Language detection resource (fake resource for Gradio I18n)
    "language": "en",
    # App title and header
    "app_title": "Comindware Analyst Copilot",
    "hero_title": "Analyst Copilot",
    # Tab labels
    "tab_home": "🏠 Home",
    "tab_chat": "💬 Chat",
    "tab_logs": "📜 Logs",
    "tab_stats": "📊 Statistics",
    "tab_config": "⚙️ Config",
    "tab_sidebar": "⚙️ Settings & Control",
    "tab_downloads": "📥 Downloads",
    # Home tab content
    "welcome_title": "Welcome!",
    "welcome_description": """
    **Comindware Analyst Copilot** is designed to work with **Comindware Platform** entities such as applications, templates, and attributes.

    The Copilot uses deterministic tools that do not rely on language models but interact with **Comindware Platform** APIs.

    **Key Features:**
    - **Entity Analysis**: Deep analysis of applications, templates, and attributes in **Comindware Platform**.
    - **Attribute Management**: Creating, editing, and deleting attributes of all types.
    - **Localization**: The agent responds in the language of the question, but can maintain conversations in any language supported by the selected model. Internal reasoning is performed in English. The agent can create entity names in any language. The interface is translated into English and Russian languages.
    """,
    "quick_start_title": "Quick Start",
    "quick_start_description": """
    1. Configure the connection to the **Comindware Platform** in the **Config** tab.
    2. Go to the **Chat** tab to start a conversation.
    3. Select a model in the sidebar.
    4. Enter your question or select a **template** in the sidebar and edit it, for example:
        - What can you do?
        - What can't you do?
        - List all applications in the platform in a convenient list.
        - Give a full audit of all applications, templates and attributes in the system.
        - Show all record templates in the "ERP" application.
        - Get a list of all attributes of the "Counterparties" template, application "ERP"
        - Create a text attribute "Comment", application "HR", template "Candidates"
        - Create a text attribute "Customer ID", application "ERP", template "Counterparties", special input mask: ([0-9]{10}|[0-9]{12})
        - For the "Contact Phone" attribute in application "CRM", template "Leads", change the display format to Russian phone
        - Get attribute: system name "Comment", application "HR", template "Candidates"
        - Archive/unarchive attribute, system name "Comment", application "HR", template "Candidates"
    5. Click **Send** and wait for the agent's response.
    6. The agent may take some time to respond, especially when accessing the **Comindware Platform**. Progress is displayed in the sidebar.
    7. When finished, the agent will show **Processing complete** message in the sidebar.
    """,
    # Quick actions
    "quick_actions_title": "Templates",
    "quick_actions_dropdown_placeholder": "Select a quick command...",
    "quick_actions_dropdown_label": "Quick Commands",
    # History compression
    "compression_enabled_label": "Щn overflow сompress conversation",
    "use_fallback_model_label": "On overflow use larger model",
    "quick_list_apps": "🔎 List all apps",
    "quick_create_attr": "🧩 Create text attribute",
    "quick_edit_mask": "🛠️ Edit phone mask",
    "quick_math": "🧮 15 * 23 + 7 = ?",
    "quick_code": "💻 Python prime check function",
    "quick_explain": "💭 Explain ML briefly",
    "quick_full_audit": "🧾 Full system audit",
    # Chat interface
    "chat_label": "Conversation with the Agent",
    "message_label": "Your Message",
    "message_placeholder": "Type your message here",
    "send_button": "Send",
    "stop_button": "⏹️ Stop",
    "clear_button": "Clear conversation",
    "download_button": "📥 Download conversation (Markdown)",
    "download_html_button": "🌐 Download conversation (HTML)",
    "download_file_label": "Download Markdown",
    # LLM Selection section
    "llm_selection_title": "LLM Selection",
    "provider_label": "Provider",
    "model_label": "Model",
    "provider_model_label": "Provider / model",
    "apply_llm_button": "Apply",
    "llm_apply_success": "✅ LLM changed: {provider} - {model}",
    "llm_apply_error": "❌ Error applying LLM",
    # Mistral switching confirmation
    "mistral_switch_title": "⚠️ Warning! Switching to Mistral",
    "mistral_switch_warning": """
    Switching to {provider} / {model}...

    Mistral is not compatible with other providers.
    The chat was cleared to continue working.
    """,
    "mistral_chat_cleared": "Chat cleared for Mistral compatibility.",
    "mistral_switch_cancelled": "❌ Mistral switching cancelled",
    "current_provider": "Provider: {provider}",
    "current_model": "**Model:** {model}",
    "model_pricing_label": "Pricing",
    "model_pricing_input_label": "Input",
    "model_pricing_output_label": "Output",
    # Status section
    "status_title": "Status",
    "status_initializing": "🟡 Initializing...",
    "status_ready": "Ready",
    "status_ready_true": "Ready: yes ✅",
    "status_ready_false": "Ready: no ❌",
    "token_budget_title": "Token usage",
    "token_budget_initializing": "🟡 Loading...",
    "token_statistics_title": "📊 Statistics",
    # Token usage components (separated for flexibility)
    "token_usage_header": "**Token usage:**",
    "token_usage_total": "Total: {total_tokens:,}",
    "token_usage_conversation": "Conversation: {conversation_tokens:,}",
    "token_usage_estimate": "Forecast: {estimated_tokens:,}",
    "token_usage_last_message": "Message:",
    "token_usage_average": "Average per message: {avg_tokens:,}",
    "token_breakdown_context": "Context: {conv_tokens:,}",
    "token_breakdown_tools": "Tools: {tool_tokens:,}",
    "token_breakdown_overhead": "Overhead: {overhead_tokens:,}",
    "token_message_context": "Context: {percentage}% ({used:,}/{context_window:,}) {status_icon}",
    "token_message_input": "Input: {tokens:,}",
    "token_message_output": "Output: {tokens:,}",
    "token_message_cached_tokens": "Cache (read): {tokens:,}",
    "token_message_cache_write_tokens": "Cache (write): {tokens:,}",
    "token_message_cost": "Cost: {cost}",
    # Status icons for token usage
    "token_status_good": "🟢",
    "token_status_moderate": "🟡",
    "token_status_warning": "🟠",
    "token_status_critical": "🔴",
    "token_status_unknown": "❓",
    # Legacy combined format (for backward compatibility)
    "token_budget_detailed": """Token usage:
- Total: {total_tokens:,}
- Conversation: {conversation_tokens:,}
- Last message {percentage}% ({used:,}/{context_window:,}) {status_icon}
- Average per message: {avg_tokens:,}""",
    "token_budget_unknown": "❓ Unknown",
    "progress_title": "Progress",
    "progress_ready": "Ready to process your request...",
    "progress_processing": "Processing request...",
    # Logs tab
    "logs_title": "Initialization Logs",
    "logs_initializing": "🟡 Starting initialization...",
    "refresh_logs_button": "🔄 Refresh Logs",
    "clear_logs_button": "🗑️ Clear Logs",
    "logs_cleared": "Logs cleared.",
    "logs_not_available": "Logs not available - main app not connected",
    # Turn summary translations
    "conversation_summary": "Conversation summary",
    "roles_sequence": "Roles",
    "tools_used_total": "Tools",
    "providers_models_total": "Providers / models",
    # Stats tab
    "stats_title": "Agent Statistics",
    "stats_loading": "Loading statistics...",
    "refresh_stats_button": "🔄 Refresh Stats",
    "clear_stats_button": "🗑️ Clear Stats",
    "stats_cleared": "Statistics cleared",
    "agent_not_available": "Agent not available",
    "stats_auto_refresh_message": "📊 Statistics are auto-refreshing. Click refresh button to view session data.",
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
    "turn_cost": "**Turn cost:** {cost}",
    "conversation_cost": "**Conversation cost:** {cost}",
    "total_cost": "**Total cost:** {cost}",
    "cost_label": "cost: {cost}",
    "last_turn_cost": "Last turn: {cost}",
    "input_tokens_label": "**Input tokens:** {tokens:,}",
    "output_tokens_label": "**Output tokens:** {tokens:,}",
    "execution_time": "**Execution time:** {time:.2f}s",
    "provider_model": "**Provider / model:** {provider} / {model}",
    "deduplication": "**Deduplication:** {duplicates} duplicate calls prevented ({breakdown})",
    "total_tool_calls": "**Total tool calls:** {calls}",
    "no_content_response": "⚠️ **No answer from the agent.** Please try again, rephrase, or switch the model.",
    # Agent status details
    "agent_status_ready": "✅ **Agent Ready**",
    "agent_status_initializing": "🟡 **Agent Initializing**",
    "provider_info": "Provider: {provider}",
    "model_info": "**Model:** {model}",
    "status_label": "**Status:** {status}",
    "tools_count_label": "**Tools:** {count} available",
    "last_used_label": "**Last Used:** {time}",
    "healthy_status": "✅ Healthy",
    "unhealthy_status": "❌ Unhealthy",
    # Config tab
    "config_title": "Comindware Platform Connection",
    "config_platform_url": "Server URL",
    "config_username": "Username",
    "config_password": "Password",
    "config_save_button": "💾 Save to browser",
    "config_load_button": "🔄 Load from browser",
    "config_save_success_session": "✅ Settings applied for current session",
    "config_save_error": "❌ Failed to save settings",
    "config_load_success": "✅ Settings loaded",
    "config_load_error": "❌ Failed to load settings",
    "config_clear_storage_button": "🧹 Clear browser storage",
    "config_clear_success": "✅ Browser storage cleared",
    "config_clear_error": "❌ Failed to clear browser storage",
    "config_platform_dotenv_notice": (
        "Используется преднастроенное подключение к Comindware Platform / "
        "Using the preconfigured connection to the Comindware Platform."
    ),
    "config_help": (
        """
        Configure connection to the **Comindware Platform**:

        - **Server URL** — your **Comindware Platform** website, e.g.,
        `https://your-host`.
        - **Username** and **Password** — credentials with API access rights.
        **Regular account won't work.**
        - Click **Save to browser** to apply settings for the current session.
        - **Load from browser** restores previously saved values.
        - **Clear browser storage** removes saved settings and clears the fields.

        Settings are stored in the browser storage and applied within the
        current session only.
        """
    ),
    # LLM override fields
    "config_llm_section": "LLM Connection",
    "config_llm_provider_label": "Provider",
    "config_llm_api_key_label": "API Key",
    "config_llm_api_keys_table_label": "Provider API keys",
    "config_llm_empty_means_default": "Leave empty to use default settings",
    # Statistics labels
    "agent_status_section": "**Agent:**",
    "conversation_section": "**Conversation:**",
    "token_usage_section": "**Token Usage:**",
    "cost_section": "**Cost:**",
    "token_usage_overall": "Total (all conversations): {total_tokens:,}",
    "avg_tokens_per_message_label": "Average per message: {avg:,}",
    "tools_section": "**Tools:**",
    "messages_label": "Messages",
    "user_messages_label": "User",
    "assistant_messages_label": "Copilot",
    "total_messages_label": "Total messages",
    "available_label": "Available",
    "used_label": "Used",
    "unique_tools_label": "unique",
    "total_calls_label": "Tools",
    "tools_used_label": "Used tools",
    "tools_label": "Tools",
    "system_prompt_label": "System",
    "memory_entries": "Memory entries: {count}",
    # Quick action messages
    "quick_math_message": "What is 15 * 23 + 7? Please show your work step by step.",
    "quick_code_message": "Write a Python function to check if a number is prime. Include tests.",
    "quick_explain_message": "Search the web. Explain the concept of machine learning in simple terms.",
    "quick_create_attr_message": (
        'Draft a plan to CREATE a text attribute "Customer ID" in application "ERP", template "Counterparties" '
        "with display_format=CustomMask and mask ([0-9]{{10}}|[0-9]{{12}}), system_name=CustomerID. "
        "Provide Intent, Plan, Validate, and a DRY-RUN payload preview (compact JSON) for the tool call, "
        "but DO NOT execute any changes yet. Wait for my confirmation."
    ),
    "quick_edit_mask_message": (
        'Prepare a safe EDIT plan for attribute "Contact Phone" (system_name=ContactPhone) in application "CRM", template "Leads" '
        "to change display_format to PhoneRuMask. Provide Intent, Plan, Validate checklist (risk notes), and a DRY-RUN payload preview. "
        "Do NOT execute changes yet—await my approval."
    ),
    "quick_list_apps_message": (
        "List all applications in the platform. Format nicely using Markdown."
    ),
    # Query example buttons (converted from try_asking_examples)
    "quick_edit_enum": "📝 Edit Enum",
    "quick_edit_enum_message": 'Get the enum attribute "Status" from application "CRM", template "Leads", then add a new value "In Progress" (system_name: in_progress, color: #FF9800) and update the attribute',
    "quick_templates_erp": "📄 ERP Templates",
    "quick_templates_erp_message": 'Show all record templates in the "ERP" application. Format nicely using Markdown.',
    "quick_attributes_contractors": "🏷️ Contractor Attributes",
    "quick_attributes_contractors_message": 'Get a list of all attributes of the "Counterparties" template, application "ERP"',
    "quick_create_comment_attr": "💬 Create Comment Attribute",
    "quick_create_comment_attr_message": 'Create a text attribute "Comment", application "HR", template "Candidates"',
    "quick_create_id_attr": "🆔 Create ID Attribute",
    "quick_create_id_attr_message": 'Create a text attribute "Customer ID", application "ERP", template "Counterparties", special input mask: ([0-9]{10}|[0-9]{12})',
    "quick_edit_phone_mask": "📞 Edit Phone Mask",
    "quick_edit_phone_mask_message": 'For the "Contact Phone" attribute in application "CRM", template "Leads", change the display format to Russian phone',
    "quick_get_comment_attr": "🔍 Get Comment Attribute",
    "quick_get_comment_attr_message": 'Get attribute: system name "Comment", application "HR", template "Candidates"',
    "quick_edit_date_time": "📅 Configure Date/Time",
    "quick_edit_date_time_message": 'Create a date/time attribute "Lead Creation Date" in application "CRM", template "Leads" with LongDateLongTime display format and use it as record title for automatic time-based sorting',
    "quick_archive_attr": "📦 Archive Attribute",
    "quick_archive_attr_message": 'Archive/unarchive attribute, system name "Comment", application "HR", template "Candidates"',
    "quick_what_can_do": "❓ What can you do?",
    "quick_what_can_do_message": "What can you do?",
    "quick_what_cannot_do": "❌ What can't you do?",
    "quick_what_cannot_do_message": "What can't you do?",
    "quick_full_audit_message": "Give a full audit of all applications, templates and attributes in the system.",
    # Status messages
    "processing_complete": "🎉 Processing complete",
    "response_completed": "Response completed",
    "processing_failed": "Processing failed",
    # Initialization messages
    "session_manager_ready": "Session manager ready",
    # Iteration messages
    "iteration_processing": "Iteration **{iteration}/{max_iterations}** - Processing...",
    "iteration_finished": "Iteration **{iteration}/{max_iterations}** - Finished",
    "iteration_completed": "Iteration **{iteration}** completed - Continuing...",
    "iteration_max_reached": "Iteration **{iteration}/{max_iterations}** - Finished (max reached)",
    "max_iterations_warning": "⚠️ Reached iteration limit **({max_iterations})**, conversation may be incomplete",
    # Tool messages
    "tool_called": "🔧 Tool called: {tool_name}",
    "call_count": "Call count: {total_calls}",
    "result": "**Result:** {tool_result}",
    "tool_error": "❌ **Tool error: {error}**",
    "unknown_tool": "❌ **Unknown tool: {tool_name}**",
    "tools_available": "🔧 Tools available: {count}",
    "tool_calls_made": "Tool calls made: {tool_names}",
    # Error messages
    "error": "❌ **Error: {error}**",
    # Provider availability messages
    "no_providers_available": "❌ No providers available",
    "no_models_available": "❌ No models available",
    "error_loading_providers": "❌ Error loading providers",
    # History compression
    "history_compression_title": "📦 Conversation History Compressed",
    "history_compression_info": "Saved ~{tokens_saved:,} tokens. Usage: {previous_pct:.1f}% → {current_pct:.1f}%. Compressions: {compression_count}.",
    "history_compression_info_before": "Compressing conversation... ({previous_pct:.1f}% used)",
    "history_compression_reason_critical": "Critical token usage (≥{threshold}%)",
    "history_compression_reason_proactive": "Compressing conversation to prevent overflow",
    "history_compression_reason_interrupted": "Turn interrupted with critical status",
    "compression_stats_label": "Compression Stats",
    "compression_count_label": "Compressions: {count}",
    "compression_tokens_saved_label": "Tokens saved: {tokens:,}",
    # UI Icons
    "clock_icons": [
        "🕐",
        "🕑",
        "🕒",
        "🕓",
        "🕔",
        "🕕",
        "🕖",
        "🕗",
        "🕘",
        "🕙",
        "🕚",
        "🕛",
    ],
    "finish_icons": ["🎉", "🏁", "✨", "🎯"],
    "completion_icons": ["✅", "✔️", "🎯", "✨"],
    "max_icons": ["⚠️", "⏰", "🔄", "⚡"],
    "completion_final_icons": ["✅", "🎯", "✨", "🏆"],
    "error_icons": ["❌", "💥", "⚠️", "🚫"],
}


def create_i18n_instance() -> gr.I18n:
    """
    Create a Gradio I18n instance with translations for all supported languages.

    Returns:
        Gradio I18n instance with both English and Russian translations
    """
    return gr.I18n(en=ENGLISH_TRANSLATIONS, ru=RUSSIAN_TRANSLATIONS)


def get_translation_key(key: str, language: str = "en") -> str:
    """
    Get a translation for a specific key in the specified language.

    Args:
        key: Translation key
        language: Language code ('en' or 'ru')

    Returns:
        Translated string
    """
    # Safety check for None key
    if key is None:
        return "Unknown"

    # Safety check for None language
    if language is None:
        language = "en"

    if language.lower() == "ru":
        return RUSSIAN_TRANSLATIONS.get(key, ENGLISH_TRANSLATIONS.get(key, key))
    return ENGLISH_TRANSLATIONS.get(key, key)


def format_translation(key: str, language: str = "en", **kwargs: Any) -> str:
    """
    Get a formatted translation for a specific key with variable substitution.

    Args:
        key: Translation key
        language: Language code ('en' or 'ru')
        **kwargs: Variables to substitute in the translation

    Returns:
        Formatted translated string
    """
    # Safety check for None key
    if key is None:
        return "Unknown"

    # Safety check for None language
    if language is None:
        language = "en"

    template = get_translation_key(key, language)
    try:
        return template.format(**kwargs)
    except KeyError as e:
        print(
            f"Warning: Missing format variable {e} for key '{key}' in language '{language}'"
        )
        return template
