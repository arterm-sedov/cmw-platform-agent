# Инструкция по командам локализации

Полное руководство по локализации приложений Comindware Platform.

## Введение

Локализация приложений включает два основных процесса:

1. **Локализация алиасов** — переименование системных имён с русского на английский
2. **Локализация UI-строк** — перевод пользовательского интерфейса с русского на английский

---

## 1. Основной инструмент — tool_localize.py

**Расположение:** `tools/localization_tools/tool_localize.py`

### Параметры

| Параметр | Описание | По умолчанию |
|----------|----------|---------------|
| `application_system_name` | Системное имя приложения | — |
| `json_folder` | Путь к папке с JSON-файлами для анализа | — |
| `output_dir` | Путь для сохранения выходных файлов | json_folder |
| `create_tr` | Создать копию _tr из оригинала | False |
| `translate_one` | Перевести один алиас | None |
| `resume` | Продолжить с последнего переведённого алиаса | False |
| `apply_renames` | Применить переименования на платформе | False |
| `fix_expressions` | Исправить алиасы _calc в выражениях | False |
| `dry_run` | Только анализ без изменений | True |
| `dangerous_suffix` | Суффикс для опасных имён | "_calc" |
| `safe_suffix` | Суффикс для безопасных имён | "_sv" |

### Полный рабочий процесс

1. Извлечение алиасов (`tool_extract_aliases.py`)
2. Сбор данных платформы (`tool_collect_platform.py`)
3. Верификация алиасов (`tool_verify_aliases.py` по папкам)
4. Поиск опасных алиасов (`tool_find_dangerous.py`)
5. Финализация (`tool_finalize.py`)
6. Конвертация в формат схемы
7. `--create-tr`: Создать копию _tr, исправить _calc в выражениях
8. `--translate-one`: Перевести один алиас, сохранить состояние
9. `--resume`: Продолжить с последнего алиаса
10. `--apply-renames`: Переименовать на платформе
11. `--fix-expressions`: Исправить алиасы _calc в выражениях

---

## 2. Вспомогательные скрипты (CLI)

### 2.1 tool_extract_aliases.py — Извлечение алиасов

Извлекает алиасы из CTF JSON-файлов по папкам. Поддерживает возобновление после прерывания.

```bash
python .agents/skills/cmw-platform/scripts/tool_extract_aliases.py \
    --app Volga \
    --extract-dir /path/to/ctf \
    --output-dir /path/to/output
```

**Параметры:**

| Параметр | Описание | Обязательно |
|----------|----------|-------------|
| `--app` | Системное имя приложения | Да |
| `--extract-dir` | Директория с извлечённым CTF | Нет |
| `--output-dir` | Директория для вывода | Нет |

**Выходные файлы:** `{app}_{folder}_aliases.json` для каждой папки

---

### 2.2 tool_collect_platform.py — Сбор данных платформы

Запрашивает все типы объектов с платформы параллельно (8 потоков по умолчанию).

```bash
python .agents/skills/cmw-platform/scripts/tool_collect_platform.py \
    --app Volga \
    --output-dir /path/to/output \
    --workers 8
```

**Параметры:**

| Параметр | Описание | По умолчанию |
|----------|----------|---------------|
| `--app` | Системное имя приложения | — |
| `--output-dir` | Директория для вывода | /tmp/cmw-transfer/{app}_tr |
| `--workers` | Количество параллельных потоков | 8 |

**Выходной файл:** `{app}_platform_cache.json`

---

### 2.3 tool_verify_aliases.py — Верификация алиасов

Проверяет ID алиасов через API платформы. Запускается отдельно для каждой папки.

```bash
python .agents/skills/cmw-platform/scripts/tool_verify_aliases.py \
    --app Volga \
    --folder RecordTemplates \
    --output-dir /path/to/output
```

**Параметры:**

| Параметр | Описание | Обязательно |
|----------|----------|-------------|
| `--app` | Системное имя приложения | Да |
| `--folder` | Имя папки для верификации | Да |
| `--output-dir` | Директория для вывода | Нет |

**Выходной файл:** `{app}_{folder}_verified.json`

---

### 2.4 tool_find_dangerous.py — Поиск опасных алиасов

Анализирует выражения в атрибутах для поиска алиасов, используемых в вычисляемых полях.

```bash
python .agents/skills/cmw-platform/scripts/tool_find_dangerous.py \
    --app Volga \
    --extract-dir /path/to/ctf \
    --output-dir /path/to/output \
    --workers 4
```

**Параметры:**

| Параметр | Описание | По умолчанию |
|----------|----------|---------------|
| `--app` | Системное имя приложения | — |
| `--extract-dir` | Директория с извлечённым CTF | /tmp/cmw-transfer/{app} |
| `--output-dir` | Директория для вывода | /tmp/cmw-transfer/{app}_tr |
| `--workers` | Количество параллельных потоков | 4 |

**Выходной файл:** `{app}_dangerous_aliases.json`

---

### 2.5 tool_finalize.py — Финализация

Объединяет все файлы верификации в единый выходной файл в формате схемы.

```bash
python .agents/skills/cmw-platform/scripts/tool_finalize.py \
    --app Volga \
    --output-dir /path/to/output
```

**Параметры:**

| Параметр | Описание | Обязательно |
|----------|----------|-------------|
| `--app` | Системное имя приложения | Да |
| `--output-dir` | Директория для вывода | Нет |

**Выходной файл:** `{app}_verified_complete.json`

Также автоматически загружает алиасы приложения из `{app}_Application_aliases.json` (создаётся tool_extract_aliases.py).

---

### 2.6 apply_renames.py — Применение переименований

Применяет переименования алиасов на платформе через API.

```bash
python .agents/skills/cmw-platform/scripts/apply_renames.py \
    --app Volga \
    --output-dir /path/to/output
```

**Параметры:**

| Параметр | Описание | Обязательно |
|----------|----------|-------------|
| `--app` | Системное имя приложения | Да |
| `--output-dir` | Директория с {app}_verified_complete.json | Нет |
| `--reverse` | Откатить переименования (renamed → original) | Нет |

---

### 2.7 harvest_strings.py — Извлечение строк

Извлекает все русские строки из JSON-файлов для перевода.

```bash
python .agents/skills/cmw-platform/scripts/harvest_strings.py \
    "path/to/Workspaces" \
    --output harvested.json
```

**Параметры:**

| Параметр | Описание | По умолчанию |
|----------|----------|---------------|
| `input_folder` | Папка с JSON-файлами | — |
| `--output` | Выходной файл | harvested.json |

**Обрабатываемые поля:** Name, DisplayName, Text, Description, Title, Header, Tooltip, Label, Caption, Value, Content, Summary, Ru, En

---

### 2.8 build_translations.py — Создание шаблона переводов

Создаёт JSON-файл с русскими строками в качестве ключей и пустыми значениями для заполнения.

```bash
python .agents/skills/cmw-platform/scripts/build_translations.py \
    harvested.json \
    --output translations.json
```

**Параметры:**

| Параметр | Описание | По умолчанию |
|----------|----------|---------------|
| `harvested` | Файл с извлечёнными строками | harvested.txt |
| `--output`, `-o` | Выходной файл | translations.json |

---

### 2.9 apply_translations.py — Применение переводов

Заменяет русские строки на английские в JSON-файлах.

```bash
python .agents/skills/cmw-platform/scripts/apply_translations.py \
    "path/to/Workspaces" \
    translations.json \
    --dry-run
```

**Параметры:**

| Параметр | Описание | По умолчанию |
|----------|----------|---------------|
| `folder` | Папка с JSON-файлами | — |
| `translations` | Файл с переводами | translations.json |
| `--dry-run` | Показать изменения без записи | False |
| `--verbose` | Подробный вывод | False |

---

### 2.10 update_csv.py — Обновление CSV-справочника

Добавляет новые термины в CSV-файл справочника локализации.

```bash
python .agents/skills/cmw-platform/scripts/update_csv.py \
    translations.json \
    translations.csv
```

---

### 2.11 localization.py — Автоматизация всего процесса

Полная автоматизация локализации алиасов от экспорта до импорта CTF.

```bash
python .agents/skills/cmw-platform/scripts/localization.py \
    --app Volga \
    --step N \
    --resume
```

**Параметры:**

| Параметр | Описание |
|----------|----------|
| `--app` | Системное имя приложения |
| `--step N` | Выполнить конкретный шаг (0-10) |
| `--resume` | Продолжить с прерванного места |

**Шаги:**

0. Экспорт CTF с датой (yyyyddMM-HHmmss)
1. Сбор алиасов из JSON
2. Верификация ID через API
3. Анализ выражений для опасных алиасов (_calc)
4. Создание _tr копии
5. Интерактивный перевод с сохранением состояния
6. Применение переименований
7. Запрос перезагрузки платформы
8. Повторный экспорт CTF
9. Исправление выражений с _calc алиасами
10. Импорт изменённого CTF

---

## 3. Выходные файлы

| Файл | Описание |
|------|----------|
| `{app}_{folder}_aliases.json` | Извлечённые алиасы по папкам |
| `{app}_platform_cache.json` | Кэш данных платформы |
| `{app}_{folder}_verified.json` | Верифицированные алиасы по папкам |
| `{app}_dangerous_aliases.json` | Алиасы с выражениями |
| `{app}_verified_complete.json` | Финальный объединённый результат |
| `{domain}_{app}_aliases.json` | Выходной файл в формате схемы |
| `{domain}_{app}_aliases_tr.json` | Копия для перевода |
| `{app}_localize_resume.json` | Файл состояния для продолжения |
| `{app}_Application_aliases.json` | Алиас приложения (из `{app}.json` в корне CTF) |

---

## 4. Пример полного рабочего процесса

### Этап 1: Локализация алиасов

```bash
# 1. Извлечь CTF и разместить в /tmp/cmw-transfer/Volga
# 2. Запустить основной инструмент
python agent_ng/app_ng_modular.py

# Ввести в чат:
localize_aliases application_system_name="Volga" json_folder="/tmp/cmw-transfer/Volga" dry_run=false
```

### Через CLI скрипты:

```bash
# Извлечение алиасов
python .agents/skills/cmw-platform/scripts/tool_extract_aliases.py \
    --app Volga \
    --extract-dir /tmp/cmw-transfer \
    --output-dir /tmp/cmw-transfer/Volga_tr

# Сбор данных платформы
python .agents/skills/cmw-platform/scripts/tool_collect_platform.py \
    --app Volga \
    --output-dir /tmp/cmw-transfer/Volga_tr

# Верификация для каждой папки
for folder in RecordTemplates Forms Datasets; do
    python .agents/skills/cmw-platform/scripts/tool_verify_aliases.py \
        --app Volga \
        --folder $folder \
        --output-dir /tmp/cmw-transfer/Volga_tr
done

# Поиск опасных алиасов
python .agents/skills/cmw-platform/scripts/tool_find_dangerous.py \
    --app Volga \
    --extract-dir /tmp/cmw-transfer \
    --output-dir /tmp/cmw-transfer/Volga_tr

# Финализация
python .agents/skills/cmw-platform/scripts/tool_finalize.py \
    --app Volga \
    --output-dir /tmp/cmw-transfer/Volga_tr
```

### Этап 2: Создание _tr копии и перевод

```bash
# Через Gradio tool:
localize_aliases application_system_name="Volga" json_folder="/tmp/cmw-transfer/Volga" create_tr=true

# Интерактивный перевод
localize_aliases application_system_name="Volga" json_folder="/tmp/cmw-transfer/Volga" translate_one="Schetchiki"

# Продолжение перевода
localize_aliases application_system_name="Volga" json_folder="/tmp/cmw-transfer/Volga" resume=true

# Применение переименований
localize_aliases application_system_name="Volga" json_folder="/tmp/cmw-transfer/Volga" apply_renames=true
```

### Этап 3: Локализация UI-строк

```bash
# Извлечение строк
python .agents/skills/cmw-platform/scripts/harvest_strings.py \
    "/tmp/cmw-transfer/Volga/RecordTemplates" \
    --output harvested.json

# Создание шаблона
python .agents/skills/cmw-platform/scripts/build_translations.py \
    harvested.json \
    --output translations.json

# РЕДАКТИРОВАТЬ translations.json (добавить английские переводы)

# Применение переводов
python .agents/skills/cmw-platform/scripts/apply_translations.py \
    "/tmp/cmw-transfer/Volga/RecordTemplates" \
    translations.json

# Обновление CSV
python .agents/skills/cmw-platform/scripts/update_csv.py \
    translations.json \
    translations.csv
```

---

## 5. Особенности типа Application

- **Извлечение:** Алиас приложения извлекается из `{app}.json` в корне CTF
- **Блокировка:** Всегда заблокирован (`aliasLocked: true`) — не переименовывается
- **ID-префикс:** `sln.` (например, `sln.23`)
- **Предикаты платформы:**
  - `cmw.solution.alias` — системное имя
  - `cmw.solution.name` — отображаемое имя

---

## 6. Типы объектов и их папки

| Тип | Папка CTF | Предикат |
|-----|-----------|----------|
| RecordTemplate | RecordTemplates | cmw.container.alias |
| ProcessTemplate | ProcessTemplates | cmw.container.alias |
| Workspace | Workspaces | cmw.alias |
| Page | Pages | cmw.desktopPage.alias |
| Attribute | Attributes | cmw.object.alias |
| Dataset | Datasets | cmw.alias |
| Form | Forms | cmw.alias |
| Toolbar | Toolbars | cmw.alias |
| UserCommand | UserCommands | cmw.alias |
| WidgetConfig | WidgetConfigs | cmw.form.alias |
| Role | Roles | cmw.role.alias |
| Trigger | Triggers | cmw.trigger.alias |
| Application | корень CTF | cmw.solution.alias |

---

## 7. Рекомендации

1. **Всегда используйте `--dry-run true`** для первого запуска
2. **Регулярно сохраняйте состояние** с помощью `--resume`
3. **Опасайтесь алиасов с `_calc` суффиксом** — они используются в выражениях
4. **Application алиас всегда заблокирован** — не пытайтесь переименовать
5. **Обновляйте CSV** после каждого сеанса перевода UI-строк

---

*Обновлено: 2026-04-30*