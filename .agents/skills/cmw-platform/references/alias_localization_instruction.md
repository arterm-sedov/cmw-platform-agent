# Инструкция по переименованию алиасов

Полное руководство по локализации системных имён (алиасов) приложений Comindware Platform.

## Введение

### Что такое алиасы?

Алиасы (system names) — это системные идентификаторы объектов в Comindware Platform:

- Имена шаблонов (RecordTemplate, ProcessTemplate)
- Имена атрибутов (Attribute)
- Имена форм (Form),数据集 (Dataset), кнопок (UserCommand)
- Имена рабочих пространств (Workspace), ролей (Role)
- Имена приложений (Application)

**Примеры алиасов:**
- `Zdaniya` (здания — шаблон)
- `Naimenovanie` (название — атрибут)
- `SpisokZdanij` (список зданий —数据集)
- `Sozdat` (создать — кнопка)
- `Volga` (приложение)

### Зачем переименовывать?

Русские алиасы затрудняют:
- Интеграцию с внешними системами
- Разработку и поддержку
- Использование английскими пользователями

**Цель:** Переименование с русского на английский:
- `Zdaniya` → `Buildings`
- `Naimenovanie` → `Name`
- `SpisokZdanij` → `BuildingsList`

---

## Основные понятия

### 1. aliasOriginal

Исходное имя алиаса в том виде, в котором оно существует в CTF и на платформе.

```json
{
  "aliasOriginal": "Zdaniya"
}
```

### 2. aliasRenamed

Новое имя алиаса после перевода. Может быть:
- Пустым (если ещё не переведено)
- С суффиксом `_calc` (опасный — используется в выражениях)
- С суффиксом `_sv` (безопасный)
- Любым другим именем

```json
{
  "aliasOriginal": "Zdaniya",
  "aliasRenamed": "Buildings"
}
```

### 3. aliasLocked

Флаг блокировки переименования:

| Значение | Описание | Поведение |
|----------|----------|-----------|
| `true` | Заблокировано | Алиас НЕ будет переименован на платформе |
| `false` | Разблокировано | Алиас будет переименован при --apply-renames |

**Когда устанавливается в true:**
- Есть отображаемое имя (displayName)
- Это алиас приложения (Application)
- Объект является системным

```json
{
  "aliasOriginal": "Zdaniya",
  "aliasLocked": true,
  "displayNameOriginal": "Здания"
}
```

### 4. displayNameOriginal / displayNameRenamed

Отображаемое имя (название в интерфейсе):

```json
{
  "displayNameOriginal": "Здания",
  "displayNameRenamed": "Buildings"
}
```

### 5. jsonPathOriginal / jsonPathRenamed

Пути к файлам в CTF, где встречается алиас:

```json
{
  "jsonPathOriginal": [
    "Volga/RecordTemplates/Zdaniya/Name",
    "Volga/RecordTemplates/Zdaniya/Attributes/Naimenovanie/Name",
    "Volga/RecordTemplates/Zdaniya/Datasets/SpisokZdanij/Name"
  ]
}
```

### 6. expressions

Выражения, использующие данный алиас. Если алиас используется в вычисляемых полях, он считается **опасным**.

```json
{
  "aliasOriginal": "Ploshchad",
  "expressions": [
    {
      "jsonPathOriginal": "Volga/RecordTemplates/Kvartira/Attributes/Ploshchad/Calculation",
      "expressionOriginal": "${Zdaniya.Ploshchad} * ${Koef}",
      "expressionRenamed": "Buildings_Ploshchad * Koef"
    }
  ]
}
```

### 7. dangerous_suffix / safe_suffix

Суффиксы для автоматического переименования:

| Суффикс | Значение | Пример |
|----------|----------|--------|
| `dangerous_suffix` | Опасный (в выражениях) | `_calc` → `Ploshchad_calc` |
| `safe_suffix` | Безопасный | `_sv` → `Naimenovanie_sv` |

**Логика:**
- Если есть expressions → используется dangerous_suffix
- Если нет expressions → используется safe_suffix

---

## Типы алиасов по опасности

### Dangerous (опасные)

Алиасы, которые используются в выражениях вычисляемых атрибутов. При переименовании необходимо обновить ВСЕ выражения.

**Пример:**
```
Alias: Ploshchad
Expression: ${Zdaniya.Ploshchad} * ${Koef}
```

**При переименовании:**
```
aliasOriginal: Ploshchad
aliasRenamed: Ploshchad_calc
expressionOriginal: ${Zdaniya.Ploshchad} * ${Koef}
expressionRenamed: ${Zdaniya_Ploshchad_calc} * Koef
```

### Safe (безопасные)

Алиасы, которые НЕ используются в выражениях. Их можно переименовать безопасно.

**Пример:**
```
Alias: Naimenovanie
No expressions
```

---

## Инструмент tool_localize.py

**Расположение:** `tools/localization_tools/tool_localize.py`

Это основной Gradio-инструмент для локализации алиасов.

### Параметры

#### Обязательные параметры

| Параметр | Тип | Описание |
|----------|-----|----------|
| `application_system_name` | str | Системное имя приложения (например, "Volga") |
| `json_folder` | str | Путь к папке с извлечённым CTF |

#### Дополнительные параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|---------------|----------|
| `output_dir` | str | json_folder | Путь для сохранения выходных файлов |
| `dry_run` | bool | true | Если true — только анализ, без изменений |
| `create_tr` | bool | false | Создать копию _tr из оригинала |
| `translate_one` | str | None | Перевести один конкретный алиас |
| `resume` | bool | false | Продолжить с последнего переведённого алиаса |
| `apply_renames` | bool | false | Применить переименования на платформе |
| `fix_expressions` | bool | false | Исправить алиасы _calc в выражениях |
| `dangerous_suffix` | str | "_calc" | Суффикс для опасных алиасов |
| `safe_suffix` | str | "_sv" | Суффикс для безопасных алиасов |

---

## Режимы работы

### 1. Режим анализа (dry_run: true)

Только анализ без фактических изменений:

```
localize_aliases
  application_system_name="Volga"
  json_folder="/tmp/cmw-transfer/Volga"
  dry_run=true
```

**Результат:**
- Извлечение алиасов
- Сбор данных платформы
- Верификация
- Поиск опасных
- Финализация
- НО создание aliasRenamed НЕ производится

### 2. Режим создания _tr копии (create_tr: true)

Создаёт копию для перевода с автоматическим заполнением aliasRenamed:

```
localize_aliases
  application_system_name="Volga"
  json_folder="/tmp/cmw-transfer/Volga"
  create_tr=true
```

**Что делает:**
1. Создаёт файл `{domain}_Volga_aliases_tr.json`
2. Для каждого алиаса:
   - Если есть expressions → добавляет dangerous_suffix
   - Если нет expressions → добавляет safe_suffix

**Пример:**
```json
{
  "aliasOriginal": "Naimenovanie",
  "aliasRenamed": "Naimenovanie_sv",
  "aliasLocked": false
}
```

### 3. Режим перевода одного алиаса (translate_one)

Интерактивный перевод конкретного алиаса:

```
localize_aliases
  application_system_name="Volga"
  json_folder="/tmp/cmw-transfer/Volga"
  translate_one="Zdaniya"
```

**Процесс:**
1. Показывает информацию об алиасе (тип, ID, displayName, выражения)
2. Запрашивает новый aliasRenamed
3. Сохраняет состояние в `{app}_localize_resume.json`

### 4. Режим продолжения (resume: true)

Продолжает перевод с последнего сохранённого алиаса:

```
localize_aliases
  application_system_name="Volga"
  json_folder="/tmp/cmw-transfer/Volga"
  resume=true
```

### 5. Режим применения (apply_renames: true)

Применяет переименования на платформе через API:

```
localize_aliases
  application_system_name="Volga"
  json_folder="/tmp/cmw-transfer/Volga"
  apply_renames=true
```

**Что делает:**
1. Читает файл `{domain}_Volga_aliases_tr.json`
2. Для каждого объекта с aliasRenamed и aliasLocked=false:
3. Отправляет API-запрос на переименование
4. Обновляет статус (success/failed)

### 6. Режим исправления выражений (fix_expressions: true)

Исправляет алиасы в выражениях после переименования:

```
localize_aliases
  application_system_name="Volga"
  json_folder="/tmp/cmw-transfer/Volga"
  fix_expressions=true
```

---

## Рабочий процесс (8 этапов)

### Этап 1: Извлечение алиасов из CTF

**Скрипт:** `tool_extract_aliases.py`

```bash
python .agents/skills/cmw-platform/scripts/tool_extract_aliases.py \
    --app Volga \
    --extract-dir /tmp/cmw-transfer \
    --output-dir /tmp/cmw-transfer/Volga_tr
```

**Что делает:**
1. Сканирует все JSON-файлы в папках CTF
2. Находит все GlobalAlias, Container, Template
3. Извлекает: alias, type, displayName, path
4. Создаёт файл `{app}_{folder}_aliases.json` для каждой папки
5. Извлекает алиас приложения из `{app}.json`

**Выходные файлы:**
- `{app}_RecordTemplates_aliases.json`
- `{app}_Forms_aliases.json`
- `{app}_Application_aliases.json`
- `{app}_extraction_state.json` (состояние)

### Этап 2: Сбор данных платформы

**Скрипт:** `tool_collect_platform.py`

```bash
python .agents/skills/cmw-platform/scripts/tool_collect_platform.py \
    --app Volga \
    --output-dir /tmp/cmw-transfer/Volga_tr \
    --workers 8
```

**Что делает:**
1. Запрашивает все типы объектов с платформы параллельно
2. Собирает ID и системные имена
3. Сохраняет в кэш

**Выходной файл:** `{app}_platform_cache.json`

**Зачем нужно:** Сопоставление алиасов из CTF с реальными ID на платформе для верификации.

### Этап 3: Верификация алиасов

**Скрипт:** `tool_verify_aliases.py`

```bash
python .agents/skills/cmw-platform/scripts/tool_verify_aliases.py \
    --app Volga \
    --folder RecordTemplates \
    --output-dir /tmp/cmw-transfer/Volga_tr
```

**Что делает:**
1. Для каждого алиаса проверяет существование на платформе
2. Разделяет на:
   - **verified** — существующие алиасы
   - **skipped** — не найденные (нет ID)

**Выходной файл:** `{app}_{folder}_verified.json`

**Зачем нужно:** Определить, какие алиасы можно переименовать (имеют ID на платформе).

### Этап 4: Поиск опасных алиасов

**Скрипт:** `tool_find_dangerous.py`

```bash
python .agents/skills/cmw-platform/scripts/tool_find_dangerous.py \
    --app Volga \
    --extract-dir /tmp/cmw-transfer \
    --output-dir /tmp/cmw-transfer/Volga_tr \
    --workers 4
```

**Что делает:**
1. Сканирует атрибуты с типами Calculation, DefaultExpression, ValidationScript
2. Находит все алиасы, используемые в выражениях
3. Сохраняет список опасных алиасов и их выражения

**Выходной файл:** `{app}_dangerous_aliases.json`

**Зачем нужно:** Определить, какие алиасы требуют осторожности (_calc).

### Этап 5: Финализация

**Скрипт:** `tool_finalize.py`

```bash
python .agents/skills/cmw-platform/scripts/tool_finalize.py \
    --app Volga \
    --output-dir /tmp/cmw-transfer/Volga_tr
```

**Что делает:**
1. Объединяет все файлы верификации
2. Применяет правила блокировки:
   - Application → всегда заблокирован
   - С displayName → заблокирован
   - Dangerous → разблокирован (нужно переименовать)
3. Создаёт единый файл в формате схемы

**Выходной файл:** `{app}_verified_complete.json`

**Формат:**
```json
[
  {
    "type": "RecordTemplate",
    "ids": ["oa.123"],
    "aliasOriginal": "Zdaniya",
    "aliasRenamed": "",
    "displayNameOriginal": "Здания",
    "displayNameRenamed": "",
    "aliasLocked": true,
    "jsonPathOriginal": ["Volga/RecordTemplates/Zdaniya/Name"],
    "expressions": []
  }
]
```

### Этап 6: Создание _tr копии

```bash
localize_aliases
  application_system_name="Volga"
  json_folder="/tmp/cmw-transfer/Volga"
  create_tr=true
```

**Что делает:**
1. Создаёт копию с суффиксами:
   - dangerous_suffix (_calc) для алиасов с expressions
   - safe_suffix (_sv) для остальных

**Выходной файл:** `{domain}_Volga_aliases_tr.json`

### Этап 7: Интерактивный перевод

**Вариант А: Один алиас**

```bash
localize_aliases
  application_system_name="Volga"
  json_folder="/tmp/cmw-transfer/Volga"
  translate_one="Naimenovanie"
```

**Вариант Б: Продолжение**

```bash
localize_aliases
  application_system_name="Volga"
  json_folder="/tmp/cmw-transfer/Volga"
  resume=true
```

**Процесс:**
1. Показывает информацию об алиасе
2. Запрашивает ввод: aliasRenamed (или Enter для авто)
3. Сохраняет в файл

### Этап 8: Применение переименований

```bash
localize_aliases
  application_system_name="Volga"
  json_folder="/tmp/cmw-transfer/Volga"
  apply_renames=true
```

**Что делает:**
1. Читает _tr файл
2. Для каждого объекта с aliasRenamed и aliasLocked=false:
3. Отправляет API-запрос `update_object_property`
4. Обновляет статус в JSON

---

## Состояния алиасов (aliasLocked)

### aliasLocked: true (заблокировано)

Алиас НЕ будет переименован.

**Когда устанавливается:**
1. **Application** — алиас приложения всегда заблокирован
2. **Есть displayNameOriginal** — если есть отображаемое имя, переименование может нарушить интерфейс
3. **Системные объекты** — системные шаблоны и компоненты

**Пример:**
```json
{
  "type": "Application",
  "aliasOriginal": "Volga",
  "aliasLocked": true,
  "displayNameOriginal": "Волга"
}
```

### aliasLocked: false (разблокировано)

Алиас будет переименован при apply_renames.

**Когда устанавливается:**
1. **Dangerous** — алиас в выражениях, требует переименования
2. **Safe** — алиас без выражений, можно безопасно переименовать

**Пример:**
```json
{
  "type": "RecordTemplate",
  "aliasOriginal": "Naimenovanie",
  "aliasLocked": false,
  "expressions": []
}
```

---

## Тип Application (особенности)

### Что это?

Application — это алиас самого приложения (системное имя решения).

### Извлечение

Алиас приложения извлекается из `{app}.json` в корне CTF:

```json
{
  "cmw.solution.alias": "Volga",
  "cmw.solution.name": "Волга"
}
```

### ID-префикс

Алиасы приложений имеют префикс `sln.`:
- `sln.1`, `sln.23`, `sln.100`

### Предикаты платформы

| Предикат | Описание |
|----------|----------|
| `cmw.solution.alias` | Системное имя приложения |
| `cmw.solution.name` | Отображаемое имя |

### Блокировка

**Application всегда заблокирован** (`aliasLocked: true`)

Это означает, что алиас приложения НЕ будет переименован на платформе. Причина:
- Переименование приложения влияет на всю систему
- Требует дополнительных действий (миграция данных, обновление ссылок)
- Слишком рискованно для автоматического процесса

### Workflow в финализации

При обработке Application применяется специальное правило:

```python
if obj.get("type") == "Application":
    alias_locked = True  # Всегда блокируем
```

Это переопределяет любую другую логику (даже если алиас в выражениях).

---

## Вспомогательные скрипты

### tool_extract_aliases.py

Извлечение алиасов из CTF по папкам.

| Параметр | Описание |
|----------|----------|
| `--app` | Системное имя приложения |
| `--extract-dir` | Директория с CTF |
| `--output-dir` | Выходная директория |

### tool_collect_platform.py

Сбор данных с платформы параллельно.

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--app` | Системное имя приложения | — |
| `--output-dir` | Выходная директория | /tmp/cmw-transfer/{app}_tr |
| `--workers` | Количество потоков | 8 |

### tool_verify_aliases.py

Верификация алиасов по папкам.

| Параметр | Описание |
|----------|----------|
| `--app` | Системное имя приложения |
| `--folder` | Папка для верификации |
| `--output-dir` | Выходная директория |

### tool_find_dangerous.py

Поиск алиасов в выражениях.

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--app` | Системное имя приложения | — |
| `--extract-dir` | Директория с CTF | /tmp/cmw-transfer/{app} |
| `--output-dir` | Выходная директория | /tmp/cmw-transfer/{app}_tr |
| `--workers` | Количество потоков | 4 |

### tool_finalize.py

Финализация и объединение.

| Параметр | Описание |
|----------|----------|
| `--app` | Системное имя приложения |
| `--output-dir` | Выходная директория |

### apply_renames.py

Применение переименований на платформе.

| Параметр | Описание |
|----------|----------|
| `--app` | Системное имя приложения |
| `--output-dir` | Директория с _tr файлом |
| `--reverse` | Откатить переименования |

---

## Выходные файлы

| Файл | Описание |
|------|----------|
| `{app}_{folder}_aliases.json` | Извлечённые алиасы по папкам |
| `{app}_Application_aliases.json` | Алиас приложения (всегда заблокирован) |
| `{app}_platform_cache.json` | Кэш данных платформы |
| `{app}_{folder}_verified.json` | Верифицированные алиасы |
| `{app}_dangerous_aliases.json` | Опасные алиасы с выражениями |
| `{app}_verified_complete.json` | Финальный объединённый файл |
| `{domain}_{app}_aliases.json` | В формате схемы (основной) |
| `{domain}_{app}_aliases_tr.json` | Копия для перевода |
| `{app}_localize_resume.json` | Состояние для resume |
| `{app}_extraction_state.json` | Состояние извлечения |

---

## Пример рабочего процесса

### Подготовка

```bash
# 1. Экспорт приложения в CTF
# (через Gradio tool export_application или UI платформы)

# 2. Извлечение CTF в папку
unzip Volga.ctf -d /tmp/cmw-transfer/Volga
```

### Основной процесс

```bash
# Запуск через Gradio tool
localize_aliases
  application_system_name="Volga"
  json_folder="/tmp/cmw-transfer/Volga"
  dry_run=false
  create_tr=true
```

**Этапы (автоматически):**
1. Извлечение алиасов → создание файлов по папкам
2. Сбор платформы → platform_cache.json
3. Верификация → verified файлы по папкам
4. Поиск опасных → dangerous_aliases.json
5. Финализация → verified_complete.json
6. Создание _tr → aliases_tr.json

### Интерактивный перевод (опционально)

```bash
# Перевод одного алиаса
localize_aliases
  application_system_name="Volga"
  json_folder="/tmp/cmw-transfer/Volga"
  translate_one="Naimenovanie"

# Или продолжение
localize_aliases
  application_system_name="Volga"
  json_folder="/tmp/cmw-transfer/Volga"
  resume=true
```

### Применение

```bash
# Применение переименований
localize_aliases
  application_system_name="Volga"
  json_folder="/tmp/cmw-transfer/Volga"
  apply_renames=true
```

### Исправление выражений

```bash
# После переименования на платформе
localize_aliases
  application_system_name="Volga"
  json_folder="/tmp/cmw-transfer/Volga"
  fix_expressions=true
```

---

## Рекомендации

### 1. Всегда начинайте с dry_run

```bash
localize_aliases
  application_system_name="Volga"
  json_folder="/tmp/cmw-transfer/Volga"
  dry_run=true
```

Покажет статистику без изменений.

### 2. Проверяйте опасные алиасы

После `tool_find_dangerous.py` проверьте файл `{app}_dangerous_aliases.json`:

```bash
cat /tmp/cmw-transfer/Volga_tr/Volga_dangerous_aliases.json
```

Убедитесь, что все выражения корректны.

### 3. Используйте resume

После прерывания продолжайте с места остановки:

```bash
localize_aliases
  application_system_name="Volga"
  json_folder="/tmp/cmw-transfer/Volga"
  resume=true
```

### 4. Не переименовывайте Application

Алиас приложения всегда заблокирован. Это правильно.

### 5. Создавайте резервные копии

Перед apply_renames экспортируйте CTF:

```bash
export_application application_system_name="Volga"
```

### 6. Запрашивайте перезагрузку платформы

После массовых переименований платформа может требовать перезагрузки (зависит от конфигурации).

---

## Справочная таблица

| Понятие | Описание |
|---------|----------|
| aliasOriginal | Исходное имя алиаса |
| aliasRenamed | Новое имя алиаса |
| aliasLocked | Флаг блокировки (true/false) |
| displayNameOriginal | Исходное отображаемое имя |
| displayNameRenamed | Новое отображаемое имя |
| jsonPathOriginal | Пути в CTF (где встречается) |
| expressions | Выражения с использованием алиаса |
| dangerous_suffix | Суффикс для опасных (_calc) |
| safe_suffix | Суффикс для безопасных (_sv) |

| Режим | Описание |
|-------|----------|
| dry_run | Только анализ |
| create_tr | Создать копию с суффиксами |
| translate_one | Перевести один алиас |
| resume | Продолжить перевод |
| apply_renames | Применить на платформе |
| fix_expressions | Исправить выражения |

---

*Обновлено: 2026-04-30*