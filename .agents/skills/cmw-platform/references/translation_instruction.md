# Инструкция по переводу UI-строк

Полное руководство по локализации пользовательского интерфейса приложений Comindware Platform.

## Введение

Перевод UI-строк отличается от локализации алиасов:

| Алиасы | UI-строки |
|--------|-----------|
| Системные имена объектов | Пользовательский интерфейс |
| Переименовываются на платформе | Изменяются в JSON-файлах |
| Требуют API-запросов | Автономная работа с файлами |
| Опасно (_calc в выражениях) | Безопасно |

**Что переводится:**
- Названия шаблонов (Template Name)
- Названия атрибутов (Attribute Names)
- Подписи форм (Form Labels)
- Названия столбцов数据集 (Dataset Columns)
- Названия команд (UserCommands / Buttons)
- Названия панелей инструментов (Toolbar Names)
- Варианты перечислений (Enum Variants)
- Навигация рабочих пространств (Workspace Navigation)
- Названия виджетов (Widget Display Names)
- Названия ролей (Role Names)
- Названия маршрутов (Route Names)

---

## Предварительные требования

1. **Извлечённый CTF** — приложение экспортировано в формат CTF и извлечено в папку
2. **Подключение к платформе** — настроенные переменные окружения (CMW_BASE_URL, CMW_TOKEN)
3. **CSV-справочник** — файл справочника локализации для добавления новых терминов

---

## Скрипты перевода

### 1. harvest_strings.py — Извлечение строк

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
| `--exclude` | Папки для исключения | — |

**Обрабатываемые поля:**

| Поле | Пример |
|------|--------|
| Name | "Здания" → "Buildings" |
| DisplayName | "Основная форма" → "Main Form" |
| Text | "Нажмите кнопку" → "Click button" |
| Description | "Описание объекта" → "Object description" |
| Title | "Заголовок окна" → "Window title" |
| Header | "Заголовок блока" → "Block header" |
| Tooltip | "Подсказка" → "Tooltip" |
| Label | "Метка поля" → "Field label" |
| Caption | "Подпись" → "Caption" |
| Value | Значение по умолчанию |
| Content | "Содержимое" → "Content" |
| Summary | "Итог" → "Summary" |
| Ru | Русский вариант перечисления |
| En | Английский вариант перечисления |

**Выходной файл:** `harvested.json` — JSON с массивом строк

```json
{
  "strings": {
    "Здания": {
      "count": 5,
      "files": ["RecordTemplates/Zdaniya/Attributes/Name.json", ...]
    },
    "Название": {...}
  }
}
```

**Пример:**
```bash
# Извлечение из папки RecordTemplates
python .agents/skills/cmw-platform/scripts/harvest_strings.py \
    "/tmp/cmw-transfer/Volga/RecordTemplates" \
    --output harvested.json

# Извлечение с исключением
python .agents/skills/cmw-platform/scripts/harvest_strings.py \
    "/tmp/cmw-transfer/Volga" \
    --output harvested.json \
    --exclude "SystemFolders,Temp"
```

---

### 2. build_translations.py — Создание шаблона переводов

Создаёт JSON-файл с русскими строками в качестве ключей и пустыми значениями.

```bash
python .agents/skills/cmw-platform/scripts/build_translations.py \
    harvested.json \
    --output translations.json
```

**Параметры:**

| Параметр | Описание | По умолчанию |
|----------|----------|---------------|
| `harvested` | Файл с извлечёнными строками | harvested.json |
| `--output`, `-o` | Выходной файл | translations.json |

**Выходной файл:** `translations.json`

```json
{
  "Здания": "",
  "Название": "",
  "Создать": "",
  ...
}
```

**После заполнения:**
```json
{
  "Здания": "Buildings",
  "Название": "Name",
  "Создать": "Create",
  ...
}
```

---

### 3. apply_translations.py — Применение переводов

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
| `--backup` | Создать резервные копии | False |

**Процесс работы:**

1. Загружает translations.json
2. Для каждого JSON-файла в папке:
   - Находит поля для перевода
   - Заменяет русские строки на английские
   - Записывает изменения
3. Выводит статистику изменений

**Примеры:**

```bash
# Только показать изменения (dry-run)
python .agents/skills/cmw-platform/scripts/apply_translations.py \
    "/tmp/cmw-transfer/Volga/RecordTemplates" \
    translations.json \
    --dry-run

# Применить переводы с подробным выводом
python .agents/skills/cmw-platform/scripts/apply_translations.py \
    "/tmp/cmw-transfer/Volga/RecordTemplates" \
    translations.json \
    --verbose

# Применить с резервным копированием
python .agents/skills/cmw-platform/scripts/apply_translations.py \
    "/tmp/cmw-transfer/Volga/RecordTemplates" \
    translations.json \
    --backup
```

**Выходная статистика:**

```
=== Translation Summary ===
Files processed: 45
Files changed: 12
Strings translated: 87
Skipped (system): 23
```

---

### 4. update_csv.py — Обновление CSV-справочника

Добавляет новые термины в CSV-файл справочника локализации.

```bash
python .agents/skills/cmw-platform/scripts/update_csv.py \
    translations.json \
    translations.csv
```

**Параметры:**

| Параметр | Описание | Обязательно |
|----------|----------|-------------|
| `translations` | Файл с переводами | Да |
| `csv` | CSV-справочник | Да |

**Формат CSV:**

```
исходное название (RU);Системное имя (RU);Английское название (EN);Системное имя (EN);Исходный JSON-Path
Здания;Zdaniya;Buildings;Buildings;Volga/RecordTemplates/Zdaniya/Name
Название;Name;Name;Name;Volga/RecordTemplates/Zdaniya/Attributes/Name/Name
```

**Пример:**
```bash
python .agents/skills/cmw-platform/scripts/update_csv.py \
    translations.json \
    translations.csv
```

---

## Рабочий процесс (5 шагов)

### Шаг 1: Извлечение строк

```bash
python .agents/skills/cmw-platform/scripts/harvest_strings.py \
    "/tmp/cmw-transfer/Volga/RecordTemplates" \
    --output harvested.json
```

### Шаг 2: Создание шаблона

```bash
python .agents/skills/cmw-platform/scripts/build_translations.py \
    harvested.json \
    --output translations.json
```

### Шаг 3: Редактирование переводов

Откройте `translations.json` в редакторе и заполните пустые значения:

```json
{
  "Здания": "Buildings",
  "Название": "Name",
  "Статус": "Status",
  "Свободно": "Vacant",
  "Занято": "Occupied"
}
```

### Шаг 4: Применение переводов

```bash
# Сначала проверить (dry-run)
python .agents/skills/cmw-platform/scripts/apply_translations.py \
    "/tmp/cmw-transfer/Volga/RecordTemplates" \
    translations.json \
    --dry-run

# Применить
python .agents/skills/cmw-platform/scripts/apply_translations.py \
    "/tmp/cmw-transfer/Volga/RecordTemplates" \
    translations.json
```

### Шаг 5: Обновление CSV

```bash
python .agents/skills/cmw-platform/scripts/update_csv.py \
    translations.json \
    translations.csv
```

---

## Типы полей для перевода

### Основные поля

| Поле | Где используется | Пример |
|------|-----------------|--------|
| Name | Шаблоны, атрибуты, столбцы | "Здания" → "Buildings" |
| DisplayName | Формы, поля, кнопки | "Основная форма" → "Main Form" |
| Text | Подписи, описания | "Нажмите кнопку" → "Click button" |
| Description | Справка, документация | "Описание объекта" → "Object description" |

### Дополнительные поля

| Поле | Где используется | Пример |
|------|-----------------|--------|
| Title | Заголовки окон, секций | "Заголовок" → "Title" |
| Header | Заголовки блоков | "Блок 1" → "Block 1" |
| Tooltip | Всплывающие подсказки | "Наведите для справки" → "Hover for help" |
| Label | Метки полей ввода | "Введите имя" → "Enter name" |
| Caption | Подписи элементов | "Рисунок 1" → "Figure 1" |
| Value | Значения по умолчанию | "По умолчанию" → "Default" |
| Content | Содержимое контента | "Текстовое содержимое" → "Text content" |
| Summary | Итоговые значения | "Итого" → "Total" |

### Поля перечислений

| Поле | Описание | Пример |
|------|----------|--------|
| Ru | Русский вариант | "Свободно" |
| En | Английский вариант | "Vacant" |

---

## Рекомендации

### 1. Всегда используйте --dry-run

Перед применением переводов всегда запускайте с флагом --dry-run:

```bash
python .agents/skills/cmw-platform/scripts/apply_translations.py \
    "/path/to/folder" \
    translations.json \
    --dry-run
```

Это покажет все изменения без фактической записи.

### 2. Создавайте резервные копии

Перед массовыми изменениями создавайте резервные копии:

```bash
python .agents/skills/cmw-platform/scripts/apply_translations.py \
    "/path/to/folder" \
    translations.json \
    --backup
```

Создаёт `.backup` копии изменённых файлов.

### 3. Обновляйте CSV после каждого сеанса

После завершения перевода обязательно обновляйте справочник:

```bash
python .agents/skills/cmw-platform/scripts/update_csv.py \
    translations.json \
    translations.csv
```

### 4. Проверяйте системные строки

Некоторые строки НЕ нужно переводить:

- Системные алиасы (Alias, GlobalAlias)
- JSON-пути и ссылки на свойства
- Выражения в вычисляемых полях
- GUID/UUID идентификаторы
- Математические формулы

### 5. Разбивайте на этапы

Для больших приложений разбивайте перевод по папкам:

```bash
# Отдельно RecordTemplates
python .agents/skills/cmw-platform/scripts/harvest_strings.py \
    "/tmp/cmw-transfer/Volga/RecordTemplates" \
    --output harvested_record.json

# Отдельно Workspaces
python .agents/skills/cmw-platform/scripts/harvest_strings.py \
    "/tmp/cmw-transfer/Volga/Workspaces" \
    --output harvested_workspace.json
```

---

## Полный пример

### Исходные данные

- Приложение: Volga
- CTF извлечён в: `/tmp/cmw-transfer/Volga`
- CSV-справочник: `/tmp/cmw-transfer/translations.csv`

### Пошаговое руководство

#### 1. Извлечение строк из RecordTemplates

```bash
python .agents/skills/cmw-platform/scripts/harvest_strings.py \
    "/tmp/cmw-transfer/Volga/RecordTemplates" \
    --output harvested_volga_record.json
```

Результат: найдено 245 русских строк

#### 2. Создание шаблона

```bash
python .agents/skills/cmw-platform/scripts/build_translations.py \
    harvested_volga_record.json \
    --output translations_volga.json
```

Результат: создан файл с 245 пустыми значениями

#### 3. Заполнение переводов

Отредактируйте `translations_volga.json`:

```json
{
  "Здания": "Buildings",
  "Помещения": "Rooms",
  "Название": "Name",
  "Адрес": "Address",
  "Статус": "Status",
  "Свободно": "Vacant",
  "Занято": "Occupied",
  "Создать": "Create",
  "Редактировать": "Edit",
  "Удалить": "Delete"
}
```

#### 4. Проверка (dry-run)

```bash
python .agents/skills/cmw-platform/scripts/apply_translations.py \
    "/tmp/cmw-transfer/Volga/RecordTemplates" \
    translations_volga.json \
    --dry-run
```

Результат: изменений в 12 файлах, 45 строк

#### 5. Применение переводов

```bash
python .agents/skills/cmw-platform/scripts/apply_translations.py \
    "/tmp/cmw-transfer/Volga/RecordTemplates" \
    translations_volga.json \
    --backup
```

Результат: изменено 12 файлов, 45 строк переведено

#### 6. Обновление CSV

```bash
python .agents/skills/cmw-platform/scripts/update_csv.py \
    translations_volga.json \
    /tmp/cmw-transfer/translations.csv
```

Результат: добавлено 10 новых терминов в справочник

#### 7. Повторение для других папок

```bash
# Workspaces
python .agents/skills/cmw-platform/scripts/harvest_strings.py \
    "/tmp/cmw-transfer/Volga/Workspaces" \
    --output harvested_volga_ws.json
python .agents/skills/cmw-platform/scripts/build_translations.py \
    harvested_volga_ws.json \
    --output translations_volga_ws.json
# ... редактирование и применение ...

# Forms
python .agents/skills/cmw-platform/scripts/harvest_strings.py \
    "/tmp/cmw-transfer/Volga/Forms" \
    --output harvested_volga_forms.json
# ... и так далее
```

---

## Устранение проблем

### Проблема: Строки не переводятся

**Причина:** Ключи в translations.json не совпадают точно

**Решение:** Проверьте, что ключи точно совпадают (включая пробелы)

```bash
# Показать уникальные ключи
python -c "import json; d=json.load(open('harvested.json')); print(list(d['strings'].keys())[:10])"
```

### Проблема: JSON-файлы не изменяются

**Причина:** Файлы защищены от записи или путь неверен

**Решение:** Проверьте путь и права доступа

```bash
ls -la "/tmp/cmw-transfer/Volga/RecordTemplates"
```

### Проблема: Слишком много системных строк

**Причина:** harvest_strings извлекает всё подряд

**Решение:** Используйте --exclude для пропуска системных папок

```bash
python .agents/skills/cmw-platform/scripts/harvest_strings.py \
    "/path/to/folder" \
    --output harvested.json \
    --exclude "SystemFolders,Temp,_backup"
```

---

## Команды для быстрого запуска

### Базовый перевод одной папки

```bash
python .agents/skills/cmw-platform/scripts/harvest_strings.py "$1" --output harvested.json && \
python .agents/skills/cmw-platform/scripts/build_translations.py harvested.json -o translations.json && \
# редактирование translations.json && \
python .agents/skills/cmw-platform/scripts/apply_translations.py "$1" translations.json --dry-run && \
python .agents/skills/cmw-platform/scripts/apply_translations.py "$1" translations.json && \
python .agents/skills/cmw-platform/scripts/update_csv.py translations.json translations.csv
```

### Перевод всего приложения

```bash
for folder in RecordTemplates Workspaces Forms Datasets; do
    echo "=== Processing $folder ==="
    python .agents/skills/cmw-platform/scripts/harvest_strings.py \
        "/tmp/cmw-transfer/Volga/$folder" \
        --output "harvested_$folder.json"
    python .agents/skills/cmw-platform/scripts/build_translations.py \
        "harvested_$folder.json" \
        -o "translations_$folder.json"
    # редактирование translations_$folder.json
    python .agents/skills/cmw-platform/scripts/apply_translations.py \
        "/tmp/cmw-transfer/Volga/$folder" \
        "translations_$folder.json"
done
```

---

## Справочная таблица

| Скрипт | Назначение | Ключевые параметры |
|--------|------------|-------------------|
| harvest_strings.py | Извлечение строк | `input_folder`, `--output` |
| build_translations.py | Шаблон переводов | `harvested`, `--output` |
| apply_translations.py | Применение переводов | `folder`, `translations`, `--dry-run` |
| update_csv.py | Обновление справочника | `translations`, `csv` |

---

*Обновлено: 2026-04-30*