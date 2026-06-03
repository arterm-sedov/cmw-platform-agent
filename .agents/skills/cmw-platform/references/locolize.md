# CSV Format

```
исходное название (RU);Системное имя (RU);Английское название (EN);Системное имя (EN);Исходный JSON-Path
```

---

## Локализация через localization.py

### Предварительные требования

1. Приложение экспортировано в CTF
2. CTF извлечён в папку
3. Выполнены шаги 1-4 (извлечение, сбор, верификация, поиск опасных)

### Правила перевода алиасов
Не переводить алиас, если "aliasLocked": true

### Запуск перевода алиасов (step 5)

```bash
python .agents/skills/cmw-platform/scripts/localization.py \
    --app Volga \
    --step 5
```

### Интерактивный процесс

Скрипт последовательно показывает каждый алиас и отображаемое название и запрашивает ввод:

```
=== Interactive Translation ===
Total aliases: 45
Type new aliasRenamed (Enter to skip), displayNameRenamed (Enter to skip)
Commands: 'q' quit, 's' save, 'r <alias>' resume from alias

[1/45] type=RecordTemplate
  aliasOriginal: Zdaniya
  displayNameOriginal: Здания
  jsonPathOriginal: RecordTemplates/Zdaniya/Name
  expressions count: 2
  aliasRenamed [Zdaniya_calc]: Buildings
  displayNameRenamed [Здания]: Buildings
  [Saved] aliasRenamed=Buildings
```

**Ввод данных:**

| Поле | Описание | Пример ввода |
|------|----------|--------------|
| aliasRenamed | Новое системное имя | `Buildings` или Enter для авто |
| displayNameRenamed | Новое отображаемое имя | `Buildings` или Enter для оригинала |

**Логика автозаполнения:**
- Если ввод пустой для aliasRenamed → используется `aliasOriginal + _calc` (если есть выражения) или `aliasOriginal + _sv`
- Если ввод пустой для displayNameRenamed → используется `displayNameOriginal`

### Продолжение перевода (resume)

**С конкретного алиаса:**
```bash
python .agents/skills/cmw-platform/scripts/localization.py \
    --app Volga \
    --step 5 \
    --resume Zdaniya
```

**После прерывания:**
```bash
python .agents/skills/cmw-platform/scripts/localization.py \
    --app Volga \
    --step 5 \
    --resume
```

# Пример запуска перевода
python localization.py --app Volga --step 5

```

---
