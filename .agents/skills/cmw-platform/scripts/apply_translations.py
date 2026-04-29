#!/usr/bin/env python3
"""Apply translations to Comindware JSON files."""

import argparse
import json
import re
from pathlib import Path


SYSTEM_ALIASES = {'Alias', 'GlobalAlias', 'Name', 'DisplayName', 'Text', 'Description'}


def is_system_string(s):
    """Check if string looks like a system identifier."""
    if not s:
        return True
    if re.match(r'^[a-zA-Z0-9_-]+$', s):
        return True
    if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}', s, re.I):
        return True
    if s.startswith('${') or s.startswith('{{'):
        return True
    return False


def apply_translations(folder_path, translations_dict, dry_run=True, verbose=False):
    """Apply translations to all JSON files in folder."""
    path = Path(folder_path)
    changed_files = []
    skipped_system = 0
    translated_count = 0

    if isinstance(translations_dict, str):
        with open(translations_dict, 'r', encoding='utf-8') as f:
            translations_dict = json.load(f)

    translations_stripped = {k.strip(): v for k, v in translations_dict.items()}

    fields = [
        'Name', 'DisplayName', 'Text', 'Description', 'MessageName',
        'Title', 'Header', 'Tooltip', 'Placeholder', 'Label',
        'Caption', 'Value', 'Content', 'Summary'
    ]

    for json_file in path.rglob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            content = f.read()

        original = content

        for field in fields:
            pattern = rf'("{field}"\s*:\s*")([^"]+)(")'

            def make_replace_func(field_name):
                def replace_func(m):
                    prefix = m.group(1)
                    russian = m.group(2)
                    suffix = m.group(3)
                    if is_system_string(russian.strip()):
                        return m.group(0)
                    stripped = russian.strip()
                    english = translations_stripped.get(stripped)
                    if english:
                        return f'{prefix}{english}{suffix}'
                    return m.group(0)
                return replace_func

            content = re.sub(pattern, make_replace_func(field), content)

        ru_pattern = r'("Ru"\s*:\s*")([^"]+)(")'
        content = re.sub(ru_pattern, lambda m: f'{m.group(1)}{translations_stripped.get(m.group(2).strip(), m.group(2))}{m.group(3)}', content)

        en_pattern = r'("En"\s*:\s*")([^"]+)(")'
        content = re.sub(en_pattern, lambda m: f'{m.group(1)}{translations_stripped.get(m.group(2).strip(), m.group(2))}{m.group(3)}', content)

        if content != original:
            if not dry_run:
                with open(json_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            changed_files.append(str(json_file))

    return changed_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply translations to Comindware JSON files")
    parser.add_argument("folder", help="Folder containing JSON files")
    parser.add_argument("translations", help="Path to translations.json")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output")

    args = parser.parse_args()

    changed = apply_translations(args.folder, args.translations, args.dry_run, args.verbose)
    action = "Would change" if args.dry_run else "Changed"
    print(f"{action} {len(changed)} files")
    for f in changed[:20]:
        print(f"  - {f}")
    if len(changed) > 20:
        print(f"  ... and {len(changed) - 20} more")
