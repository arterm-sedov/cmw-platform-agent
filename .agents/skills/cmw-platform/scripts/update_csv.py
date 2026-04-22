#!/usr/bin/env python3
"""Update localization CSV with new translations."""

import argparse
import csv
import json
import sys
from pathlib import Path


def update_csv(translations_json, csv_path, json_paths=None):
    """Append new translations to localization CSV.

    Args:
        translations_json: Path to translations.json with {ru: en} mappings
        csv_path: Path to localization CSV
        json_paths: Optional dict mapping Russian strings to their JSON paths
    """
    with open(translations_json, 'r', encoding='utf-8') as f:
        translations = json.load(f)

    existing_ru = set()
    if Path(csv_path).exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader, None)
            for row in reader:
                if len(row) >= 1:
                    existing_ru.add(row[0])

    new_entries = []
    for russian, english in translations.items():
        if russian not in existing_ru and english:
            path = json_paths.get(russian, "") if json_paths else ""
            new_entries.append({
                'ru': russian,
                'system_ru': '',
                'en': english,
                'system_en': '',
                'path': path
            })

    if new_entries:
        with open(csv_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            for entry in new_entries:
                writer.writerow([
                    entry['ru'],
                    entry['system_ru'],
                    entry['en'],
                    entry['system_en'],
                    entry['path']
                ])
        print(f"Added {len(new_entries)} new entries to {csv_path}")
    else:
        print("No new entries to add")

    return new_entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update localization CSV with new translations")
    parser.add_argument("translations", help="Path to translations.json")
    parser.add_argument("csv", help="Path to localization CSV")

    args = parser.parse_args()
    update_csv(args.translations, args.csv)
