#!/usr/bin/env python3
"""Harvest all user-facing strings from Comindware JSON files."""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import OrderedDict


def extract_strings_from_file(filepath):
    """Extract all translatable strings from a JSON file."""
    strings_found = []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    patterns = [
        (r'"Name"\s*:\s*"([^"]+)"', 'Name'),
        (r'"DisplayName"\s*:\s*"([^"]+)"', 'DisplayName'),
        (r'"Text"\s*:\s*"([^"]+)"', 'Text'),
        (r'"Description"\s*:\s*"([^"]+)"', 'Description'),
        (r'"MessageName"\s*:\s*"([^"]+)"', 'MessageName'),
        (r'"Title"\s*:\s*"([^"]+)"', 'Title'),
        (r'"Header"\s*:\s*"([^"]+)"', 'Header'),
        (r'"Tooltip"\s*:\s*"([^"]+)"', 'Tooltip'),
        (r'"Placeholder"\s*:\s*"([^"]+)"', 'Placeholder'),
        (r'"Label"\s*:\s*"([^"]+)"', 'Label'),
        (r'"Caption"\s*:\s*"([^"]+)"', 'Caption'),
        (r'"Value"\s*:\s*"([^"]+)"', 'Value'),
        (r'"Content"\s*:\s*"([^"]+)"', 'Content'),
        (r'"Summary"\s*:\s*"([^"]+)"', 'Summary'),
        (r'"Ru"\s*:\s*"([^"]+)"', 'Ru'),
        (r'"En"\s*:\s*"([^"]+)"', 'En'),
    ]

    for pattern, field_type in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            if re.match(r'^[a-zA-Z0-9_-]+$', match):
                continue
            if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}', match, re.I):
                continue
            if len(match) < 2:
                continue
            if any('\u0400' <= c <= '\u04FF' for c in match):
                strings_found.append({
                    'string': match,
                    'field': field_type,
                    'file': str(filepath)
                })

    return strings_found


def harvest_folder(folder_path, output_file=None):
    """Harvest all strings from all JSON files in folder."""
    path = Path(folder_path)
    all_strings = OrderedDict()

    for json_file in path.rglob("*.json"):
        strings = extract_strings_from_file(json_file)
        for s in strings:
            key = s['string']
            if key not in all_strings:
                all_strings[key] = s

    result = {
        "metadata": {
            "source": str(folder_path),
            "count": len(all_strings)
        },
        "strings": {
            russian: {
                "field": data['field'],
                "path": data['file']
            }
            for russian, data in sorted(all_strings.items())
        }
    }

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Harvested {len(all_strings)} unique strings to {output_file}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return all_strings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harvest translatable strings from Comindware JSON files")
    parser.add_argument("folder", nargs="?", default=".", help="Folder to search for JSON files")
    parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()
    harvest_folder(args.folder, args.output)
