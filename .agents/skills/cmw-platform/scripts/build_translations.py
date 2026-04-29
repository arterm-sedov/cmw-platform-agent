#!/usr/bin/env python3
"""Build translation dictionary from harvested strings."""

import argparse
import json
import sys
from pathlib import Path


def build_dictionary(harvested_file, output_file="translations.json"):
    """Convert harvested JSON to translation dictionary template."""
    with open(harvested_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    translations = {russian: "" for russian in data.get("strings", {})}

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translations, f, ensure_ascii=False, indent=2)

    print(f"Created {output_file} with {len(translations)} entries")
    print("Edit the JSON to add English translations, then run apply_translations.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build translation dictionary from harvested strings")
    parser.add_argument("harvested", nargs="?", default="harvested.txt", help="Harvested strings file")
    parser.add_argument("--output", "-o", default="translations.json", help="Output JSON file")

    args = parser.parse_args()
    build_dictionary(args.harvested, args.output)
