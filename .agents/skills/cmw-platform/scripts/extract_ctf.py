#!/usr/bin/env python3
"""
Extract CTF (zip archive) to folder.

Usage:
    python extract_ctf.py --input C:\\tmp\\cmw-transfer\\Volga.ctf --output C:\\tmp\\cmw-transfer\\Volga_ctf
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path


def main(input_ctf: str, output_dir: str) -> int:
    """Extract CTF archive to folder."""
    if not os.path.exists(input_ctf):
        print(f"Error: CTF file not found: {input_ctf}")
        return 1

    os.makedirs(output_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(input_ctf, "r") as zf:
            file_count = 0
            for member in zf.infolist():
                if member.is_dir():
                    continue
                zf.extract(member, output_dir)
                file_count += 1
    except zipfile.BadZipFile:
        print(f"Error: Not a valid zip file: {input_ctf}")
        return 1
    except Exception as e:
        print(f"Error: Failed to extract: {e}")
        return 1

    print(f"Extracted {file_count} files to: {output_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CTF archive to folder")
    parser.add_argument("--input", required=True, help="Path to CTF file")
    parser.add_argument("--output", required=True, help="Path to output directory")
    args = parser.parse_args()
    sys.exit(main(args.input, args.output))